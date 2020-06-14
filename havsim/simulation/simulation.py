"""Houses the main code for running simulations.

A simulation is defined by a collection of lanes/roads (a road network) and an initial
collection of vehicles. The road network defines both the network topology (i.e. how roads connect
with each other) as well as the inflow/outflow boundary conditions, which determine how
vehicles enter/leave the simulation. The inflow conditions additionally control what types of
vehicles enter the simulation. Vehicles are implemented in the Vehicle class and a road network
is made up of instances of the Lane class.
"""

import math
import numpy as np
import scipy.optimize as sc
import havsim.simulation.models as hm


def update_net(vehicles, lc_actions, inflow_lanes, merge_lanes, vehid, timeind, dt):
    """Updates all quantities for a road network.

    After evaluating all vehicle's longitudinal and latitudinal actions, this function does the rest
    of the updates. In order:
        completeing requested lane changes
            -moving vehicle to new lane and setting its new route/lane events
            -updating all leader/follower relationships (including l/rfol) for all vehicles involved
            -possibly applying relaxation
        updating state and memory
        updating all leader/follower relationships (also updates merge anchors)
        updating all vehicles lane/route events (including possibly removing vehicles)
        updating inflow conditions for all lanes with inflow (including possibly adding new vehicles and
        generating the parameters for the next new vehicle)

    Args:
        vehicles: set containing all vehicle objects being actively simulated
        lc_actions: dictionary with keys as vehicles which request lane changes in the current timestep,
            values are a string either 'l' or 'r' which indicates the side of the change
        inflow_lanes: iterable of lanes which have upstream (inflow) boundary conditions. These lanes
            must have get_inflow, increment_inflow, and new_vehicle methods
        merge_lanes: iterable of lanes which have merge anchors. These lanes must have a merge_anchors
            attribute
        vehid: int giving the unique vehicle ID for the next vehicle to be generated
        timeind: int giving the timestep of the simulation (0 indexed)
        dt: float of time unit that passes in each timestep

    Returns:
        vehid: updated int value of next vehicle ID to be added
        remove_vehicles: set of vehicles which were removed from simulation at current timestep

        Modifies vehicles in place (potentially all their non-parameter/bounds attributes, e.g. pos/spd,
        lead/fol relationships, their lane/roads, memory, etc.).
    """
    # update followers/leaders for all lane changes
    for veh in lc_actions.keys():
        oldfol = veh.fol
        # update leader follower relationships, lane/road
        update_change(lc_actions, veh, timeind)  # this cannot be done in parralel

        # apply relaxation
        new_relaxation(veh, timeind, dt)
        newfol = veh.fol
        if newfol.cf_parameters is None:
            pass
        else:
            new_relaxation(newfol, timeind, dt)
        if oldfol.cf_parameters is None:
            pass
        else:
            new_relaxation(oldfol, timeind, dt)

        # update a vehicle's lane events and route events for the new lane
        set_lane_events(veh)
        set_route_events(veh)

    # update all states, memory and headway
    for veh in vehicles:
        veh.update(timeind, dt)
    for veh in vehicles:
        if veh.lead is not None:
            veh.hd = veh.lane.get_headway(veh, veh.lead)
        # else:  # for robustness only, should not be needed
        #     veh.hd = None

    # update merge_anchors
    for curlane in merge_lanes:
        update_merge_anchors(curlane, lc_actions)

    # update roads (lane events) and routes
    remove_vehicles = set()
    for veh in vehicles:
        # check vehicle's lane events and route events, acting if necessary
        update_lane_events(veh, timeind, remove_vehicles)
        update_route(veh)
    # remove vehicles which leave
    for veh in remove_vehicles:
        vehicles.remove(veh)

    # update left and right followers
    update_all_lrfol(vehicles)
    # update_all_lrfol_multiple(vehicles)

    for veh in vehicles:  # debugging
        if not veh._chk_leadfol(verbose = False):
            # print('-------- Report for Vehicle '+str(veh.vehid)+' at time '+str(timeind)+'--------')
            # veh._leadfol()
            veh._chk_leadfol()
            # raise ValueError('incorrect vehicle order')

    # update inflow, adding vehicles if necessary
    for curlane in inflow_lanes:
        vehid = curlane.increment_inflow(vehicles, vehid, timeind, dt)

    return vehid, remove_vehicles


def update_lane_events(veh, timeind, remove_vehicles):
    """Check if the next event from a Vehicle's lane_events should be applied, and apply it if so.

    lane_events are a list of events which handle anything related to the network topology,
    i.e. when the current lane ends, or when the current lane's left or right connections change.
    Each event is a dictionary with the keys of
    'pos': the float position the event occurs (relative to the vehicle's current lane)
    'event': one of
        'new lane' - occurs when a vehicle reaches the end of its current lane and transitions to a new lane
        'update lr' - occurs when the current lane's left or right connections change
        'exit' - occurs when a vehicle reaches the end of its current lane and exits the road network
    'left': for 'new lane' or 'update lr', if the left connection changes, 'left' needs a value of either
        'add' if there is a new left lane, or 'remove' if the current left connection is no longer possible
    'left anchor': if 'left' is 'add', 'left anchor' is an index giving the merge anchor for the
        new left lane
    'right': same as left, for right side
    'right anchor': same as left anchor, for right side

    Args:
        veh: Vehicle object to update
        timeind: int giving the timestep of the simulation (0 indexed)
        remove_vehicles: set of vehicles which will be removed from simulation at current timestep

    Returns:
        None. (Modifies Vehicle attributes in place, adds to remove_vehicles in place.)
    """
    if not veh.lane_events:
        return
    curevent = veh.lane_events[0]
    if veh.pos > curevent['pos']:
        if curevent['event'] == 'new lane':
            # update lane/road/position
            newlane = veh.lane.connect_to
            update_veh_lane(veh, veh.lane, newlane, timeind+1)

            # updates left and right connections
            update_lane_lr(veh, newlane, curevent)

            # enter new road/lane -> need new lane/route events
            set_lane_events(veh)
            set_route_events(veh)

        elif curevent['event'] == 'update lr':
            update_lane_lr(veh, veh.lane, curevent)
            veh.lane_events.pop(0)  # event is over, so we shouldn't check it in the future

        elif curevent['event'] == 'exit':
            fol = veh.fol
            # need to check l/rlead for edge case when you overtake and exit in same timestep
            for i in veh.llead:
                i.rfol = fol
                fol.llead.add(i)
            for i in veh.rlead:
                i.lfol = fol
                fol.rlead.add(i)

            # update vehicle orders
            fol.lead = None
            fol.leadmem.append((None, timeind+1))
            if veh.lfol is not None:
                veh.lfol.rlead.remove(veh)
            if veh.rfol is not None:
                veh.rfol.llead.remove(veh)

            # to remove the vehicle set its endtime and put it in the remove_vehicles
            veh.endtime = timeind+1
            remove_vehicles.add(veh)
    return


def update_lane_lr(veh, curlane, curevent):
    """Updates a vehicle's attributes when its lane changes its left/right connections.

    For a Vehicle veh which reaches a point where its curlane.get_connect_left or get_connect_right
    go from None to some Lane, or some Lane to None, there needs to be 'add' or 'remove' events for the
    corresponding sides. This handles those events.
    Updates the vehicle orders and defaults the lane change states to the correct behavior (by default,
    enter discretionary only if the left/right lane is in the same road as the current lane)

    Args:
        veh: Vehicle object to update
        curlane: the Lane veh is currently on, curlane has the new/ending connections
        curevent: The event (dictionary) triggering the update

    Returns:
        None (Modifies veh attributes in place.)
    """
    if curevent['left'] == 'remove':
        # handle edge case where veh overtakes lfol in same timestep the left lane ends
        update_lrfol(veh.lfol)
        # update lead/fol order
        veh.lfol.rlead.remove(veh)
        veh.lfol = None
        veh.l_lc = None
        veh.llane = None
        if veh.lc_side == 'l':  # end tactical/coop if necessary
            veh.coop_veh = veh.lc_side = None
    elif curevent['left'] == 'add':
        newllane = curlane.get_connect_left(curevent['pos'])
        merge_anchor = newllane.merge_anchors[curevent['left anchor']][0]
        unused, newfol = curlane.leadfol_find(veh, merge_anchor, 'l')

        veh.lfol = newfol
        newfol.rlead.add(veh)
        if newllane.roadname == curlane.roadname:
            veh.l_lc = 'discretionary'
        else:
            veh.l_lc = None
        veh.llane = newllane

    # same thing for right
    if curevent['right'] == 'remove':

        update_lrfol(veh.rfol)

        veh.rfol.llead.remove(veh)
        veh.rfol = None
        veh.r_lc = None
        veh.rlane = None
        if veh.lc_side == 'r':  # end tactical/coop if necessary
            veh.coop_veh = veh.lc_side = None
    elif curevent['right'] == 'add':
        newrlane = curlane.get_connect_right(curevent['pos'])
        merge_anchor = newrlane.merge_anchors[curevent['right anchor']][0]
        unused, newfol = curlane.leadfol_find(veh, merge_anchor, 'r')

        veh.rfol = newfol
        newfol.llead.add(veh)
        if newrlane.roadname == curlane.roadname:
            veh.r_lc = 'discretionary'
        else:
            veh.r_lc = None
        veh.rlane = newrlane


def set_lane_events(veh):
    """Creates lane_events attribute for Vehicle after entering a new lane.

    Refer to update_lane_events for description of lane events. Note that we only need to add upcoming
    lane events, and past lane events are not applied (this is in contrast to route_events, where past
    events ARE applied.)

    Args:
        veh: Vehicle to be updated

    Returns:
        None (Modifies veh in place.)
    """
    veh.lane_events = []
    for i in veh.lane.events:
        if i['pos'] > veh.pos:
            veh.lane_events.append(i)


def update_route(veh):
    """Check if the next event from a vehicle's route_events should be applied, and apply it if so.

    route_events are a list of events which handles any lane changing behavior related to
    a vehicle's route, i.e. route events ensure that the vehicle follows its route.
    Each event is a dictionary with the keys of
    'pos': the float position the event occurs (relative to the vehicle's current lane).
    'event': 'end discretionary' or 'mandatory', which end discretionary or start mandatory
        lane changing states
    'side': 'l_lc' or 'r_lc' the side which is updated by the event
    'lc_urgency': only for a 'mandatory' event, a tuple giving the position for 0% and 100% forced cooperation

    Args:
        veh: Vehicle object to update

    Returns:
        bool: True if we made a change, to the route, False otherwise
    """
    if not veh.route_events:
        return False
    curevent = veh.route_events[0]
    if veh.pos > curevent['pos']:

        if curevent['event'] == 'end discretionary':
            side = curevent['side']
            setattr(veh, side, None)
            if veh.lc_side == side[0]:  # end tactical/coop if necessary
                veh.coop_veh = veh.lc_side = None

        elif curevent['event'] == 'mandatory':
            setattr(veh, curevent['side'], 'mandatory')
            veh.lc_urgency = curevent['lc_urgency']  # must always set urgency for mandatory changes
        veh.route_events.pop(0)
        return True
    return False


def make_cur_route(p, curlane, nextroadname):
    """Creates cur_route attribute (stores route events) for Vehicle after entering a new lane.

    Refer to update_route for a description of route events.
    Upon entering a new road, we create a cur_route which stores the list of route events for several lanes,
    specifically the lanes we will ultimately end up on, as well as all lanes which we will need to cross
    to reach those lanes we want to be on. We do not create the routes for every single lane on a road.
    Roads have a 'connect to' key whose value is a tuple of:
        pos: for 'continue' change_type, a float which gives the position that the current road
            changes to the desired road.
            for 'merge' type, a tuple of the first position that changing into the desired road
            becomes possible, and the last position where it is still possible to change into that road.
        change_type: if 'continue', this corresponds to the case where the current road turns into
            the next road in the route; the vehicle still needs to make sure it is in the right lane
            (different lanes may transition to different roads)
            if 'merge', this is the situation where the vehicle needs to change lanes onto a different road.
            Thus in the 'merge' case after completing its lane change, the vehicle is on the next desired
            road, in contrast to the continue case where the vehicle actually needs to reach the end of
            the lane in order to transition.
        laneind: if 'continue', a tuple of 2 ints, giving the leftmost and rightmost lanes which will
            continue to the desired lane. if 'merge', the laneind of the lane we need to be on to merge.
        side: for 'merge' type only, gives whether we want to do a left or right change upon reaching
            laneind ('l_lc' or 'r_lc')
        nextroad: desired road

    Explanation of current route model -
    suppose you need to be in lane '2' by position 'x' and start in lane '1', then starting:
        at x - 2*p[0] - 2*p[1] you will end discretionary changing into lane '0'
        at x - p[0] - p[1] you wil begin mandatory changing into lane '2'
        at x - p[0] your mandatory change will have urgency of 100% which will always force
            cooperation of your l/rfol (assuming you have cooperation added to your lc model)
    for lane changing with a merge/diverse (e.g. on/off-ramp) which begins at 'x' and ends at 'y',
    you will start mandatory at 'x' always, reaching 100% cooperation by 'y' - p[0]

    Args:
        p: parameters, length 2 list of floats, where p[0] is a safety buffer for merging and p[1]
            is a comfortable distance for merging
        curlane: Lane object to create route events for
        nextroadname: str name of the next road in the route (the next road you want to be on after leaving
            curlane's road)

    Returns:
        cur_route: dictionary where keys are lanes, value is a list of route event dictionaries which
            defines the route a vehicle with parameters p needs to take on that lane
    """
    # TODO we only get the route for the current road - no look ahead to take into account
    # future roads. This modification may be needed if roads are short.
    # Should only have to look forward one road at a time.

    # TODO handle cases where LC cannot be completed successfully (put necessary info into cur_route dict)
    # if you fail to follow your route, you need to be given a new route.
    # the code which makes a new route can also be used if route = None when creating a vehicle
    # would need to know latest point when change can take place ('pos' for 'continue' type
    # or pos[1] for merge type)
    # in lane changing model, it would need to check if we are getting too close and act accordingly
    # (e.g. slow down) if needed. in this function, would need to add events if you miss the change,
    # and in that case you would need to be given a new route.
    # another option other simulators use is they simply remove a vehicle if it fails to follow its route.

    curroad = curlane.road
    curlaneind = curlane.laneind
    # position or tuple of positions, str, tuple of 2 ints or single int, str, dict for the next road
    pos, change_type, laneind, side, nextroad = curroad['connect to'][nextroadname][:]  # nextroad unused?
    # roads also have 'name', 'len', 'laneinds', all lanes are values with their indexes as keys

    cur_route = {}
    if change_type == 'continue':  # -> vehicle needs to reach end of lane
        # initialize for lanes which vehicle needs to continue on
        leftind, rightind = laneind[:]
        for i in range(leftind, rightind+1):
            cur_route[curroad[i]] = []

        if leftind > 0:
            templane = curroad[leftind]
            curpos = min(templane.end, curroad[leftind-1].end)  # check case where templane.start > curpos?
            # see todo on make_route_helper for edge case
            cur_route[templane].append({'pos': curpos - p[0] - p[1],
                                        'event': 'end discretionary', 'side': 'l_lc'})

        if rightind < curroad['laneinds']-1:
            templane = curroad[rightind]
            curpos = min(templane.end, curroad[rightind+1].end)
            cur_route[templane].append({'pos': curpos - p[0] - p[1],
                                        'event': 'end discretionary', 'side': 'r_lc'})

        if curlaneind >= leftind and curlaneind <= rightind:  # if on correct lane already, do no more work
            return cur_route

        elif curlaneind < laneind[0]:  # need to change right possibly multiple times
            uselaneind = laneind[0]
        else:
            uselaneind = laneind[1]

        cur_route = make_route_helper(p, cur_route, curroad, curlaneind, uselaneind, curroad[uselaneind].end)

    elif change_type == 'merge':  # logic is similar and also uses make_route_helper
        templane = curroad[laneind]
        pos, endpos = pos[:]
        cur_route[templane] = []

        # determine end discretionary event if necessary
        if side == 'l_lc':
            if laneind < curroad['laneinds']-1:
                enddisc = min(pos, curroad[laneind+1].end)
                cur_route[templane].append({'pos': enddisc - p[0] - p[1],
                                            'event': 'end discretionary', 'side': 'r_lc'})
        else:
            if laneind > 0:
                enddisc = min(pos, curroad[laneind-1].end)
                cur_route[templane].append({'pos': enddisc - p[0] - p[1],
                                            'event': 'end discretionary', 'side': 'l_lc'})

        cur_route[templane].append({'pos': pos, 'event': 'mandatory', 'side': side,
                                    'lc_urgency': [pos, endpos - p[0]]})

        if curlaneind != laneind:
            cur_route = make_route_helper(p, cur_route, curroad, curlaneind, laneind, pos)

    return cur_route


def make_route_helper(p, cur_route, curroad, curlaneind, laneind, curpos):
    """Generates list of route events for all lanes with indexes [curlaneind, laneind).

    Starting on curroad in lane with index curlaneind, wanting to be in lane index laneind by position curpos,
    generates route events for all lanes in [curlaneind, laneind). If curlaneind < laneind, starts at
    laneind -1, moving to the left until routes on all lanes are defined. Similarly for curlaneind > laneind.
    Assumes we already have the route for laneind in cur_route.
    Edge cases where routes have different lengths are handled.

    Args:
        p: parameters, length 2 list of floats, where p[0] is a safety buffer for merging and p[1]
            is a comfortable distance for merging
        cur_route: dictionary where keys are lanes, value is a list of route event dictionaries which
            defines the route a vehicle with parameters p needs to take on that lane
        curroad: road that the route is being generated for
        curlaneind: index of the lane that the vehicle starts in
        laneind: index of the lane that we want to be in by position curpos
        curpos: we want to be in lane with index laneind by curpos

    Returns:
        cur_route: Updates cur_route in place
    """
    if curlaneind < laneind:
        curind = laneind - 1
        prevtemplane = curroad[curind+1]
        templane = curroad[curind]
        cur_route[templane] = []
        while not curind < curlaneind:
            # determine curpos = where the mandatory change starts (different meaning than the 'curpos'
            # which is passed in)
            if templane.end < curpos:  # in case templane ends before the curpos
                curpos = templane.end
            curpos += -p[0] - p[1]
            curpos = max(prevtemplane.start, curpos)  # in case the lane doesn't exist at curpos
            # TODO but what about the case where you can actually change before the lane starts?
            # i.e. the case where you have a 3 lane road (indexes, 0, 1, 3) that splits into 2 roads
            # with 2 lanes each, (0,1) and (2,3) indexed, where lane 1 splits into 1 and 2. So in this case,
            # you can only change into 2 once it starts, but really if you change before 2 starts, it's OK
            # because you will just go to 3 which is also fine.
            # maybe should just do something special in the case prevtemplane.start > curpos. In this case,
            # is the end discretionary going to be correct for 1/3? Assume that the indexes do give
            # a correct order but lanes may start/end at their own times.

            # determine enddiscpos = where the discretionary ends
            # only necessary if there is something to end the discretionary into
            if curind > 0:
                nexttemplane = curroad[curind-1]
                enddiscpos = min(curpos, nexttemplane.end)
                enddiscpos = enddiscpos - p[0] - p[1]
                cur_route[templane].append({'pos': enddiscpos, 'event': 'end discretionary', 'side': 'l_lc'})

            # there is always a mandatory event
            cur_route[templane].append({'pos': curpos, 'event': 'mandatory', 'side': 'r_lc',
                                        'urgency': [curpos, curpos + p[1]]})

            # update iteration
            curind += -1
            prevtemplane = templane
            templane = nexttemplane

    # same code but for opposite side
    elif curlaneind > laneind:
        curind = laneind+1
        prevtemplane = curroad[curind - 1]
        templane = curroad[curind]
        cur_route[templane] = []
        while not curind > curlaneind:
            # determine curpos = where the mandatory change starts
            if templane.end < curpos:
                curpos = templane.end
            curpos += -p[0] - p[1]
            curpos = max(prevtemplane.start, curpos)

            if curind < curroad['laneinds'] - 1:
                nexttemplane = curroad[curind + 1]
                enddiscpos = min(curpos, nexttemplane.end)
                enddiscpos = enddiscpos - p[0] - p[1]
                cur_route[templane].append({'pos': enddiscpos, 'event': 'end discretionary', 'side': 'r_lc'})

            cur_route[templane].append({'pos': curpos, 'event': 'mandatory', 'side': 'l_lc',
                                        'urgency': [curpos, curpos + p[1]]})

            # update iteration
            curind += -1
            prevtemplane = templane
            templane = nexttemplane

    return cur_route


def set_route_events(veh):
    """When a vehicle enters a new lane, this function generates all its route events for that lane.

    Every Lane has a list of 'route events' defined for it, which ensure that the Vehicle follows its
    specified route. Refer to update_route for a description of route events, and make_cur_route for
    a description of the route model.
    If a vehicle enters a new road, this function will generate the cur_route for that road and a subset
    of its lanes. This function will pop from the vehicle's route when that occurs. The exception to this
    is when vehicles are first initialized, the initialize method of Vehicle creates the first
    cur_route, and therefore pops from the route the first time.
    If a vehicle enters a new lane on the same road, it will either get the existing route
    from cur_route, or if the route for the new lane does not exist, it will create it add the key/value
    to cur_route. When creating a route for a new lane on the same road, it uses make_route_helper.

    Args:
        veh: Vehicle object which we will set its current route_events for.

    Returns:
        None. Modifies veh attributes in place (route_events, cur_route, possibly applies route events).
    """
    # for testing purposes for infinite road only#########
    # if veh.route == []:
    #     return
    # ######

    # get new route events if they are stored in memory already
    newlane = veh.lane
    if newlane in veh.cur_route:
        veh.route_events = veh.cur_route[newlane].copy()  # route_events store current route events, cur_route
        # stores all route events for subset of lanes on current road
    else:
        p = veh.route_parameters
        prevlane = veh.lanemem[-2][0]
        if prevlane.road is newlane.road:  # on same road - use helper function to update cur_route
            # need to figure out what situation we are in to give make route helper right call
            prevlane_events = veh.cur_route[prevlane]
            if not prevlane_events:  # this can only happen for continue event => curpos = end of lane
                curpos = prevlane.end
            elif prevlane_events[0]['event'] == 'end discretionary':
                curpos = prevlane_events[0]['pos'] + p[0] + p[1]
            else:  # mandatory event
                curpos = prevlane_events[0]['pos']
            make_route_helper(p, veh.cur_route, veh.road, newlane.laneind, prevlane.laneind, curpos)
        else:  # on new road - we need to generate new cur_route and update the vehicle's route
            veh.cur_route = make_cur_route(p, newlane, veh.route.pop(0))

        veh.route_events = veh.cur_route[newlane].copy()

    # for route events, past events need to be applied.
    curbool = True
    while curbool:
        curbool = update_route(veh)


def update_lrfol(veh):
    """After a vehicle's state has been updated, this updates its left and right followers.

    We keep each vehicle's left/right followers updated by doing a single distance compute each timestep.
    The current way of dealing with l/r followers is designed for timesteps which are relatively small.
    (e.g. .1 or .25 seconds) where we also keep l/rfol updated at all timesteps. Other strategies would be
    better for larger timesteps or if l/rfol don't need to always be updated. See block comment
    above update_change to discuss alternative strategies for defining leader/follower relationships.
    See update_change documentation for explanation of naming conventions.
    Called in simulation by update_all_lrfol.
    update_all_lrfol_multiple can be used to handle edge cases so that the vehicle order is always correct,
    but is slightly more computationally expensive and not required in normal situations.
    The edge cases occur when vehicles overtake multiple vehicles in a single timestep.

    Args:
        veh: Vehicle to check its lfol/rfol to be updated.

    Returns:
        None. Modifies veh attributes, attributes of its lfol/rfol, in place.
    """
    lfol, rfol = veh.lfol, veh.rfol
    if lfol is None:
        pass
    elif veh.lane.get_dist(veh, lfol) > 0:
        # update for veh
        veh.lfol = lfol.fol
        veh.lfol.rlead.add(veh)
        lfol.rlead.remove(veh)
        # update for lfol
        lfol.rfol.llead.remove(lfol)
        lfol.rfol = veh
        veh.llead.add(lfol)

    # similarly for right
    if rfol is None:
        pass
    elif veh.lane.get_dist(veh, rfol) > 0:
        veh.rfol = rfol.fol
        veh.rfol.llead.add(veh)
        rfol.llead.remove(veh)

        rfol.lfol.rlead.remove(rfol)
        rfol.lfol = veh
        veh.rlead.add(rfol)


def update_all_lrfol(vehicles):
    """Updates all vehicles left and right followers, without handling multiple overtakes edge cases."""
    for veh in vehicles:
        update_lrfol(veh)


def update_all_lrfol_multiple(vehicles):
    """Handles edge cases where a single vehicle overtakes 2 or more vehicles in a timestep.

    In update_lrfol, when there are multiple overtakes in a single timestep (i.e. an l/rfol passes 2 or more
    l/rlead vehicles, OR a vehicle is overtaken by its lfol, lfol.fol, lfol.fol.fol, etc.) the vehicle order
    can be wrong for the next timestep. For every overtaken vehicle, 1 additional timestep with no overtakes
    is sufficient but not necessary to correct the vehicle order. So if an ego vehicle overtakes 2 vehicles
    on timestep 10, timestep 11 will have an incorrect order. More detail in comments.
    This version is slightly slower than update_all_lrfol but handles the edge cases.
    """
    # edge case 1 - a vehicle veh overtakes 2 llead, llead and llead.lead, in a timestep. If llead.lead
    # has its update_lrfol called first, and then llead, the vehicle order will be wrong because veh will
    # have llead as a rfol. If llead has its update_lrfol called first, the vehicle order will be correct.
    # for the wrong order to be correct, there will need to be 1 timestep where veh does not overtake, so that
    # the llead.lead will see veh overtook it and the order will be corrected. If, instead, veh overtakes
    # llead.lead.lead, then again whether the order will be correct depends on the update order (it will
    # be correct if llead.lead updates first, otherwise it will be behind 1 vehicle like before).

    # edge case 2 - a vehicle veh is overtaken by lfol and lfol.fol in the same timestep. Here veh will have
    # lfol.fol as its lfol. The order will always be corrected in the next timestep, unless another
    # edge case (either 1 or 2) occurs.

    lovertaken = {}  # key with overtaking vehicles as keys, values are a list of vehicles the key overtook
    # lovertaken = left overtaken meaning a vehicles lfol overtook
    rovertaken = {}  # same as lovertaken but for right side
    # first loop we update all vehicles l/rfol and keep track of overtaking vehicles
    for veh in vehicles:
        lfol, rfol = veh.lfol, veh.rfol
        if lfol is None:
            pass
        elif veh.lane.get_dist(veh, lfol) > 0:
            # update for veh
            veh.lfol = lfol.fol
            veh.lfol.rlead.add(veh)
            lfol.rlead.remove(veh)
            # to handle edge case 1 we keep track of vehicles lfol overtakes
            if lovertaken.has_keys('lfol'):
                lovertaken[lfol].append(veh)
            else:
                lovertaken[lfol] = [veh]
            # to handle edge case 2 we update recursively if lfol overtakes
            update_lfol_recursive(veh, lfol.fol, lovertaken)

        # same for right side
        if rfol is None:
            pass
        elif veh.lane.get_dist(veh, rfol) > 0:

            veh.rfol = rfol.fol
            veh.rfol.llead.add(veh)
            rfol.llead.remove(veh)

            if rovertaken.has_keys('rfol'):
                rovertaken[rfol].append(veh)
            else:
                rovertaken[rfol] = [veh]

            update_rfol_recursive(veh, rfol.fol, rovertaken)

    #now to finish the order we have to update all vehicles which overtook
    for lfol, overtook in lovertaken.items():
        if len(overtook) == 1:  # we know what lfol new rfol is - it can only be one thing
            # update for lfol
            veh = overtook[0]
            lfol.rfol.llead.remove(lfol)
            lfol.rfol = veh
            veh.llead.add(lfol)
        else:
            distlist = [veh.lane.get_dist(veh, lfol) for veh in overtook]
            ind = np.argmin(distlist)
            veh = overtook[ind]
            lfol.rfol.llead.remove(lfol)
            lfol.rfol = veh
            veh.llead.add(lfol)

    # same for right side
    for rfol, overtook in rovertaken.items():
        if len(overtook) == 1:  # we know what lfol new rfol is - it can only be one thing
            # update for lfol
            veh = overtook[0]
            rfol.lfol.rlead.remove(rfol)
            rfol.lfol = veh
            veh.rlead.add(rfol)

        else:
            distlist = [veh.lane.get_dist(veh, rfol) for veh in overtook]
            ind = np.argmin(distlist)
            veh = overtook[ind]
            rfol.lfol.rlead.remove(rfol)
            rfol.lfol = veh
            veh.rlead.add(rfol)


def update_lfol_recursive(veh, lfol, lovertaken):
    """Handles edge case 2 for update_all_lrfol_multiple by allowing lfol to update multiple times."""
    if veh.lane.get_dist(veh, lfol) > 0:
        # update for veh
        veh.lfol = lfol.fol
        veh.lfol.rlead.add(veh)
        lfol.rlead.remove(veh)
        # handles edge case 1
        if lovertaken.has_keys('lfol'):
            lovertaken[lfol].append(veh)
        else:
            lovertaken[lfol] = [veh]
        update_lfol_recursive(veh, lfol.fol)


def update_rfol_recursive(veh, rfol, rovertaken):
    """Handles edge case 2 for update_all_lrfol_multiple by allowing rfol to update multiple times."""
    if veh.lane.get_dist(veh, rfol) > 0:

        veh.rfol = rfol.fol
        veh.rfol.llead.add(veh)
        rfol.llead.remove(veh)

        if rovertaken.has_keys('rfol'):
            rovertaken[rfol].append(veh)
        else:
            rovertaken[rfol] = [veh]
        update_rfol_recursive(veh, rfol.fol, rovertaken)


def update_merge_anchors(curlane, lc_actions):
    """Updates merge_anchors attribute for curlane.

    Lanes have lists of merge anchors, they are used as guesses for leadfol_find for 'new lane' or
    'update lanes' events when a left or right lane is added. Thus, merge anchors are used to ensure
    the leader/follower relationships are updated correctly when the network topology changes.
    A merge anchor is defined as a (vehicle, position) tuple. vehicle can be either an anchor or normal
    vehicle. position can be either None or a float position. If position is None, vehicle is an anchor,
    and does not need to be updated. Otherwise, position is a float of the position on curlane,
    and the merge anchor is the vehicle on the same track as curlane which is closest to position without
    yet passing position.
    position being None corresponds to the situation where a new lane starts.
    position being a float corresponds to the situation where two lanes initially meet.
    Unlike lfol/rfol, merge anchors do not need to be completely updated. They should be kept
    in the same track as curlane however.

    Args:
        curlane: Lane object to update
        lc_actions: dictionary with keys as vehicles which request lane changes in the current timestep,
            values are a string either 'l' or 'r' which indicates the side of the change

    Returns:
        None. Modifies merge_anchors attribute for curlane
    """
    for i in range(len(curlane.merge_anchors)):
        veh, pos = curlane.merge_anchors[i][:]
        if pos is None:  # merge anchor is always an anchor we do nothing
            # update_lrfol(veh)  # no need to update lrfol for anchors
            pass
        else:
            # veh is an anchor -> we see if we can make its leader the new merge anchor
            if veh.cf_parameters is None:
                lead = veh.lead
                if lead is not None:
                    temp = curlane.roadlen[lead.road] + lead.pos
                    if temp - pos < 0:
                        curlane.merge_anchors[i][0] = lead

            elif veh in lc_actions:
                if lc_actions[veh] == 'l':
                    curlane.merge_anchors[i][0] = veh.rfol
                else:
                    curlane.merge_anchors[i][0] = veh.lfol

            elif curlane.roadlen[veh.road]+veh.pos - pos > 0:
                curlane.merge_anchors[i][0] = veh.fol


def new_relaxation(veh, timeind, dt):
    """Generates relaxation for a vehicle after it experiences a lane change.

    This is called directly after a vehicle changes it lane, while it still has the old value for its
    headway, and its position has not yet been updated.
    See (https://arxiv.org/abs/1904.08395) for an explanation of the relaxation model.
    This implements single parameter relaxation. The Vehicle attributes associated with relaxation are
        relax_parameters: float which gives the relaxation constant (units of time)
        in_relax: bool of whether or not vehicle is experiencing relaxation currently
        relax: list of floats which stores the relaxation values
        relax_start: time index (timeind) of the 0 index of relax
        relaxmem: memory of past relaxation, list of tuples of (starttime, endtime, relax)

    Args:
        veh: Vehicle to add relaxation to
        timeind: int giving the timestep of the simulation (0 indexed)
        dt: float of time unit that passes in each timestep

    Returns:
        None. Modifies relaxation attributes for vehicle in place.
    """
    rp = veh.relax_parameters
    if veh.lead is None or rp is None:
        return

    prevlead = veh.leadmem[-2][0]
    if prevlead is None:
        # olds = veh.get_eql(veh.speed)
        olds = veh.get_eql(veh.lead.speed)
    else:
        olds = veh.hd
    news = veh.lane.get_headway(veh, veh.lead)


    relaxamount = olds-news
    relaxlen = math.ceil(rp/dt) - 1
    curr = relaxamount*np.linspace(1 - dt/rp, 1 - dt/rp*relaxlen, relaxlen)

    if veh.in_relax:  # add to existing relax
        curlen = len(veh.relax)
        newend = timeind + relaxlen  # time index when relax ends
        newrelax = np.zeros((newend - veh.relax_start+1))
        newrelax[0:curlen] = veh.relax
        newrelax[timeind-veh.relax_start+1:] += curr
        veh.relax = newrelax
    else:  # create new relax
        veh.in_relax = True
        veh.relax_start = timeind + 1
        veh.relax = curr


def update_veh_lane(veh, oldlane, newlane, timeind, side=None):
    """When a vehicle enters a new lane, this updates the lane, road, pos, and lanemem attributes.

    Args:
        veh: Vehicle object to update.
        oldlane: current Lane veh is on.
        newlane: The new Lane the vehicle is changing to.
        timeind: int giving the timestep of the simulation (0 indexed)
        side: if side is not None, also updates the 'r_lc'/'l_lc' attributes (if a vehicle changes to left,
            you need to update r_lc attribute, so pass 'r_lc'). The l/r attributes are set to discretionary
            only if the corresponding lane is in the same road

    Returns:
        None. Modifies veh in place
    """
    newroadname = newlane.roadname
    if side is None:
        if newroadname != veh.road:
            veh.pos += -oldlane.roadlen[newroadname]  # add the newlane.start?
            veh.road = newroadname
    else:
        if newroadname != veh.road:
            veh.pos += -oldlane.roadlen[newroadname]
            veh.road = newroadname
            setattr(veh, side, None)
        else:
            setattr(veh, side, 'discretionary')
    veh.lane = newlane
    veh.lanemem.append((newlane, timeind))


# ######
# explanation of why leader follower relationships are designed this way (i.e. why we have lfol/llead etc.)
# in current logic, main cost per timestep is just one distance compute in update_lrfol
# whenever there is a lane change, there are a fair number of extra updates we have to do to keep all
# of the rlead/llead updated. Also, whenever an lfol/rfol changes, there are two rlead/lead attributes
# that need to be changed as well. Thus this strategy is very efficient assuming we want to keep lfol/rfol
# updated (call lc every timestep), lane changes aren't super common, and all vehicles travel at around
# the same speed. (which are all reasonable assumptions)

# naive way would be having to do something like keep a sorted list, every time we want lfol/rfol
# we have to do log(n) dist computes, where n is the number of vehicles in the current lane.
# whenever a vehicle changes lanes, you need to remove from the current list and add to the new,
# so it is log(n) dist computations + 2n for searching/updating the 2 lists.
# Thus the current implementation is definitely much better than the naive way.

# Another option you could do is to only store lfol/rfol, to keep it updated you would have to
# do 2 dist calculations per side per timestep (do a call of leadfol find where we already have either
# a follower or leader as guess). When there is a lane change store a dict which has the lane changing
# vehicle as a key, and store as the value the new guess to use. You would need to use this dict to update
# guesses whenever it is time to update the lfol/rfol. Updating guesses efficiently is the challenge here.
# This strategy would have higher costs per timestep to keep lfol/rfol updated, but would be simpler to
# update when there is a lane change. Thus it might be more efficient if the number of timesteps is small
# relative to the number of lane changes.
# Because you don't need the llead/rlead in this option, you also can avoid fully updating l/rfol every
# timestep, although you would need to check if your l/rfol had a lane change, but that operation
# just checks for a hash value or compares anchor vehicles, which is less expensive than a get_dist call.

# More generally, the different strategies (always updating lead/fol relationships, having l/rlead or not)
# can be combined. If you don't need l/rlead, then there is no reason to keep it updated (or even have it).
# likewise, if you don't need to always have the lead/fol updated every timestep, there is no reason to
# keep it updated.
# TODO Thus it makes sense to implement versions of update_change and update_lrfol for the case where we don't
# have llead/rlead attributes, and we only update a vehicle l/rfol when the lane changing model is called.
# this can be a performance optimization that can be done after adding parralelism to the code.
# ######
def update_change(lc_actions, veh, timeind):
    """When a vehicle changes lanes, this function does all the necessary updates.

    When a vehicle changes lanes, we need to update it's lane, road, llane/rlane, r/l_lc, lanemem,
    lc_side, coop_veh, lc_urgency attributes.
    More importantly, we need to update all the leader/follower relationships.
    ***Naming conventions***
    Every vehicle has its leader (lead) and follower (fol). Putting l or r in front of lead/fol indicates
    that it is the left/right leader/follower. Consider some vehicle, the 'ego vehicle'. The ego vehicle's
    lfol is the vehicle in the left lane closest to the ego vehicle, without going past the position of the
    ego vehicle. llead has two possible meanings. The llead attribute is the set of all vehicles which
    have the ego vehicle as a rfol. In the context of a lane changing model, we use llead to refer to the
    leader of lfol. Note that the leader of lfol is not even necessarily in the set which defines
    the llead attribute.
    The same definitions apply to rfol and rlead as to lfol and llead.
    The other naming conventions are lcside, newlcside, and opside. If a vehicle changes to the left,
    lcside (lane change side) refers to the left lane, the opside (opposite lane change side) refers to
    the right lane. The newlcside (new lane change side) is the new lcside after changing lanes, so if the
    side is left, it refers to two lanes to the left.
    Note in this case we are using 'new' to refer to the situation after the lane change. This is another
    convention used for lane changing models.

    Args:
        lc_actions: dictionary with keys as vehicles which request lane changes in the current timestep,
            values are a string either 'l' or 'r' which indicates the side of the change
        veh: Vehicle object which changes lanes, and has a key/value in lc_actions
        timeind: int giving the timestep of the simulation (0 indexed)

    Returns:
        None. Modifies veh, and all vehicles which have a relationship with veh, in place.
    """
    # TODO no check for vehicles moving into same gap (store the lcside fol/lead in lc_actions,
    # check if they are the same?)

    # initialization, update lane/road/position and update l/r_lc attributes
    if lc_actions[veh] == 'l':
        # update lane/road/position, and the opside l/r_lc attributes
        veh.rlane = veh.lane
        lcsidelane = veh.llane
        update_veh_lane(veh, veh.lane, lcsidelane, timeind+1, 'r_lc')
        # update new lcside lane change attribute
        newlcsidelane = lcsidelane.get_connect_left(veh.pos)
        veh.llane = newlcsidelane
        if newlcsidelane is not None and newlcsidelane.roadname == veh.road:
            veh.l_lc = 'discretionary'
        else:
            veh.l_lc = None

    else:
        veh.llane = veh.lane
        lcsidelane = veh.rlane
        update_veh_lane(veh, veh.lane, lcsidelane, timeind+1, 'l_lc')

        newlcsidelane = lcsidelane.get_connect_right(veh.pos)
        veh.rlane = newlcsidelane
        if newlcsidelane is not None and newlcsidelane.roadname == veh.road:
            veh.r_lc = 'discretionary'
        else:
            veh.r_lc = None

    # update tact/coop components
    veh.lc_side = veh.coop_veh = veh.lc_urgency = None

    # ######update all leader/follower relationships
    update_leadfol_after_lc(veh, lcsidelane, newlcsidelane, lc_actions[veh], timeind)

    return


def update_leadfol_after_lc(veh, lcsidelane, newlcsidelane, side, timeind):
    """Logic for updating all the leader/follower relationships for veh following a lane change.

    Args:
        veh: Vehicle object which changed lanes.
        lcsidelane: The new lane veh changed onto.
        newlcsidelane: The new lane on the lane change side for veh.
        side: side of lane change, either left ('l') or right ('r').
        timeind: int giving the timestep of the simulation (0 indexed)

    Returns:
        None. Modifies veh and any vehicles which have leader/follower relationships with veh in place.
    """
    if side == 'l':
        # define lcside/opside
        lcsidefol, opsidefol, lcsidelead, opsidelead = 'lfol', 'rfol', 'llead', 'rlead'
    else:
        lcsidefol, opsidefol, lcsidelead, opsidelead = 'rfol', 'lfol', 'rlead', 'llead'

    # update current leader
    lead = veh.lead
    fol = veh.fol
    if lead is None:
        pass
    else:
        lead.fol = fol

    # update opposite/lc side leaders
    for j in getattr(veh, opsidelead):
        setattr(j, lcsidefol, fol)
    for j in getattr(veh, lcsidelead):
        setattr(j, opsidefol, fol)

    # update follower
    getattr(fol, lcsidelead).update(getattr(veh, lcsidelead))
    getattr(fol, opsidelead).update(getattr(veh, opsidelead))
    fol.lead = lead
    fol.leadmem.append((lead, timeind+1))

    # update opposite side for vehicle
    vehopsidefol = getattr(veh, opsidefol)
    if vehopsidefol is not None:
        getattr(vehopsidefol, lcsidelead).remove(veh)
    setattr(veh, opsidefol, fol)
    getattr(fol, lcsidelead).add(veh)
    # update cur lc side follower for vehicle
    lcfol = getattr(veh, lcsidefol)
    lclead = lcfol.lead
    lcfol.lead = veh
    lcfol.leadmem.append((veh, timeind+1))
    getattr(lcfol, opsidelead).remove(veh)
    veh.fol = lcfol
    # update lc side leader
    veh.lead = lclead
    veh.leadmem.append((lclead, timeind+1))

    if lclead is not None:
        lclead.fol = veh
    # update for new left/right leaders - opside first
    newleads = set()
    oldleads = getattr(lcfol, opsidelead)
    for j in oldleads.copy():
        curdist = lcsidelane.get_dist(veh, j)
        if curdist > 0:
            setattr(j, lcsidefol, veh)
            newleads.add(j)
            oldleads.remove(j)
    setattr(veh, opsidelead, newleads)
    # lcside leaders
    newleads = set()
    oldleads = getattr(lcfol, lcsidelead)
    mindist = math.inf
    minveh = None
    for j in oldleads.copy():
        curdist = lcsidelane.get_dist(veh, j)
        if curdist > 0:
            setattr(j, opsidefol, veh)
            newleads.add(j)
            oldleads.remove(j)
            if curdist < mindist:
                mindist = curdist
                minveh = j  # minveh is the leader of new lc side follower
    setattr(veh, lcsidelead, newleads)

    # update new lcside leaders/follower
    if newlcsidelane is None:
        setattr(veh, lcsidefol, None)
    else:
        if minveh is not None:
            setattr(veh, lcsidefol, minveh.fol)
            getattr(minveh.fol, opsidelead).add(veh)
        else:
            guess = get_guess(lcfol, lclead, veh, lcsidefol, newlcsidelane)
            unused, newlcsidefol = lcsidelane.leadfol_find(veh, guess, side)
            setattr(veh, lcsidefol, newlcsidefol)
            getattr(newlcsidefol, opsidelead).add(veh)


def get_guess(lcfol, lclead, veh, lcsidefol, newlcsidelane):
    """Generates a guess to use for finding the newlcside follower for a vehicle.

    Since we keep the lfol/rfol updated, after a lane change we need to find the newlcside follower if
    applicable. This function will generate a guess by using the lcside fol of the lcfol/lclead.
    If such a guess cannot be found, we use the AnchorVehicle for the newlcside lane.


    Args:
        lcfol: lcfol for veh (fol after lane change)
        lclead: lclead for veh (lead after lane change)
        veh: want to generate a guess to find veh's new lcside follower.
        lcsidefol: str, either lfol if vehicle changes left or rfol otherwise.
        newlcsidelane: new lane change side lane after veh changes lanes.

    Returns:
        guess: Vehicle/AnchorVehicle to use as guess for leadfol_find

    """
    # need to find new lcside follower for veh
    guess = getattr(lcfol, lcsidefol)
    anchor = newlcsidelane.anchor
    if guess is None or guess.lane.anchor is not anchor:
        if lclead is not None:
            guess = getattr(lclead, lcsidefol)
            if guess is None or guess.lane.anchor is not anchor:
                guess = anchor
        else:
            guess = anchor
    return guess


class Simulation:
    """Implements a traffic microsimulation.

    Basically just a wrapper for update_net. For more information on traffic
    microsimulation refer to the full documentation which has extra details and explanation.

    Attributes:
        inflow lanes: list of all lanes which have inflow to them (i.e. all lanes which have upstream
            boundary conditions, meaning they can add vehicles to the simulation)
        merge_lanes: list of all lanes which have merge anchors
        vehicles: set of all vehicles which are in the simulation at the first time index. This is kept
            updated so that vehicles is always the set of all vehicles currently being simulated.
        prev_vehicles: set of all vehicles which have been removed from simulation. So prev_vehicles and
            vehicles are disjoint sets, and their union contains all vehicles which have been simulated.
        vehid: starting vehicle ID for the next vehicle to be added. Used for hashing vehicles.
        timeind: the current time index of the simulation (int). Updated as simulation progresses.
        dt: constant float. timestep for the simulation.
    """

    def __init__(self, inflow_lanes, merge_lanes, vehicles=None, prev_vehicles=None, vehid=0,
                 timeind=0, dt=.25):
        """Inits simulation.

        Args:
            inflow_lanes: list of all Lanes which have inflow to them
            merge_lanes: list of all Lanes which have merge anchors
            vehicles: set of all Vehicles in simulation in first timestep
            prev_vehicles: set of all Vehicles which were previously removed from simulation.
            vehid: vehicle ID used for the next vehicle to be created.
            timeind): starting time index (int) for the simulation.
            dt: float for how many time units pass for each timestep. Defaults to .25.

        Returns:
            None. Note that we keep references to all vehicles through vehicles and prev_vehicles,
            a Vehicle stores its own memory.
        """
        self.inflow_lanes = inflow_lanes
        self.merge_lanes = merge_lanes
        self.vehicles = set() if vehicles is None else vehicles
        self.prev_vehicles = set() if prev_vehicles is None else prev_vehicles
        self.vehid = vehid
        self.timeind = timeind
        self.dt = dt

        for curlane in inflow_lanes:  # need to generate parameters of the next vehicles
            if curlane.newveh is None:
                # cf_parameters, lc_parameters, kwargs = curlane.new_vehicle()
                # curlane.newveh = Vehicle(self.vehid, curlane, cf_parameters, lc_parameters, **kwargs)
                curlane.new_vehicle(self.vehid)
                self.vehid += 1

    def step(self, timeind):
        """Logic for doing a single step of simulation.

        Args:
            timeind (TYPE): current time index (int) for the simulation

        Returns:
            None.
        """
        lc_actions = {}

        for veh in self.vehicles:
            veh.set_cf(self.timeind, self.dt)

        for veh in self.vehicles:
            veh.set_lc(lc_actions, self.timeind, self.dt)

        self.vehid, remove_vehicles = update_net(self.vehicles, lc_actions, self.inflow_lanes,
                                                 self.merge_lanes, self.vehid, self.timeind, self.dt)

        self.timeind += 1
        self.prev_vehicles.update(remove_vehicles)

    def simulate(self, timesteps):
        """Call step method timesteps number of times."""
        for i in range(timesteps):
            self.step(self.timeind)

    def reset(self):  # noqa # TODO - ability to put simulation back to initial time
        pass


def get_eql_helper(veh, x, input_type='v', eql_type='v', spdbounds=(0, 1e4), hdbounds=(0, 1e4), tol=.1):
    """Solves for the equilibrium solution of vehicle veh given input x.

    To use this method, the Vehicle must have an eqlfun method defined. The eqlfun can typically be defined
    analyticallly for one input type.
    The equilibrium (eql) solution is defined as a pair of (headway, speed) such that the car following model
    will give 0 acceleration. For any possible speed, (0 through maxspeed) there is a unique headway which
    defines the equilibrium solution.

    Args:
        veh: Vehicle to obtain equilibrium solution for
        x: float of either speed or headway
        input_type: if input_type is 'v' (v for velocity), then x is a speed. Otherwise x is a headway.
            If x is a speed, we return a headway. Otherwise we return a speed.
        eql_type: If 'v', the vehicle's eqlfun accepts a speed and returns a headway. Otherwise it
            accepts a headway and returns a speed. If input_type != eql_type, we numerically invert the
            eqlfun.
        spdbounds: Bounds on speed. If x is outside the bounds, we project it onto bounds. Also used
            if eql_type != input_type.
        hdbounds: Bounds on headway. If x is outside the bounds, we project it onto bounds. Also used
            if eql_type != input_type.
        tol: tolerance for solver.

    Raises:
        RuntimeError: If the solver cannot invert the equilibrium function, we return an error. If this
            happens, it's very likely because your equilibrium function is wrong, or because your bounds
            are wrong.

    Returns:
        float value of either headway or speed which together with input x defines the equilibrium solution.
    """
    if input_type == 'v':
        if x < spdbounds[0]:
            x = spdbounds[0]
        elif x > spdbounds[1]:
            x = spdbounds[1]
    elif input_type == 's':
        if x < hdbounds[0]:
            x = hdbounds[0]
        elif x > hdbounds[1]:
            x = hdbounds[1]
    if input_type == eql_type:
        return veh.eqlfun(veh.cf_parameters, x)
    elif input_type != eql_type:
        def inveql(y):
            return x - veh.eqlfun(veh.cf_parameters, y)
        if eql_type == 'v':
            bracket = spdbounds
        else:
            bracket = hdbounds
        ans = sc.root_scalar(inveql, bracket=bracket, xtol=tol, method='brentq')
        if ans.converged:
            return ans.root
        else:
            raise RuntimeError('could not invert provided equilibrium function')


def inv_flow_helper(veh, x, leadlen=None, output_type='v', congested=True, eql_type='v',
                    spdbounds=(0, 1e4), hdbounds=(0, 1e4), tol=.1, ftol=.01):
    """Solves for the equilibrium solution corresponding to a given flow x.

    To use this method, a vehicle must have an eqlfun defined. A equilibrium solution can be converted to
    a flow. This function takes a flow and finds the corresponding equilibrium solution. To do this,
    it first finds the maximum flow possible, and then based on whether the flow corresponds to the
    congested or free flow regime, we solve for the correct equilibrium.

    Args:
        veh: Vehicle to invert flow for.
        x: flow (float) to invert
        leadlen: When converting an equilibrium solution to a flow, we must use a vehicle length. leadlen
            is that vehicle length. If None, we will infer the vehicle length.
        output_type: if 'v', we want to return a velocity, if 's' we want to return a 'headway'. If 'both',
            we want to return a tuple.
        congested: True if we assume the given flow corresponds to the congested regime, otherwise
            we assume it corresponds to the free flow regime.
        eql_type: If 'v', the vehicle's eqlfun accepts a speed and returns a headway. Otherwise it
            accepts a headway and returns a speed. If input_type != eql_type, we numerically invert the
            eqlfun.
        spdbounds: Bounds on speed. If x is outside the bounds, we project it onto bounds. Also used
            if eql_type != input_type.
        hdbounds: Bounds on headway. If x is outside the bounds, we project it onto bounds. Also used
            if eql_type != input_type.
        tol: tolerance for solver.
        ftol: tolerance for solver for finding maximum flow.

    Raises:
        RuntimeError: Raised if either solver fails.

    Returns:
        float velocity if output_type = 'v', float headway if output_type = 's', tuple of (velocity, headway)
        if output_type = 'both'
    """
    if leadlen is None:
        lead = veh.lead
    if lead is not None:
        leadlen = lead.len
    else:
        leadlen = veh.len

    if eql_type == 'v':
        bracket = spdbounds
    else:
        bracket = hdbounds

    def maxfun(y):
        return -veh.get_flow(y, leadlen=leadlen, input_type=eql_type)
    res = sc.minimize_scalar(maxfun, bracket=bracket, options={'xatol': ftol}, method='bounded',
                             bounds=bracket)

    if res['success']:
        if congested:
            invbounds = (bracket[0], res['x'])
        else:
            invbounds = (res['x'], bracket[1])
        if -res['fun'] < x:
            print('warning - inputted flow is too large to be achieved')
            if output_type == eql_type:
                return res['x']
            else:
                return veh.get_eql(res['x'], input_type=eql_type)
    else:
        raise RuntimeError('could not find maximum flow')

    if eql_type == 'v':
        def invfun(y):
            return x - y/(veh.eqlfun(veh.cf_parameters, y) + leadlen)
    elif eql_type == 's':
        def invfun(y):
            return x - veh.eqlfun(veh.cf_parameters, y)/(y+leadlen)

    ans = sc.root_scalar(invfun, bracket=invbounds, xtol=tol, method='brentq')

    if ans.converged:
        if output_type == 'both':
            if eql_type == 'v':
                return (ans.root, ans.root/x - leadlen)
            else:
                return ((ans.root+leadlen)*x, ans.root)
        else:
            if output_type == eql_type:
                return ans.root
            elif output_type == 's':
                return ans.root/x - leadlen
            elif output_type == 'v':
                return (ans.root+leadlen)*x
    else:
        raise RuntimeError('could not invert provided equilibrium function')


def set_lc_helper(veh, chk_lc=1, get_fol=True):
    """Calculates the new headways to be passed to the lane changing (LC) model.

    Evaluates the lane changing situation to decide if we need to evaluate lane changing model on the
    left side, right side, both sides, or neither. For any sides we need to evaluate, finds the new headways
    (new vehicle headway, new lcside follower headway).
    If new headways are negative, returns positive instead.

    Args:
        veh: Vehicle to have their lane changing model called.
        chk_lc: float between 0 and 1 which gives the probability of checking the lane changing model
            when the vehicle is in a discretionary state.
        get_fol: if True, we also find the new follower headway.

    Returns:
        bool: True if we want to call the lane changing model.
        tuple of floats: (lside, rside, newlfolhd, newlhd, newrfolhd, newrhd, newfolhd). lside/rside
            are bools which are True if we need to check that side in the LC model. rest are float headways,
            giving the new headway for that vehicle. If get_fol = False, newfolhd is not present.
    """
    # first determine what situation we are in and which sides we need to check
    l_lc, r_lc = veh.l_lc, veh.r_lc
    if l_lc is None:
        if r_lc is None:
            return False, None
        elif r_lc == 'discretionary':
            lside, rside = False, True
            chk_cond = not veh.lc_side
        else:
            lside, rside = False, True
            chk_cond = False
    elif l_lc == 'discretionary':
        if r_lc is None:
            lside, rside = True, False
            chk_cond = not veh.lc_side
        elif r_lc == 'discretionary':
            if veh.lc_side is not None:
                chk_cond = False
                if veh.lc_side == 'l':
                    lside, rside = True, False
                else:
                    lside, rside = False, True
            else:
                chk_cond = True
                lside, rside = True, True
        else:
            lside, rside = False, True
            chk_cond = False
    else:
        lside, rside = True, False
        chk_cond = False

    if chk_cond:  # decide if we want to evaluate lc model or not - this only applies to discretionary state
        # when there is no cooperation or tactical positioning
        if chk_lc >= 1:
            pass
        elif np.random.rand() > chk_lc:
            return False, None

    # next we compute quantities to send to LC model for the required sides
    if lside:
        newlfolhd, newlhd = get_new_hd(veh.lfol, veh, veh.llane)  # better to just store left/right lanes
    else:
        newlfolhd = newlhd = None

    if rside:
        newrfolhd, newrhd = get_new_hd(veh.rfol, veh, veh.rlane)
    else:
        newrfolhd = newrhd = None

    # if get_fol option is given to wrapper, it means model requires the follower's quantities as well
    if get_fol:
        fol, lead = veh.fol, veh.lead
        if fol.cf_parameters is None:
            newfolhd = None
        elif lead is None:
            newfolhd = None
        else:
            newfolhd = fol.lane.get_headway(fol, lead)
    else:
        return True, (lside, rside, newlfolhd, newlhd, newrfolhd, newrhd)

    return True, (lside, rside, newlfolhd, newlhd, newrfolhd, newrhd, newfolhd)


def get_new_hd(lcsidefol, veh, lcsidelane):
    """Calculates new headways for a vehicle and its left or right follower.

    Args:
        lcsidefol: either the lfol or rfol of veh.
        veh: Vehicle whose lane changing model is being evaluated
        lcsidelane: the lcside lane of veh.

    Returns:
        newlcsidefolhd: new float headway for lcsidefol
        newlcsidehd: new float headway for veh
    """
    # helper functino for set_lc_helper
    lcsidelead = lcsidefol.lead
    if lcsidelead is None:
        newlcsidehd = None
    else:
        newlcsidehd = lcsidelane.get_headway(veh, lcsidelead)
        # if newlcsidehd < 0:
            # newlcsidehd = 1e-6
    if lcsidefol.cf_parameters is None:
        newlcsidefolhd = None
    else:
        newlcsidefolhd = lcsidefol.lane.get_headway(lcsidefol, veh)
        # if newlcsidefolhd < 0:
            # newlcsidefolhd = 1e-6

    return newlcsidefolhd, newlcsidehd


class Vehicle:
    """Base Vehicle class. Implemented for a second order ODE car following model.

    Vehicles are responsible for implementing the rules to update their positions. This includes a
    'car following' (cf) model which is used to update the longitudinal (in the direction of travel) position.
    There is also a 'lane changing' (lc) model which is used to update the latitudinal (which lane) position.
    Besides these two fundamental components, Vehicles also need an update method, which updates their
    longitudinal positions and memory of their past (past memory includes any quantities which are needed
    to differentiate the simulation).
    Vehicles also contain the quantities lead, fol, lfol, rfol, llead, and rlead, (i.e. their vehicle
    relationships) which define the order of vehicles on the road, and is necessary for calling the cf
    and lc models.
    Vehicles also maintain a route, which defines what roads they want to travel on in the road network.
    Besides their actual lc model, Vehicles also handle any additional components of lane changing,
    such as relaxation, cooperation, or tactical lane changing models.
    Lastly, the vehicle class has some methods which may be used for certain boundary conditions.

    Attributes:
        vehid: unique vehicle ID for hashing
        len: length of vehicle (float)
        lane: Lane object vehicle is currently on
        road: str name of the road lane belongs to
        cf_parameters: list of float parameters for the cf model
        lc_parameters: list of float parameters for the lc model
        relax_parameters: float parameter for relaxation; if None, no relaxation
        in_relax: bool, True if there is currently relaxation
        relax: if there is currently relaxation, relax is a list of floats giving the relaxation values.
        relax_start: time index corresponding to relax[0] if in_relax, otherwise None. (int)
        route_parameters: parameters for the route model (list of floats)
        route: list of road names (str). When the vehicle first enters the simulation or enters a new road,
            the route gets pop().
        routemem: route which was used to init vehicle.
        minacc: minimum allowed acceleration (float)
        maxacc: maxmimum allowed acceleration(float)
        maxspeed: maximum allowed speed (float)
        hdbounds: tuple of minimum and maximum possible headway.
        eql_type: If 'v', the vehicle's eqlfun accepts a speed and returns a headway. Otherwise it
            accepts a headway and returns a speed.
        shift_parameters: list of float parameters for the tactical/cooperative model. shift_parameters
            control how a vehicle can modify its acceleration in order to facilitate lane changing.
        coop_parameters: float between (0, 1) which gives the base probability of the vehicle
            cooperating with a vehicle wanting to change lanes
        lc_side: if the vehicle enters into a tactical or cooperative state, lc_side gives which side the
            vehicle wants to change in, either 'l' or 'r'
        lc_urgency: for mandatory lane changes, lc_urgency is a tuple of floats which control if
            the ego vehicle can force cooperation (simulating aggressive behavior)
        coop_veh: For cooperation, coop_veh is a reference the vehicle giving cooperation. There is no
            attribute (currently) which allows the ego vehicle to see if itself is giving cooperation.
        lead: leading vehicle (Vehicle)
        fol: following vehicle (Vehicle)
        lfol: left follower (Vehicle)
        rfol: right follower (Vehicle)
        llead: set of all vehicles which have the ego vehicle as a right follower
        rlead: set of all vehicles which have the ego vehicle as a left follower
        starttime: first time index a vehicle is simulated.
        endtime: the last time index a vehicle is simulated. (or None)
        leadmem: list of tuples, where each tuple is (lead vehicle, time) giving the time the ego vehicle
            first begins to follow the lead vehicle.
        lanemem: list of tuples, where each tuple is (Lane, time) giving the time the ego vehicle
            first enters the Lane.
        posmem: list of floats giving the position, where the 0 index corresponds to the position at starttime
        speedmem: list of floats giving the speed, where the 0 index corresponds to the speed at starttime
        relaxmem: list of tuples where each tuple is (first time, last time, relaxation) where relaxation
            gives the relaxation values for between first time and last time
        pos: position (float)
        speed: speed (float)
        hd: headway (float)
        acc: acceleration (float)
        llane: the Lane to the left of the current lane the vehicle is on, or None
        rlane: the Lane to the right of the current lane the vehicle is on, or None
        l_lc: the current lane changing state for the left side, None, 'discretionary' or 'mandatory'
        r_lc: the current lane changing state for the right side, None, 'discretionary' or 'mandatory'
        cur_route: dictionary where keys are lanes, value is a list of route event dictionaries which
            defines the route a vehicle with parameters p needs to take on that lane
        route_events: list of current route events for current lane
        lane_events: list of lane events for current lane
    """
    # TODO implementation of adjoint method for cf, relax, shift parameters

    def __init__(self, vehid, curlane, cf_parameters, lc_parameters, lead=None, fol=None, lfol=None,
                 rfol=None, llead=None, rlead=None, length=3, eql_type='v', relax_parameters=15,
                 shift_parameters=None, coop_parameters=.2, route_parameters=None, route=None, accbounds=None,
                 maxspeed=1e4, hdbounds=None):
        """Inits Vehicle. Cannot be used for simulation until initialize is also called.

        After a Vehicle is created, it is not immediatley added to simulation. This is because different
        upstream (inflow) boundary conditions may require to have access to the vehicle's parameters
        and methods before actually adding the vehicle. Thus, to use a vehicle you need to first call
        initialize, which sets the remaining attributes.

        Args:
            vehid: unique vehicle ID for hashing
            curlane: lane vehicle starts on
            cf_parameters: list of float parameters for the cf model
            lc_parameters: list of float parameters for the lc model
            lead: leading vehicle (Vehicle). Optional, can be set by the boundary condition.
            fol: following vehicle (Vehicle). Optional, can be set by the boundary condition.
            lfol: left follower (Vehicle). Optional, can be set by the boundary condition.
            rfol: right follower (Vehicle). Optional, can be set by the boundary condition.
            llead: set of all vehicles which have the ego vehicle as a right follower. Optional, can be set
                by the boundary condition.
            rlead: set of all vehicles which have the ego vehicle as a left follower. Optional, can be set
                by the boundary condition.
            length: float (optional). length of vehicle.
            eql_type: If 'v', the vehicle's eqlfun accepts a speed and returns a headway. Otherwise it
            accepts a headway and returns a speed.
            relax_parameters: float parameter for relaxation; if None, no relaxation
            shift_parameters: list of float parameters for the tactical/cooperative model. shift_parameters
                control how a vehicle can modify its acceleration in order to facilitate lane changing.
            coop_parameters: float between (0, 1) which gives the base probability of the vehicle
                cooperating with a vehicle wanting to change lanes
            route: list of road names (str) which defines the route for the vehicle.
            route_parameters: parameters for the route model (list of floats).
            accbounds: tuple of bounds for acceleration.
            maxspeed: maximum allowed speed.
            hdbounds: tuple of bounds for headway.
        """
        self.vehid = vehid
        self.len = length
        self.lane = curlane
        self.road = curlane.road['name'] if curlane is not None else None
        # model parameters
        self.cf_parameters = cf_parameters
        self.lc_parameters = lc_parameters

        # relaxation
        self.relax_parameters = relax_parameters
        self.in_relax = False
        self.relax = None
        self.relax_start = None

        # route parameters
        self.route_parameters = [200, 200] if route_parameters is None else route_parameters
        self.route = [] if route is None else route
        # TODO check if route is empty
        self.routemem = self.route.copy()

        # bounds
        if accbounds is None:
            self.minacc, self.maxacc = -7, 3
        else:
            self.minacc, self.maxacc = accbounds[0], accbounds[1]
        self.maxspeed = maxspeed
        self.hdbounds = (0, 1e4) if hdbounds is None else hdbounds
        self.eql_type = eql_type

        # cooperative/tactical model
        self.shift_parameters = [2, .4] if shift_parameters is None else shift_parameters
        self.coop_parameters = coop_parameters
        self.lc_side = None
        self.lc_urgency = None
        self.coop_veh = None

        # leader/follower relationships
        self.lead = lead
        self.fol = fol
        self.lfol = lfol
        self.rfol = rfol
        self.llead = llead
        self.rlead = rlead

        # memory
        self.endtime = None
        self.leadmem = []
        self.lanemem = []
        self.posmem = []
        self.speedmem = []
        self.relaxmem = []

    def initialize(self, pos, spd, hd, starttime):
        """Sets the remaining attributes of the vehicle, making it able to be simulated.

        Args:
            pos: position at starttime
            spd: speed at starttime
            hd: headway at starttime
            starttime: first time index vehicle is simulated

        Returns:
            None.
        """
        # state
        self.pos = pos
        self.speed = spd
        self.hd = hd

        # memory
        self.starttime = starttime
        self.leadmem.append((self.lead, starttime))
        self.lanemem.append((self.lane, starttime))
        self.posmem.append(pos)
        self.speedmem.append(spd)

        # llane/rlane and l/r
        self.llane = self.lane.get_connect_left(pos)
        if self.llane is None:
            self.l_lc = None
        elif self.llane.roadname == self.road:
            self.l_lc = 'discretionary'
        else:
            self.l_lc = None
        self.rlane = self.lane.get_connect_right(pos)
        if self.rlane is None:
            self.r_lc = None
        elif self.rlane.roadname == self.road:
            self.r_lc = 'discretionary'
        else:
            self.r_lc = None

        # set lane/route events - sets lane_events, route_events, cur_route attributes
        self.cur_route = make_cur_route(self.route_parameters, self.lane, self.route.pop(0))
        # self.route_events = self.cur_route[self.lane].copy()
        set_lane_events(self)
        set_route_events(self)

    def cf_model(self, p, state):
        """Defines car following model.

        Args:
            p: parameters for model (cf_parameters)
            state: list of headway, speed, leader speed

        Returns:
            float acceleration of the model.
        """
        # if state[0] < 0:  # need bound on headway because IDM will not act correctly for negative headways
            # state[0] = .1  # bound is in mobil
        return p[3]*(1-(state[1]/p[0])**4-((p[2]+state[1]*p[1]+(state[1]*(state[1]-state[2])) /
                                            (2*(p[3]*p[4])**(1/2)))/(state[0]))**2)

    def get_cf(self, hd, spd, lead, curlane, timeind, dt, userelax):
        """Wrapper for cf_model.

        Args:
            hd (TYPE): headway
            spd (TYPE): speed
            lead (TYPE): lead Vehicle
            curlane (TYPE): lane self Vehicle is on
            timeind (TYPE): time index
            dt (TYPE): timestep
            userelax (TYPE): boolean for relaxation

        Returns:
            acc (TYPE): DESCRIPTION.
        """
        if lead is None:
            acc = curlane.call_downstream(self, timeind, dt)

        else:
            if userelax:
                # currelax = (1 - math.e**-hd)*self.relax[timeind - self.relax_start]  # don't allow negative
                # relaxed headway
                # currelax = self.relax[timeind - self.relax_start]
                if hd < 15:
                    usehd = hd
                else:
                    usehd = hd + self.relax[timeind - self.relax_start]
                acc = self.cf_model(self.cf_parameters, [usehd, spd, lead.speed])
            else:
                acc = self.cf_model(self.cf_parameters, [hd, spd, lead.speed])

        return acc

    def set_cf(self, timeind, dt):
        """Sets a vehicle's acceleration by calling get_cf."""
        self.acc = self.get_cf(self.hd, self.speed, self.lead, self.lane, timeind, dt, self.in_relax)

    def free_cf(self, p, spd):
        """Defines car following model in free flow.

        The free flow model can be obtained simply by letting the headway go to infinity for cf_model.

        Args:
            p: parameters for model (cf_parameters)
            spd: speed (float)

        Returns:
            float acceleration corresponding to the car following model in free flow.
        """
        return p[3]*(1-(spd/p[0])**4)

    def eqlfun(self, p, v):
        """Equilibrium function.

        Args:
            p:. car following parameters
            v: velocity/speed

        Returns:
            s: headway such that (v,s) is an equilibrium solution for parameters p
        """
        s = ((p[2]+p[1]*v)**2/(1-(v/p[0])**4))**.5
        return s

    def get_eql(self, x, input_type='v'):
        """Get equilibrium using provided function eqlfun, possibly inverting it if necessary."""
        return get_eql_helper(self, x, input_type, self.eql_type, (0, self.maxspeed), self.hdbounds)

    def get_flow(self, x, leadlen=None, input_type='v'):
        """Input a speed or headway, and output the flow based on the equilibrium solution.

        Args:
            x: Input, either headway or speed
            leadlen: When converting an equilibrium solution to a flow, we must use a vehicle length. leadlen
                is that vehicle length. If None, we will infer the vehicle length.
            input_type: if input_type is 'v' (v for velocity), then x is a speed. Otherwise x is a headway.
                If x is a speed, we return a headway. Otherwise we return a speed.

        Returns:
            TYPE: DESCRIPTION.

        """
        if leadlen is None:
            lead = self.lead
            if lead is not None:
                leadlen = lead.len
            else:
                leadlen = self.len
        if input_type == 'v':
            s = self.get_eql(x, input_type=input_type)
            return x / (s + leadlen)
        elif input_type == 's':
            v = self.get_eql(x, input_type=input_type)
            return v / (s + leadlen)

    def inv_flow(self, x, leadlen=None, output_type='v', congested=True):
        """Get equilibrium solution corresponding to the provided flow."""
        return inv_flow_helper(self, x, leadlen, output_type, congested, self.eql_type,
                               (0, self.maxspeed), self.hdbounds)

    def shift_eql(self, p, v, shift_parameters, state):
        """Model used for applying tactical/cooperative acceleration during lane changes.

        The model works by solving for an acceleration which modifies the equilibrium solution by some
        fixed amount. Any model which has an equilibrium solution which can be solved analytically,
        it should be possible to define a model in this way. For IDM, the result that comes out lets
        you modify the equilibrium by a multiple.
        E.g. if the shift parameter = .5, and the equilibrium headway is usually 20 at the provided speed,
        we return an acceleration which makes the equilibrium headway 10. If we request 'decel', the parameter
        must be > 1 so that the acceleration is negative. Otherwise the parameter must be < 1.


        Args:
            p: car following parameters
            v: speed
            shift_parameters: list of deceleration, acceleration parameters. eq'l at v goes to n
                times of normal, where n is the parameter
            state: if state = 'decel' we use shift_parameters[0] else shift_parameters[1]

        Returns:
            TYPE: float acceleration
        """
        # TODO fix this - acceleration too weak, use different formulation
        # could use formulation based on another DE which pushes towards a new eql, uses the eql fun of
        # the base model to calculate what the base acceleration at this new eql would be

        # In Treiber/Kesting JS code they have another way of doing this where vehicles will use their
        # new deceleration if its greater than -2b

        # if state == 'decel':
        #     temp = shift_parameters[0]**2
        # else:
        #     temp = shift_parameters[1]**2

        # return (1 - temp)/temp*p[3]*(1 - (v/p[0])**4)
        return hm.generic_shift(p, v, shift_parameters, state)

    def set_lc(self, lc_actions, timeind, dt):
        """Evaluates a vehicle's lane changing model, recording the result in lc_actions.

        The result of the lane changing (lc) model can be either 'l' or 'r' for left/right respectively,
        or None, in which case there is no lane change. If the model has tactical/cooperative elements added,
        calling the lc model may cause some vehicles to enter into a tactical or cooperative state, which
        modifies the vehicle's acceleration by using the shift_eql method.

        Args:
            lc_actions: dictionary where keys are Vehicles which changed lanes, values are the side of change
            timeind: time index
            dt: timestep

        Returns:
            None. (Modifies lc_actions, some vehicle attributes, in place)
        """
        call_model, args = set_lc_helper(self, self.lc_parameters[-1]*dt)
        if call_model:
            hm.mobil(self, lc_actions, *args, timeind, dt)
        return

    def acc_bounds(self, acc):
        """Apply acceleration bounds."""
        if acc > self.maxacc:
            acc = self.maxacc
        elif acc < self.minacc:
            acc = self.minacc
        return acc

    def update(self, timeind, dt):
        """Applies bounds and updates a vehicle longitudinal state/memory."""
        # bounds on acceleration
        # acc = self.acc_bounds(self.acc)
        acc = self.acc

        # bounds on speed
        temp = acc*dt
        nextspeed = self.speed + temp
        if nextspeed < 0:
            nextspeed = 0
            temp = -self.speed
            nextspeed = 0
        elif nextspeed > self.maxspeed:
            nextspeed = self.maxspeed
            temp = self.maxspeed - self.speed
            nextspeed = self.maxspeed

        # update state
        self.pos += self.speed*dt + .5*temp*dt
        self.speed = nextspeed

        # update memory and relax
        self.posmem.append(self.pos)
        self.speedmem.append(self.speed)
        if self.in_relax:
            if timeind == self.relax_start + len(self.relax) - 1:
                self.in_relax = False
                self.relaxmem.append((self.relax_start, timeind, self.relax))

    def __hash__(self):
        """Vehicles need to be hashable. We hash them with a unique vehicle ID."""
        return hash(self.vehid)

    def __eq__(self, other):
        """Used for comparing two vehicles with ==."""
        return self.vehid == other.vehid

    def __ne__(self, other):
        """Used for comparing two vehicles with !=."""
        return not self.vehid == other.vehid

    def __repr__(self):
        """Display for vehicle in interactive console."""
        return 'vehicle '+str(self.vehid)+' on lane '+str(self.lane)+' at position '+str(self.pos)

    def __str__(self):
        """Convert vehicle to a str representation."""
        return self.__repr__()

    def _leadfol(self):
        """Summarize the leader/follower relationships of the Vehicle."""
        print('-------leader and follower-------')
        if self.lead is None:
            print('No leader')
        else:
            print('leader is '+str(self.lead))
        print('follower is '+str(self.fol))
        print('-------left and right followers-------')
        if self.lfol is None:
            print('no left follower')
        else:
            print('left follower is '+str(self.lfol))
        if self.rfol is None:
            print('no right follower')
        else:
            print('right follower is '+str(self.rfol))

        print('-------'+str(len(self.llead))+' left leaders-------')
        for i in self.llead:
            print(i)
        print('-------'+str(len(self.rlead))+' right leaders-------')
        for i in self.rlead:
            print(i)
        return

    def _chk_leadfol(self, verbose=True):
        """Returns True if the leader/follower relationships of the Vehicle are correct."""
        # If verbose = True, we print whether each relationship is passing or not. Note that this does
        # not actually verify the relationships are correct, it verifies that they are possible.
        lfolpass = True
        lfolmsg = []
        if self.lfol is not None:
            if self.lfol is self:
                lfolpass = False
                lfolmsg.append('lfol is self')
            if self not in self.lfol.rlead:
                lfolpass = False
                lfolmsg.append('rlead of lfol is missing self')
            if self.lfol.lane.anchor is not self.llane.anchor:
                lfolpass = False
                lfolmsg.append('lfol is not in left lane')
            if self.lane.get_dist(self, self.lfol) > 0:
                lfolpass = False
                lfolmsg.append('lfol is in front of self')
            if self.lfol.lead is not None:
                if self.lane.get_dist(self, self.lfol.lead) < 0:
                    lfolpass = False
                    lfolmsg.append('lfol leader is behind self')
            lead, fol = self.lane.leadfol_find(self, self.lfol)
            if fol is not self.lfol:
                if self.lfol.pos == self.pos:
                    pass
                else:
                    lfolpass = False
                    lfolmsg.append('lfol is not correct vehicle - should be '+str(fol))
        rfolpass = True
        rfolmsg = []
        if self.rfol is not None:
            if self.rfol is self:
                rfolpass = False
                rfolmsg.append('rfol is self')
            if self not in self.rfol.llead:
                rfolpass = False
                rfolmsg.append('llead of rfol is missing self')
            if self.rfol.lane.anchor is not self.rlane.anchor:
                rfolpass = False
                rfolmsg.append('rfol is not in right lane')
            if self.lane.get_dist(self, self.rfol) > 0:
                rfolpass = False
                rfolmsg.append('rfol is in front of self')
            if self.rfol.lead is not None:
                if self.lane.get_dist(self, self.rfol.lead) < 0:
                    rfolpass = False
                    rfolmsg.append('rfol leader is behind self')
            lead, fol = self.lane.leadfol_find(self, self.rfol)
            if fol is not self.rfol:
                if self.rfol.pos == self.pos:
                    pass
                else:
                    rfolpass = False
                    rfolmsg.append('rfol is not correct vehicle - should be '+str(fol))
        rleadpass = True
        rleadmsg = []
        for i in self.rlead:
            if i.lfol is not self:
                rleadpass = False
                rleadmsg.append('rlead does not have self as lfol')
            if self.lane.get_dist(self, i) < 0:
                rleadpass = False
                rleadmsg.append('rlead is behind self')
        lleadpass = True
        lleadmsg = []
        for i in self.llead:
            if i.rfol is not self:
                lleadpass = False
                lleadmsg.append('llead does not have self as rfol')
            if self.lane.get_dist(self,i) < 0:
                lleadpass = False
                lleadmsg.append('llead is behind self')
        leadpass = True
        leadmsg = []
        if self.lead is not None:
            if self.lead.fol is not self:
                leadpass = False
                leadmsg.append('leader does not have self as follower')
            if self.lane.get_headway(self,self.lead) < 0:
                leadpass = False
                leadmsg.append('leader is behind self')

        folpass = True
        folmsg = []
        if self.fol.lead is not self:
            folpass = False
            folmsg.append('follower does not have self as leader')
        if self.lane.get_headway(self, self.fol) > 0:
            folpass = False
            folmsg.append('follower is ahead of self')

        res = lfolpass and rfolpass and rleadpass and lleadpass and leadpass and folpass
        if verbose:
            if res:
                print('passing results for '+str(self))
            else:
                print('errors for '+str(self))
            if not lfolpass:
                for i in lfolmsg:
                    print(i)
            if not rfolpass:
                for i in rfolmsg:
                    print(i)
            if not rleadpass:
                for i in rleadmsg:
                    print(i)
            if not lleadpass:
                for i in lleadmsg:
                    print(i)
            if not leadpass:
                for i in leadmsg:
                    print(i)
            if not folpass:
                for i in folmsg:
                    print(i)

        return res


def downstream_wrapper(method='speed', time_series=None, congested=True, merge_side='l',
                       merge_anchor_ind=None, target_lane=None, self_lane=None, shift=1, minacc=-2,
                       stopping = 'car following'):
    """Defines call_downstream method for Lane. keyword options control behavior of call_downstream.

    call_downstream is used instead of the cf model in order to get the acceleration in cases where there is
    no lead vehicle (e.g. because the lead vehicle has left the simulation). Essentially, call_downstream
    will determine the rate at which vehicles can exit the simulation.
    call_downstream also gets used for on-ramps/merges where you have no leader not because you are
    leaving the simulation, but rather because the lane is about to end and you need to move over.
    Lastly, if the simulation starts with no Vehicles, call_downstream will entirely determine
    The keyword arg 'method' defines the behavior of call_downstream method for a Lane.

    Args:
        method: one of 'speed', 'free', 'flow', 'free merge', 'merge'
            'speed' - Give a function which explicitly returns the speed, and we compute the acceleration.
                Options -
                time_series: function takes in timeind, returns speed

            'free' - We use the vehicle's free_cf method to update the acceleration. This is as if vehicles
            can exit the simulation as quickly as possible.
                Options -
                None.

            'flow' - We get a flow from time_series; we then use the vehicle's inverse flow method to find
            the speed corresponding to the flow
                Options -
                time_series: function takes in timeind, returns flow
                congested: whether to assume flow is in congested or free flow branch

            'free merge' - We use the vehicle's free flow method to update, unless we are getting too close
            to the end of the road, in which case we ensure the Vehicle will stop before reaching the end of
            the lane. We assume that vehicles stop as if there was a stationary lead vehicle at the end of
            the lane. This is done by creating an AnchorVehicle at the end of the lane which is used as a
            lead vehicle.
                Options -
                minacc: We always compute the vehicle's acceleration using the anchor as a leader. If the
                    vehicle has an acceleration more negative than minacc, we use the anchor as a leader.
                    otherwise, we use the vehicle's free flow method
                stopping: if 'car following', we use the strategy with minacc. if 'ballistic', we stop
                    only when it is necessary as determined by the vehicle's minimum acceleration
                self_lane: vehicle needs to stop when reaching the end of self_lane
                time_series: Can use either free flow method or time_series; we use time_series if it is
                    not None

            'merge' - This is meant to give a longitudinal update in congested conditions while on a
            bottleneck (on ramp or lane ending, where you must merge). minacc and self_lane give behavior
            the same as in 'free merge'.
                Options -
                merge_side: either 'l' or 'r' depending on which side vehicles need to merge
                target_lane: lanes vehicles need to merge into
                merge_anchor_ind: index for merge anchor in target_lane
                self_lane: if not None, the vehicle needs to stop at the end of self_lane. If None, the
                    vehicle won't stop.
                minacc: if the acceleration needed to stop is more negative than minacc, we begin to stop.
                stopping: if 'car following', we use the strategy with minacc. if 'ballistic', we stop
                    only when it is necessary as determined by the vehicle's minimum acceleration
                shift: we infer a speed based on conditions in target_lane. We do this by shifting the speed
                    of a vehicle in target_lane by shift. (e.g. shift = 1, use the speed from 1 second ago)
                time_series: if we aren't stopping at the end of self_lane,  and can't find a vehicle to infer
                    the speed from, time_series controls the behavior. If None, the vehicle uses its free cf
                    method. Otherwise, time_series specifies a speed which is used.

    Returns:
        call_downstream method for a Lane. Takes in (veh, timeind, dt) and returns acceleration.
    """
    # options - time_series
    if method == 'speed':  # specify a function which takes in time and returns the speed
        def call_downstream(self, veh, timeind, dt):
            speed = time_series(timeind)
            return (speed - veh.speed)/dt
        return call_downstream

    # options - none
    elif method == 'free':  # use free flow method of the vehicle
        def free_downstream(self, veh, *args):
            return veh.free_cf(veh.cf_parameters, veh.speed)
        return free_downstream

    # options - time_series, congested
    elif method == 'flow':  # specify a function which gives the flow, we invert the flow to obtain speed
        def call_downstream(self, veh, timeind, dt):
            flow = time_series(timeind)
            speed = veh.inv_flow(flow, output_type='v', congested=congested)
            return (speed - veh.speed)/dt
        return call_downstream

    # options - minacc, self_lane
    elif method == 'free merge':  # use free flow method of the vehicle, stop at end of lane
        endanchor = AnchorVehicle(self_lane, None)
        endanchor.pos = self_lane.end

        def free_downstream(self, veh, timeind, dt):
            hd = veh.lane.get_headway(veh, endanchor)

            # more aggressive breaking strategy is based on car following model
            if stopping[0] == 'c':
                acc = veh.get_cf(hd, veh.speed, endanchor, veh.lane, timeind, dt, veh.in_relax)
                if acc < minacc:
                    return acc

            # another strategy is to only decelerate when absolutely necessary
            else:
                if hd < veh.speed**2*.5/-veh.minacc+dt*veh.speed:
                    return veh.minacc
            if time_series is not None:
                return (time_series(timeind) - veh.speed)/dt
            return veh.free_cf(veh.cf_parameters, veh.speed)
        return free_downstream
    # options - merge_side, merge_anchor_ind, target_lane, self_lane, shift, minacc, time_series
    elif method == 'merge':
        # first try to get a vehicle in the target_lane and use its shifted speed. Cannot be an AnchorVehicle
        # if we fail to find such a vehicle and time_series is not None: we use time_series
        # otherwise we will use the vehicle's free_cf method
        if merge_side == 'l':
            folside = 'lfol'
        elif merge_side == 'r':
            folside = 'rfol'
        if self_lane is not None:
            endanchor = AnchorVehicle(self_lane, None)
            endanchor.pos = self_lane.end
        else:
            endanchor = None

        def call_downstream(self, veh, timeind, dt):
            # stop if we are nearing end of self_lane
            if endanchor is not None:
                hd = veh.lane.get_headway(veh, endanchor)
                # more aggressive breaking strategy is based on car following model
                if stopping[0] == 'c':
                    acc = veh.get_cf(hd, veh.speed, endanchor, veh.lane, timeind, dt, veh.in_relax)
                    if acc < minacc:
                        return acc

                # another strategy is to only decelerate when absolutely necessary
                else:
                    if hd < veh.speed**2*.5/-veh.minacc+dt*veh.speed:
                        return veh.minacc

            # try to find a vehicle to use for shifted speed
            # first check if we can use your current lc side follower
            # if that fails, try using the merge anchor for the target_lane.
            # can also try the leader of either of the above.
            fol = getattr(veh, folside)
            if merge_anchor_ind is not None:
                if fol is None:
                    fol = target_lane.merge_anchors[merge_anchor_ind][0]
                if fol.cf_parameters is None:
                    fol = fol.lead
            elif fol is None:
                pass
            elif fol.cf_parameters is None:
                fol = fol.lead

            if fol is not None:  # fol must either be none or a vehicle (can't be anchor)
                speed = shift_speed(fol.speedmem, shift, dt)
            elif time_series is not None:
                speed = time_series(timeind)
            else:
                return veh.free_cf(veh.cf_parameters, veh.speed)
            return (speed - veh.speed)/dt

        return call_downstream


class AnchorVehicle:
    """Anchors are 'dummy' Vehicles which can be used as placeholders (e.g. at the beginning/end of Lanes).

    Anchors are used at the beginning of Lanes, to maintain vehicle order, so that vehicles will have
    a correct vehicle order upon being added to the start of the lane. Because of anchors, it is not
    possible for vehicles to have None as a fol/lfol/rfol attribute (unless the left/right lane don't exist).
    Anchors can also be used at the end of the lanes, e.g. to simulate vehicles needing to stop because
    of a traffic light or because the lane ends.
    All Lanes have an anchor attribute which is an AnchorVehicle at the start of the Lanes' track. A track
    is a continuous series of lanes such that a vehicle can travel on all the constituent lanes without
    performing any lane changes (i.e. the end of any lane in the track connects to the start of the next
    lane in the track). Therefore comparing Lane's anchors can also be used to compare their tracks.
    Compared to Vehicles, Anchors don't have a cf or lc model, have much fewer attributes, and don't have
    any methods which update their attributes.
    The way we check for anchors is because they have cf_parameters = None.

    Attributes:
        cf_parameters: Always None, used to identify a vehicle as being an anchor
        lane, road, lfol, rfol, lead, rlead, llead, all have the same meaning as for Vehicle
        pos: position anchor is on, used for headway/dist calculations
        speed: speed of anchor, can be used if anchor is used as a leader for a Vehicle
        hd: always None
        len: length of anchor, should be 0
        leadmem: same format as Vehicle
    """

    def __init__(self, curlane, starttime, lead=None, rlead=None, llead=None):
        """Inits for Anchor."""
        self.cf_parameters = None
        self.lane = curlane
        self.road = curlane.road['name']

        self.lfol = None  # anchor vehicles just need the lead/llead/rlead attributes. no need for (l/r)fol
        self.rfol = None
        self.lead = lead
        self.rlead = set() if rlead is None else rlead
        self.llead = set() if llead is None else llead

        self.pos = curlane.start
        self.speed = 0
        self.hd = None
        self.len = 0

        self.leadmem = [[lead, starttime]]

    def __repr__(self):
        """Representation in ipython console."""
        return 'anchor for lane '+str(self.lane)

    def __str__(self):
        """Convert to string."""
        return self.__repr__()


def get_inflow_wrapper(time_series, inflow_type='flow'):
    """Defines get_inflow method for Lane.

    get_inflow is used for a lane with upstream boundary conditions to increment the inflow_buffer
    attribute which controls when we attempt to add vehicles to the simulation.

    Args:
        time_series: function which takes in a timeind and returns either a flow (inflow_type = 'flow') or
            speed (inflow_type = 'speed' or 'congested').
        inflow_type: Method to add vehicles. One of 'flow', 'speed', 'congested'
            'flow' - time_series returns the flow explicitly

            'speed' - time_series returns a speed, we get a flow from the speed using the get_eql method of
            the Vehicle.

            'congested' - This is meant to add a vehicle with ~0 acceleration as soon as it is possible to do
            so. This is similar to 'speed', but instead of getting speed from time_series, we get it from
            the anchor's lead vehicle. This may help remove artifacts from the upstream boundary
            condition caused by simulations with different Vehicle parameters.
            Requires get_eql method of the Vehicle.

    Returns:
        get_inflow method for a Lane. Takes in (timeind) and returns instantaneous flow, vehicle speed,
        at that time. If we return None for the speed, increment_inflow will invert the flow to obtain speed.
    """
    # give flow series - simple
    if inflow_type == 'flow':
        def get_inflow(self, timeind):
            return time_series(timeind), None
    # give speed series, we convert to equilibrium flow using the parameters of the next vehicle to be added
    # note that if all vehicles have same parameters/length, this is exactly equivalent to the 'flow' method
    # (where the speeds correspond to the flows for the eql soln being used)
    elif inflow_type == 'speed':
        def get_inflow(self, timeind):
            spd = time_series(timeind)
            lead = self.anchor.lead
            if lead is not None:
                leadlen = lead.len
            else:
                leadlen = self.newveh.len
            s = self.newveh.get_eql(spd, find='s')
            return spd / (s + leadlen), spd

    # in congested type it is similar to the 'speed' method but uses the speed from the anchor.lead instead of
    # a speed which is specified a priori. This is basically supposed to add a vehicle with 0 acceleration
    elif inflow_type == 'congested':
        def get_inflow(self, timeind):
            lead = self.anchor.lead
            if lead is not None:
                leadlen = lead.len
                spd = lead.speed
            else:
                leadlen = self.newveh.len
                spd = time_series(timeind)
            s = self.newveh.get_eql(spd, find='s')
            return spd / (s + leadlen), spd

    return get_inflow


def timeseries_wrapper(timeseries, starttimeind=0):
    """Decorator to convert a list or numpy array into a function which accepts a timeind."""
    def out(timeind):
        return timeseries[timeind-starttimeind]
    return out


def eql_inflow_congested(curlane, inflow, c=.8, check_gap=True):
    """Extra condition when adding vehicles for use in congested conditions. Requires to invert flow.

    Suggested by Treiber, Kesting in their traffic flow book for congested conditions. Requires to invert
    the inflow to obtain the equilibrium headway. The actual headway on the road must be at least c times
    the equilibrium headway for the vehicle to be added, where c is a constant.

    Args:
        curlane: Lane with upstream boundary condition, which will possibly have a vehicle added.
        inflow: current instantaneous flow.
        c: Constant, should be less than or equal to 1. Lower is less strict - Treiber, Kesting suggest .8
        check_gap: If False, we don't check the Treiber, Kesting condition, so we don't have to invert
            the flow. We always just add the vehicle

    Returns:
        If The vehicle is not to be added, we return None. Otherwise, we return the (pos, spd, hd) for the
        vehicle to be added with.
    """
    lead = curlane.anchor.lead
    hd = curlane.get_headway(curlane.anchor, lead)
    if check_gap:
        (spd, se) = curlane.newveh.inv_flow(inflow, leadlen=lead.len, output_type='both')  # inverts flow
    else:
        se = -math.inf
        spd = curlane.veh.get_eql(hd, input_type='s')
    if hd > c*se:  # condition met
        return curlane.start, spd, hd
    else:
        return None


def eql_inflow_free(curlane, inflow):
    """Suggested by Treiber, Kesting for free conditions. Requires to invert the inflow to obtain velocity."""
    lead = curlane.anchor.lead
    hd = curlane.get_headway(curlane.anchor, lead)
    # get speed corresponding to current flow
    spd = curlane.newveh.inv_flow(inflow, leadlen=lead.len, output_type='v', congested=False)
    return curlane.start, spd, hd


def shifted_speed_inflow(curlane, dt, shift=1, accel_bound=-.5):
    """Extra condition for upstream boundary based on Newell model and a vehicle's car following model.

    We get the first speed for the vehicle based on the shifted speed of the lead vehicle (similar to Newell
    model). Then we compute the vehicle's acceleration using its own car following model. If the acceleration
    is too negative, we don't add it to the simulation. If we add it, it's with the shifted leader speed.

    Args:
        curlane: Lane with upstream boundary condition, which will possibly have a vehicle added.
        dt: timestep
        shift: amount (time) to shift the leader's speed by
        accel_bound: minimum acceleration a vehicle can be added with

    Returns:
        If The vehicle is not to be added, we return None. Otherwise, we return the (pos, spd, hd) for the
        vehicle to be added with.
    """
    lead = curlane.anchor.lead
    hd = curlane.get_headway(curlane.anchor, lead)
    spd = shift_speed(lead.speedmem, shift, dt)

    if accel_bound is not None:
        newveh = curlane.newveh
        acc = newveh.get_cf(hd, spd, lead, curlane, None, dt, False)
        if acc > accel_bound and hd > 0:  # headway required to be positive for IDM
            return curlane.start, spd, hd
        else:
            return None

    return curlane.start, spd, hd


def shift_speed(speed_series, shift, dt):
    """Given series of speeds, returns the speed shifted by 'shift' amount of time.

    speed_series is a list speeds with constant discretization dt. We assume that the last entry in
    speed_series is the current speed, and we want the speed from shift time ago. If shift is not a multiple
    of dt, we use linear interpolation between the two nearest speeds. If shift time ago is before the
    earliest measurement in speed_series, we return the first entry in speed_series.
    Returns a speed.
    """
    ind = int(shift // dt)
    if ind+1 > len(speed_series):
        return speed_series[0]
    remainder = shift - ind*dt
    spd = (speed_series[-ind-1]*(dt - remainder) + speed_series[-ind]*remainder)/dt  # weighted average
    return spd


def newell_inflow(curlane, dt, p = [1,2], accel_bound = -2):
    """Extra condition for upstream boundary based on DE form of Newell model.

    Args:
        curlane: Lane with upstream boundary condition
        dt: timestep
        p: parameters for Newell model, p[0] = time delay = 1/ speed-headway slope. p[1] = jam spacing
        accel_bound: vehicle must have accel greater than this to be added

    Returns: None if no vehicle is to be added, otherwise a (pos, speed, headway) tuple for IC of new vehicle.
    """

    lead = curlane.anchor.lead
    hd = curlane.get_headway(curlane.anchor,lead)
    newveh = curlane.newveh
    spd = max(min((hd - p[1])/p[0], newveh.maxspeed),0)

    if accel_bound is not None:
        acc = newveh.get_cf(hd, spd, lead, curlane, None, dt, False)
        if acc > accel_bound and hd > 0:
            return curlane.start, spd, hd
        else:
            return None

    return curlane.start, spd, hd


def speed_inflow(curlane, speed_series, timeind, dt, accel_bound=-2):
    """Like shifted_speed_inflow, but gets speed from speed_series instead of the shifted leader speed."""
    lead = curlane.anchor.lead
    hd = curlane.get_headway(curlane.anchor, lead)
    spd = speed_series(timeind)

    if accel_bound is not None:
        newveh = curlane.newveh
        acc = newveh.get_cf(hd, spd, lead, curlane, None, dt, False)
        if acc > accel_bound and hd > 0:
            return curlane.start, spd, hd
        else:
            return None
    return curlane.start, spd, hd


def increment_inflow_wrapper(method='ceql', speed_series=None, accel_bound=-.5, check_gap=True, shift=1, c=.8,
                             p = [1, 2]):
    """Defines increment_inflow method for Lane. keyword args control behavior of increment_inflow.

    The increment_inflow method has two parts to it. First, it is responsible for determining when to add
    vehicles to the simulation. It does this by updating an attribute inflow_buffer. When inflow_buffer >= 1,
    it attempts to add a vehicle to the simulation. There are extra conditions required to add a vehicle,
    which are controlled by the 'method' keyword arg.
    Once it has been determined a new vehicle can be added, this function is also responsible for calling
    the initialize method of the new vehicle, adding the new vehicle with a correct leader/follower
    relationships, and also inits the next vehicle which is to be added.

    Args:
        method: One of 'ceql' (eql_inflow_congested), 'feql' (eql_inflow_free),
            'shifted' (shifted_speed_inflow), or 'speed' (speed_inflow) - refer to those functions
        speed_series: for speed_inflow method
        accel_bound: for speed_inflow and shifted_speed_inflow methods
        check_gap: for eql_inflow_congested method
        shift: for shifted_speed_inflow method
        c: for eql_inflow_congested method
        p: for newell_inflow method

    Returns:
        increment_inflow method -
        Args:
            vehicles: set of vehicles
            vehid: vehicle ID to be used for next created vehicle
            timeind: time index
            dt: timestep
        Returns:
            None. Modifies vehicles, Lanes in place.
    """

    def increment_inflow(self, vehicles, vehid, timeind, dt):
        inflow, spd = self.get_inflow(timeind)
        self.inflow_buffer += inflow * dt

        if self.inflow_buffer >= 1:
            if self.anchor.lead is None:
                if spd is None:
                    # spd = speed_series(timeind)
                    spd = self.newveh.inv_flow(inflow, congested=False)
                out = (self.start, spd, None)
            elif method == 'ceql':
                out = eql_inflow_congested(self, inflow, c=c, check_gap=check_gap)
            elif method == 'feql':
                out = eql_inflow_free(self, inflow)
            elif method == 'shifted':
                out = shifted_speed_inflow(self, dt, shift=shift, accel_bound=accel_bound)
            elif method == 'speed':
                out = speed_inflow(self, speed_series, timeind, dt, accel_bound=accel_bound)
            elif method == 'newell':
                out = newell_inflow(self, dt, p=p, accel_bound=accel_bound)

            if out is None:
                return vehid
            # add vehicle with the given initial conditions
            pos, speed, hd = out[:]
            newveh = self.newveh
            anchor = self.anchor
            lead = anchor.lead
            newveh.lead = lead

            # initialize state
            newveh.initialize(pos, speed, hd, timeind+1)

            # update leader/follower relationships######
            # leader relationships
            if lead is not None:
                lead.fol = newveh
            for rlead in anchor.rlead:
                rlead.lfol = newveh
            newveh.rlead = anchor.rlead
            anchor.rlead = set()
            for llead in anchor.llead:
                llead.rfol = newveh
            newveh.llead = anchor.llead
            anchor.llead = set()

            # update anchor and follower relationships
            # Note that we assume that for an inflow lane, it's left/right lanes start at the same positions,
            # so that the anchors of the left/right lanes can be used as the lfol/rfol for a new vehicle.
            # This is because we don't update the lfol/rfol of AnchorVehicles during simulation.
            anchor.lead = newveh
            anchor.leadmem.append((newveh, timeind+1))
            newveh.fol = anchor

            llane = self.get_connect_left(pos)
            if llane is not None:
                leftanchor = llane.anchor
                newveh.lfol = leftanchor
                leftanchor.rlead.add(newveh)
            else:
                newveh.lfol = None
            rlane = self.get_connect_right(pos)
            if rlane is not None:
                rightanchor = rlane.anchor
                newveh.rfol = rightanchor
                rightanchor.llead.add(newveh)
            else:
                newveh.rfol = None

            # update simulation
            self.inflow_buffer += -1
            vehicles.add(newveh)

            # create next vehicle
            # cf_parameters, lc_parameters, kwargs = self.new_vehicle()
            # self.newveh = Vehicle(vehid, self, cf_parameters, lc_parameters, **kwargs)
            self.new_vehicle(vehid)
            vehid = vehid + 1
        return vehid

    return increment_inflow


class Lane:
    """Basic building block for roads/road networks.

    Lanes are responsible for defining the topology (e.g. when lanes start/end, which lanes connect
    to what, what it is possible to change left/right into) and are responsible for doing headway/distance
    calculations between Vehicles. Positions are relative to the road a lane belongs to.
    They also define boundary conditions, and are responsible for creating new vehicles and adding them
    to the network.

    Attributes:
        start: starting position of lane
        end: ending position of lane
        road: road dictionary Lane belongs to. A road has keys -
            name: string name of the road
            len: length of road
            laneinds: number of Lanes which belong to the road
            Lanes: every Lane belonging to the road is hashed by its laneind. Lanes are ordered left - right
                where the leftmost lane has index 0 and rightmost laneinds -1
            connect to: tuple with information needed for routes, see make_cur_route
        roadname: string name of road
        laneind: index of Lane
        connect_left: defines left connections for Lane
        connect_right: defines right connections for Lane
        connect_to: what the end of Lane connects to
        anchor: AnchorVehicle for lane
        roadlen: defines distance between Lane's road and other roads.
        merge_anchors: any merge anchors for the lane (see update_merge_anchors)
        events: lane events (see update_lane_events)
    """
    # TODO need a RoadNetwork object, possibly Road object as well.
    # should create common road configurations. Should different road configurations have their own
    # logics to create routes?
    # Also need a better (easier) way to allow boundary conditions to be defined, and in a more modular way
    # E.g. of good design - create road network by specifying types of roads (e.g. road, on/off ramp, merge)
    # add boundary conditions to road network.

    def __init__(self, start, end, road, laneind, connect_left=None, connect_right=None, connect_to=None,
                 downstream=None, increment_inflow=None, get_inflow=None, new_vehicle=None):
        """Inits Lane. Note methods for boundary conditions are defined (and bound) here.

        Args:
            start: starting position for lane
            end: ending position for lane
            road: road dictionary Lane belongs to
            laneind: unique index for Lane (unique to road)
            connect_left: list of tuples where each tuple is a (Lane or None, position) pair such that Lane
                is the left connection of self starting at position.
            connect_right: list of tuples where each tuple is a (Lane or None, position) pair such that Lane
                is the right connection of self starting at position.
            connect_to: Lane or None which a Vehicle transitions to after reaching end of Lane
            downstream: dictionary of keyword args which defines call_downstream method, or None
            increment_inflow: dictionary of keyword args which defines increment_inflow method, or None
            get_inflow: dictionary of keyword args which defines increment_inflow method, or None
            new_vehicle: new_vehicle method
        """
        self.laneind = laneind
        self.road = road
        self.roadname = road['name']
        # starting position/end (float)
        self.start = start
        self.end = end
        # connect_left/right has format of list of (pos (float), lane (object)) tuples where lane
        # is the connection starting at pos
        self.connect_left = connect_left if connect_left is not None else [(0, None)]
        self.connect_right = connect_right if connect_right is not None else [(0, None)]
        self.connect_to = connect_to

        if downstream is not None:
            self.call_downstream = downstream_wrapper(**downstream).__get__(self, Lane)
        """call_downstream returns an acceleration for Vehicles to use when they have no lead Vehicle.
        See downstream_wrapper for more details on specific methods and different options.

        Args:
            veh: Vehicle
            timeind: time index
            dt: timestep
        Returns:
            acceleration
        """

        if get_inflow is not None:
            self.get_inflow = get_inflow_wrapper(**get_inflow).__get__(self, Lane)
        """refer to get_inflow_wrapper for documentation"""

        if new_vehicle is not None:
            self.new_vehicle = new_vehicle.__get__(self, Lane)
        """new_vehicle generates new instance of Vehicle."""

        if increment_inflow is not None:
            self.inflow_buffer = 0
            self.newveh = None
            # cf_parameters, lc_parameters, kwargs = self.new_vehicle()  # done in Simulation.__init__
            # self.newveh = vehicle(vehid, self, cf_parameters, lc_parameters, **kwargs)
            self.increment_inflow = increment_inflow_wrapper(**increment_inflow).__get__(self, Lane)
        """refer to increment_inflow_wrapper for documentation"""

    def get_headway(self, veh, lead):
        """Calculates distance from veh to the back of lead. Assumes veh.road = self.road."""
        hd = lead.pos - veh.pos - lead.len
        if self.roadname != lead.road:
            hd += self.roadlen[lead.road]  # lead.road is hashable because its a string
        return hd

    def get_dist(self, veh, lead):
        """Calculates distance from veh to the front of lead. Assumes veh.road = self.road."""
        dist = lead.pos-veh.pos
        if self.roadname != lead.road:
            dist += self.roadlen[lead.road]  # lead.road is hashable because its a string
        return dist

    def leadfol_find(self, veh, guess, side=None):
        """Find the leader/follower for veh, in the same track as guess (can be a different track than veh's).

        Used primarily to find the new lcside follower of veh. Note that we can't use binary search because
        it's inefficient to store a sorted list of vehicles. Since we are just doing a regular search, guess
        should be close to the leader/follower.

        Args:
            veh: Vehicle to find leader/follower of
            guess: Vehicle in the track we want the leader/follower in.
            side: if side is not None, we make sure that the side leader can actually have veh as a
                opside follower. Only used for the lead Vehicle.

        Returns:
            (lead Vehicle, following Vehicle) in that order, for veh
        """
        if side is None:
            checkfol = None
        elif side == 'r':
            checkfol = 'lfol'
        else:
            checkfol = 'rfol'

        get_dist = self.get_dist
        hd = get_dist(veh, guess)
        if hd < 0:
            nextguess = guess.lead
            if nextguess is None:  # None -> reached end of network
                return nextguess, guess
            nexthd = get_dist(veh, nextguess)
            while nexthd < 0:
                # counter += 1
                guess = nextguess
                nextguess = guess.lead
                if nextguess is None:
                    return nextguess, guess
                nexthd = get_dist(veh, nextguess)

            if checkfol is not None and nextguess is not None:
                if getattr(nextguess, checkfol) is None:
                    nextguess = None
            return nextguess, guess
        else:
            nextguess = guess.fol
            if nextguess.cf_parameters is None:
                return guess, nextguess
            nexthd = get_dist(veh, nextguess)
            while nexthd > 0:
                # counter +=1
                guess = nextguess
                nextguess = guess.fol
                if nextguess.cf_parameters is None:  # reached anchor -> beginning of network
                    return guess, nextguess
                nexthd = get_dist(veh, nextguess)

            if checkfol is not None and guess is not None:
                if getattr(guess, checkfol) is None:
                    guess = None
            return guess, nextguess

    def get_connect_left(self, pos):
        """Takes in a position and returns the left connection (Lane or None) at that position."""
        return connect_helper(self.connect_left, pos)

    def get_connect_right(self, pos):
        """Takes in a position and returns the right connection (Lane or None) at that position."""
        return connect_helper(self.connect_right, pos)

    def __hash__(self):
        """Hash Lane based on its road name, and its lane index."""
        return hash((self.roadname, self.laneind))

    def __eq__(self, other):
        """Comparison for Lanes using ==."""
        return self.roadname == other.roadname and self.laneind == other.laneind
        # return self is other

    def __ne__(self, other):
        """Comparison for Lanes using !=."""
        return self is not other

    def __repr__(self):
        """Representation in ipython console."""
        return self.roadname+' ('+str(self.laneind)+')'

    def __str__(self):
        """Convert Lane to a string."""
        return self.__repr__()


def connect_helper(connect, pos):
    """Helper function takes in connect_left/right attribute, position, and returns the correct connection."""
    out = connect[-1][1]  # default to last lane for edge case or case when there is only one possible
    # connection
    for i in range(len(connect)-1):
        if pos < connect[i+1][0]:
            out = connect[i][1]
            break
    return out
