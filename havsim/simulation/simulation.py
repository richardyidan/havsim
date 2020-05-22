
"""
@author: rlk268@cornell.edu
houses the main code for running simulations 
    
    
"""

import numpy as np 
import math 
import scipy.optimize as sc 
import havsim.simulation.models as hm
    
def update_net(vehicles, lc_actions, inflow_lanes, merge_lanes, vehid, timeind, dt): 
    #update followers/leaders for all lane changes 
    for veh in lc_actions.keys(): 
        oldfol = veh.fol
        #update leader follower relationships, lane/road
        update_change(lc_actions, veh, timeind) #this cannot be done in parralel
    
        #update tact/coop components
        veh.lcside = veh.coop_veh = veh.lc_urgency = None
        #apply relaxation 
        new_relaxation(veh, timeind, dt)
        newfol = veh.fol
        if newfol.cf_parameters == None:
            pass
        else:
            new_relaxation(newfol, timeind, dt)
        if oldfol.cf_parameters == None:
            pass
        else:
            new_relaxation(oldfol, timeind, dt)
        
        
        #update a vehicle's lane events and route events for the new lane 
        set_lane_events(veh)
        set_route_events(veh)
        
    
    #update all states, memory and headway 
    for veh in vehicles: 
        veh.update(timeind, dt)
    for veh in vehicles:
        if veh.lead is not None: 
            veh.hd = veh.lane.get_headway(veh, veh.lead)
        else: #for robustness only, should not be needed
            veh.hd = None
    
    #update left and right followers
    for veh in vehicles:
        update_lrfol(veh)
    
    #update merge_anchors
    #lanes have lists of merge anchors, they are used to update vehicle order for 'new lane' or 'update lanes' events
    #merge anchors are defined as (vehicle, position) tuples. If position = None, then the vehicle is an anchor vehicle, and 
    #corresponds to the situation when a new lane starts, and the merge anchor is the anchor for the new lane. 
    #if position is not None, then the merge anchor can be either an anchor vehicle or a regular vehicle, and this corresponds 
    #to the situation where two lanes meet.  The merge anchor is a guess which is in the right track, and should 
    #ideally be right before pos 
    for curlane in merge_lanes:
        update_merge_anchors(curlane, lc_actions)
        
    #update roads (lane events) and routes
    remove_vehicles = set()
    for veh in vehicles: 
        #check vehicle's lane events and route events, acting if necessary 
        update_lane_events(veh, timeind, remove_vehicles)
        update_route(veh)
        

    #remove vehicles which leave
    for veh in remove_vehicles: 
        vehicles.remove(veh)
        
    #update inflow, adding vehicles if necessary 
    for curlane in inflow_lanes: 
        vehid = curlane.increment_inflow(vehicles, vehid, timeind, dt)
                    
    return vehid, remove_vehicles

def update_merge_anchors(curlane, lc_actions):
    for i in range(len(curlane.merge_anchors)):
        veh, pos = curlane.merge_anchors[i][:]
        if pos == None: #merge anchor is always an anchor, it just needs to have its lfol/rfol updated
            update_lrfol(veh)
        else:
            if veh.cf_parameters == None:  #veh is an anchor -> we see if we can make its leader the new merge anchor
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
                

def update_route(veh):
    #will check a vehicle's current route events and see if we need to do anything 
    #returns True if we make a change, False otherwise
    #expect a list of dictionarys, each dict is an event with keys of pos, event, side 
    #giving the position when the event occurs, event type, and side of the event 
    if len(veh.route_events) == 0:
        return False
    curevent = veh.route_events[0]
    if veh.pos > curevent['pos']:
        
        if curevent['event'] == 'end discretionary': 
            side = curevent['side']
            setattr(veh, side ,None)
            if veh.lcside == side: #end tactical/coop if necessary 
                veh.coop_veh = veh.lcside = None
                
        elif curevent['event'] == 'mandatory': 
            setattr(veh, curevent['side'],'mandatory')
            veh.lc_urgency = curevent['lc_urgency'] #must always set urgency for mandatory changes
        veh.route_events.pop(0)
        return True
    return False

def update_lane_events(veh, timeind, remove_vehicles): 
    if len(veh.lane_events) == 0: 
        return
    curevent = veh.lane_events[0]
    if veh.pos > curevent['pos']:
        if curevent['event'] == 'new lane': #needs 'left', 'left merge', 'right' 'right anchor' 'pos'
            #update lane/road/position
            newlane = veh.lane.connect_to
            update_veh_lane(veh, veh.lane, newlane, timeind+1)
            
            #need to update vehicle orders and lane change state
            update_lane_helper(veh, newlane, curevent)
            
            #enter new road/lane = need new lane/route events
            set_lane_events(veh)
            set_route_events(veh)
            
        
        elif curevent['event'] == 'update lr': #needs 'left', 'left merge', 'right' 'right anchor' 'pos'
            update_lane_helper(veh, veh.lane, curevent)
            veh.lane_events.pop(0) #event is over, move to next 
        
        elif curevent['event'] == 'exit':
            #update vehicle orders
            #there shouldn't be any leaders
            fol = veh.fol
            # if veh.lead is not None:
            #     veh.lead.fol = fol
            # for i in veh.llead:
            #     i.rfol = fol
            # for i in veh.rlead: 
            #     i.lfol = fol
                
            #update followers
            fol.lead = None
            fol.leadmem.append((None, timeind+1))
            if veh.lfol != None: 
                veh.lfol.rlead.remove(veh)

            if veh.rfol != None: 
                veh.rfol.llead.remove(veh)
            
            #to remove the vehicle set its endtime and put it in the remove_vehicles
            veh.endtime = timeind
            remove_vehicles.add(veh)
    return
            
                
        
        
def update_lane_helper(veh, curlane, curevent):
    #for a vehicle veh which reaches a point where its curlane.get_connect_left or get_connect_right
    #go from None to a lane, or a lane to None, there needs to be 'add' or 'remove' events
    #this handles those events
    
    #updates the vehicle orders and defaults the lane change states to the correct behavior 
    #(enter discretionary by default only if lane is in same road)
    if curevent['left'] == 'remove': 
        veh.lfol.rlead.remove(veh)
        veh.lfol = None
        veh.l = None
        veh.llane = None
        if veh.lcside == 'l': #end tactical/coop if necessary 
                veh.coop_veh = veh.lcside = None
    elif curevent['left'] == 'add':
        newllane = curlane.get_connect_left(curevent['pos'])
        merge_anchor = newllane.merge_anchors[curevent['left anchor']][0]
        unused, newfol = curlane.leadfol_find(veh,merge_anchor,'l')
        
        veh.lfol = newfol
        newfol.rlead.add(veh)
        if newllane.roadname == curlane.roadname:
            veh.l = 'discretionary'
        else:
            veh.l = None
        veh.llane = newllane
        
    
    #same thing for right
    if curevent['right'] == 'remove': 
        veh.rfol.llead.remove(veh)
        veh.rfol = None
        veh.r = None
        veh.rlane = None
        if veh.lcside == 'r': #end tactical/coop if necessary 
                veh.coop_veh = veh.lcside = None
    elif curevent['right'] == 'add': 
        newrlane  = curlane.get_connect_right(curevent['pos'])
        merge_anchor = newrlane.merge_anchors[curevent['right anchor']][0]
        unused, newfol = curlane.leadfol_find(veh, merge_anchor, 'r')
        
        veh.rfol = newfol
        newfol.llead.add(veh)
        if newrlane.roadname == curlane.roadname:
            veh.r = 'discretionary'
        else:
            veh.r = None
        veh.rlane = newrlane

def set_lane_events(veh):
    veh.lane_events = []
    for i in veh.lane.events: 
        if i['pos'] > veh.pos: 
            veh.lane_events.append(i)
    

def make_cur_route(p, curlane, nextroadname): 
    #generates route events with parameters p in lane lane 
    #p - parameters - currently len 2 list with constant, p[0] is a constant which is a like a safety buffer, and p[1]
    #controls the distance you need for a change (also a constant)
    #curlane- lane object that the route events start on 
    #nextroad - once you leave curlane.road, you want to end up on nextroad
    
    #output - dictionary where keys are lanes, values are the route events a vehicle with 
    #parameters p needs to follow on that lane. 
    
    #explanation of current model - 
    #if  you need to be in lane '2' by position 'x' and start in lane '1', 
    #then starting at x - 2*p[0] - 2*p[1] you will end discretionary changing into lane '0'
    #at x - p[0] - p[1] you wil begin mandatory changing into lane '2'
    #at x - p[0] your mandatory change will have urgency of 100% which will always force cooperation of your l/rfol 
    #for merging onto/off an on-ramp which begins at 'x' and ends at 'y', you will start mandatory at 'x' always, 
    #reaching 100% cooperation by 'y' - p[0]
    
    #TODO low priority we only get the route for the current road - no look ahead to take into account future roads. 
    #(This is non trivial, be careful. Should only have to look forward one road )
    
    #TODO nothing to handle cases where LC cannot be completed successfully // medium priority (put necessary info into cur_route dict) 
    ####if you fail to follow your route, you need to be given a new route, and the code which does this can also be used in vehicle.set_state if route = None at initialization####
    #would need to know latest point when change can take place ('pos' for 'continue' type or pos[1] for merge type), 
    #in lane changing model, it would need to check if we are getting too close and act accordingly (e.g. slow down) if so 
    #in this function, would need to add events if you miss the change, and in that case you would need to be given a new route 
    
    curroad = curlane.road
    curlaneind = curlane.laneind
    #position or tuple of positions, str, tuple of 2 ints or single int, str, dict for the next road
    pos, change_type, laneind, side, nextroad  = curroad['connect to'][nextroadname][:]
    #roads also have 'name', 'len', 'laneinds', all lanes are values with their indexes as keys 
    
    cur_route = {}
    
    if change_type == 'continue': #-> vehicle needs to reach end of lane to transition to next road or exit simulation
        #initialize for lanes which vehicle needs to continue on 
        leftind, rightind = laneind[:]
        for i in range(leftind, rightind+1):
            cur_route[curroad[i]] = []
            
        if leftind > 0:
            templane = curroad[leftind]
            curpos = min(templane.end, curroad[leftind-1].end)
            cur_route[templane].append({'pos': curpos - p[0] - p[1], 'event': 'end discretionary', 'side': 'l'})
        
        if rightind< curroad['laneinds']-1:
            templane = curroad[rightind]
            curpos = min(templane.end, curroad[rightind+1].end)
            cur_route[templane].append({'pos': curpos - p[0] - p[1], 'event': 'end discretionary', 'side': 'r'})
        
        if curlaneind >= leftind and curlaneind <= rightind: #if on correct lane(s) already, do no more work 
            return cur_route
        
        elif curlaneind < laneind[0]: #need to change right possibly multiple times
            uselaneind = laneind[0]
        else: 
            uselaneind = laneind[1]
            
        cur_route = make_route_helper(p, cur_route, curroad, curlaneind, uselaneind, curroad[uselaneind].end)
                
            
    elif change_type =='merge': #logic is similar and also uses make_route_helper
        templane = curroad[laneind]
        pos, endpos = pos[:]
        cur_route[templane] = []
        
        #determine end discretionary event if necessary 
        if side == 'l': 
            if laneind < curroad['laneinds'] - 1: 
                enddisc = min(pos, curroad[laneind+1].end)
                cur_route[templane].append({'pos': enddisc -p[0] - p[1], 'event':'end discretionary', 'side':'r'})
        else: 
            if laneind > 0: 
                enddisc = min(pos, curroad[laneind-1].end)
                cur_route[templane].append({'pos': enddisc -p[0] - p[1], 'event':'end discretionary', 'side':'l'})


        cur_route[templane].append({'pos': pos, 'event':'mandatory', 'side':side, 'lc_urgency': [pos, endpos - p[0]]})
        
        if curlaneind != laneind:
            cur_route = make_route_helper(p, cur_route, curroad, curlaneind, laneind, pos)
    
    return cur_route
    
def make_route_helper(p, cur_route, curroad, curlaneind, laneind, curpos):
    #p - parameters for route 
    #cur_route - dictionary to add entries to 
    #curroad - current road 
    #curlaneind - current index of lane you start in 
    #laneind, curpos - index of lane you want to be in by position curpos 
    
    #starting on curroad in lane with index curlaneind, and wanting to be in laneind by curpos position, 
    #generates routes cur all roads in [curlaneind, laneind)
    #assumes you already have the route for laneind
    #edge cases where routes have different lengths are handled. 
    if curlaneind < laneind:
        curind = laneind - 1
        prevtemplane = curroad[curind+1]
        templane = curroad[curind]
        cur_route[templane] = []
        while not (curind < curlaneind):
            #determine curpos = where the mandatory change starts (different meaning than the 'curpos' which is passed in)
            if templane.end < curpos: #in case templane ends before the curpos
                curpos = templane.end
            curpos += -p[0] - p[1]
            curpos = max(prevtemplane.start, curpos) #in case the lane doesn't exist at curpos
            
            #determine enddiscpos = where the discretionary ends
            #only necessary if there is something to end the discretionary into
            if curind > 0: 
                nexttemplane = curroad[curind-1]
                enddiscpos = min(curpos, nexttemplane.end)
                enddiscpos = enddiscpos - p[0] - p[1]
                cur_route[templane].append({'pos': enddiscpos, 'event': 'end discretionary', 'side': 'l'})
            
            #there is always a mandatory event
            cur_route[templane].append({'pos': curpos, 'event': 'mandatory', 'side': 'r', 'urgency':[curpos, curpos + p[1]]})
            
            #update iteration 
            curind += -1 
            prevtemplane = templane
            templane = nexttemplane
            
    #same code but for opposite side 
    elif curlaneind > laneind: 
        curind = laneind +1
        prevtemplane = curroad[curind - 1]
        templane = curroad[curind]
        cur_route[templane] = []
        while not (curind > curlaneind):
            #determine curpos = where the mandatory change starts 
            if templane.end < curpos: 
                curpos = templane.end
            curpos += -p[0] - p[1]
            curpos = max(prevtemplane.start, curpos)
            
            if curind < curroad['laneinds'] - 1: 
                nexttemplane = curroad[curind + 1]
                enddiscpos = min(curpos, nexttemplane.end)
                enddiscpos = enddiscpos - p[0] - p[1]
                cur_route[templane].append({'pos': enddiscpos, 'event': 'end discretionary', 'side': 'r'})
                
            
            cur_route[templane].append({'pos': curpos, 'event': 'mandatory', 'side': 'l', 'urgency':[curpos, curpos + p[1]]})
            
            #update iteration 
            curind += -1 
            prevtemplane = templane
            templane = nexttemplane

    return cur_route
        
def set_route_events(veh):
    #when a vehicle enters a new road, they will initialize lane events for a number of lanes on the road (stored in cur_route dict)
    #cur_route is created by make_route_helper - at that point we also pop from the veh.route
    #if a vehicle enters a lane on the same road, which is not in cur_route, the make_route_helper adds the lane events for the new lane
    
    # if veh.route == []:  #for testing purposes for infinite road only#########
        # return 
    
    #get new route events if they are stored in memory already 
    newlane = veh.lane 
    if newlane in veh.cur_route: 
        veh.route_events = veh.cur_route[newlane].copy() #use shallow copy - copy references only
    # #otherwise we will make it  #now made in initialize
    # elif len(veh.lanemem) == 1: #only possible when vehicle first enters simulation
    #     veh.cur_route = make_cur_route(veh.route_parameters, newlane, veh.route.pop())
        
    #     veh.route_events = veh.cur_route[veh.lane].copy()
    else:
        p = veh.route_parameters
        prevlane = veh.lanemem[-2][0]
        if prevlane.road is newlane.road: #on same road - we can just use helper function to update cur_route
            prevlane_events = veh.cur_route[prevlane] #need to figure out what situation we are in to give make route helper right call
            if len(prevlane_events) == 0: #this can only happen for continue event => curpos = the end of lane
                curpos = prevlane.end
            elif prevlane_events[0]['event'] == 'end discretionary':
                curpos = prevlane_events[0]['pos'] + p[0] + p[1]
            else: #mandatory event
                curpos = prevlane_events[0]['pos']
            make_route_helper(p, veh.cur_route, veh.road, newlane.laneind, prevlane.laneind, curpos)
        else: #on new road - we need to generate new cur_route and update the vehicle's route
            veh.cur_route = make_cur_route(p, newlane, veh.route.pop(0))
            
        
        veh.route_events = veh.cur_route[newlane].copy()
    
    #for route events, past events need to be applied. This is different for lane events, 
    #where past events are not applied. 
    curbool = True
    while curbool: 
        curbool = update_route(veh)
        
    return 
    

def update_lrfol(veh):
    lfol, rfol = veh.lfol, veh.rfol
    if lfol == None:
        pass
    elif veh.lane.get_dist(veh,lfol) > 0: 
        veh.lfol = lfol.fol
        veh.lfol.rlead.add(veh)
        lfol.rlead.remove(veh)
        

        lfol.rfol.llead.remove(lfol) 
        lfol.rfol = veh
        veh.llead.add(lfol)
        
    if rfol == None:
        pass
    elif veh.lane.get_dist(veh,rfol) > 0: 
        veh.rfol = rfol.fol
        veh.rfol.llead.add(veh)
        rfol.llead.remove(veh)
    
        rfol.lfol.rlead.remove(rfol)
        rfol.lfol = veh
        veh.rlead.add(rfol)
        
        
        
    

def new_relaxation(veh,timeind, dt):
    rp = veh.relaxp
    if veh.lead == None or rp == None: 
        return
    olds = veh.hd
    news = veh.lane.get_headway(veh, veh.lead)
    if olds is None: 
        olds = veh.get_eql(veh.speed)
    
    relaxamount = olds-news
    relaxlen = math.ceil(rp/dt) - 1
    curr =  relaxamount*np.linspace(1 - dt/rp, 1 - dt/rp*relaxlen,relaxlen)
    
    if veh.in_relax: #add to existing relax
        curlen = len(veh.relax)
        newend = timeind + relaxlen #time index when relax ends 
        newrelax = np.zeros((newend - veh.relax_start+1))
        newrelax[0:curlen] = veh.relax
        newrelax[timeind-veh.relax_start+1:] += curr
        veh.relax = newrelax
    else: #create new relax
        veh.in_relax = True
        veh.relax_start = timeind + 1
        veh.relax = curr
    
    return

def update_veh_lane(veh, oldlane, newlane, timeind, side = None):
    #for veh which was previously on oldlane and moves to newlane at timeind, 
    #updates road, lane, lanemem, position, 
    #if side is not None, also updates the 'r'/'l' attributes (if a vehicle changes to left, you need to update r attribute)
    #by default, the l/r attributes are set to discretionary only if the corresponding lane is in the same road 
    #changes to different roads are controlled by route events
    
    # newroad = newlane.road 
    newroadname = newlane.roadname
    if side == None:
        if newroadname != veh.road: 
            veh.pos += -oldlane.roadlen[newroadname]
            veh.road = newroadname
    else: 
        if newroadname != veh.road: 
            veh.pos += -oldlane.roadlen[newroadname]
            veh.road = newroadname
            setattr(veh,side,None)
        else: 
            setattr(veh,side,'discretionary')
    veh.lane = newlane
    veh.lanemem.append((newlane, timeind))
    return

#in current logic, main cost per timestep is just one distance compute in update_lrfol
#whenever there is a lane change, there are a fair number of extra updates we have to do to keep all 
#of the rlead/llead updated. Also, whenever an lfol/rfol changes, there are two rlead/lead attributes 
#that need to be changed as well. 
#Thus this strategy is very efficient assuming we want to keep lfol/rfol updated (call lc every timestep), lane changes aren't 
#super common, and all vehicles travel at around the same speed. (which are all reasonable assumptions)
    
#naive way would be like having to do something like keep a sorted list, every time we want lfol/rfol
#we have to do log(n) dist computes, where n is the number of vehicles in the current lane. 
#whenever a vehicle changes lanes, you need to remove from the current list and add to the new, 
#so it is log(n) dist computations + 2n for searching/updating the 2 lists. Thus the current implementation is definitely much better than the naive way. 
    
#Another option you could do is to only store lfol/rfol, to keep it updated you would have to 
#do 2 dist calculations per side per timestep (do a call of leadfol find where we already have either a follower or leader as guess). 
#When there is a lane change store a dict which has the lane changing vehicle as a key, and store as the value the new guess to use. 
#in lfol/rfol update, you know there was a lane change if your fol is in the wrong lane Then can get a new guess for the fol from the dict. 
#This strategy would have higher costs per timestep to keep lfol/rfol updated, but would be simpler to update when there is a lane change. 
#Thus it might be more efficient if the timesteps are long relative to the number of lane changes. 
#Overall I doubt there would be much practical difference between this option and the first option unless the timesteps are very long (~10-15 sec) and changes very often (every ~30 seconds)
def update_change(lc_actions, veh, timeind): 
    
    #TODO no check for vehicles moving into same gap low priority 
    
    #initialization, update lane/road/position and update l/r attributes
    if lc_actions[veh] == 'l':
        #update lane/road/position, and the l/r attributes
        veh.rlane = veh.lane
        # lcsidelane = veh.lane.get_connect_left(veh.pos) 
        lcsidelane = veh.llane
        update_veh_lane(veh, veh.lane, lcsidelane, timeind+1, 'r')
        #update new lcside lane change attribute
        newlcsidelane = lcsidelane.get_connect_left(veh.pos)
        veh.llane = newlcsidelane
        if newlcsidelane != None and newlcsidelane.roadname == veh.road:
            veh.l = 'discretionary'
        else:
            veh.l = None
            
        
    else: 
        veh.llane = veh.lane
        # lcsidelane = veh.lane.get_connect_right(veh.pos)
        lcsidelane = veh.rlane
        update_veh_lane(veh, veh.lane, lcsidelane, timeind+1, 'l')
        
        newlcsidelane = lcsidelane.get_connect_right(veh.pos)
        veh.rlane = newlcsidelane
        if newlcsidelane != None and newlcsidelane.roadname == veh.road: 
            veh.r = 'discretionary'
        else:
            veh.r = None
    
        
    ######update all leader/follower relationships#####
    
    update_leadfol_after_lc(veh, lcsidelane, newlcsidelane, lc_actions[veh], timeind)
            
    return 

def update_leadfol_after_lc(veh, lcsidelane, newlcsidelane, side, timeind):
    #updates all logics for keeping vehicle orders updated after vehicle 
    #performs a lane change in side (either 'l' or 'r') which causes it to transition to lane lcsidelane
    #with a new lcsidelane at timeind
    if side == 'l': 
        #define lcside/opside
        lcsidefol, opsidefol, lcsidelead, opsidelead = 'lfol', 'rfol', 'llead', 'rlead'
    else:
        lcsidefol, opsidefol, lcsidelead, opsidelead = 'rfol', 'lfol', 'rlead', 'llead'
    
    #update current leader
    lead = veh.lead 
    fol = veh.fol
    if lead == None: 
        pass
    else: 
        lead.fol = fol
    
    #update opposite/lc side leaders
    for j in getattr(veh, opsidelead):
        setattr(j, lcsidefol, fol)
    for j in getattr(veh, lcsidelead):
        setattr(j, opsidefol, fol)
    
    #update follower
    getattr(fol,lcsidelead).update(getattr(veh, lcsidelead))
    getattr(fol,opsidelead).update(getattr(veh, opsidelead))
    fol.lead = lead
    fol.leadmem.append((lead, timeind+1))
    
    #update opposite side for vehicle 
    vehopsidefol = getattr(veh, opsidefol)
    if vehopsidefol != None: 
        getattr(vehopsidefol, lcsidelead).remove(veh)
    setattr(veh, opsidefol, fol)
    getattr(fol, lcsidelead).add(veh)
    #update cur lc side follower for vehicle 
    lcfol = getattr(veh, lcsidefol)
    lclead = lcfol.lead
    lcfol.lead = veh
    lcfol.leadmem.append((veh, timeind+1))
    getattr(lcfol, opsidelead).remove(veh)
    veh.fol = lcfol
    #update lc side leader
    veh.lead = lclead
    veh.leadmem.append((lclead, timeind+1))
    
    if lclead is not None: 
        lclead.fol = veh
    #update for new left/right leaders - opside first 
    newleads = set()
    oldleads = getattr(lcfol, opsidelead)
    for j in oldleads.copy(): 
        curdist = lcsidelane.get_dist(veh,j)
        if curdist > 0: 
            setattr(j, lcsidefol, veh)
            newleads.add(j)
            oldleads.remove(j)
    setattr(veh, opsidelead, newleads)
    #lcside 
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
                minveh = j #minveh is the leader of new lc side follower 
    setattr(veh, lcsidelead, newleads)
    
    #update new lcside leaders/follower
    if newlcsidelane is None: 
        setattr(veh, lcsidefol, None)
    else: 
        if minveh is not None: 
            setattr(veh, lcsidefol, minveh.fol)
            getattr(minveh.fol,opsidelead).add(veh)
        else: 
            guess = get_guess(lcfol, lclead, veh, lcsidefol, newlcsidelane)
            unused, newlcsidefol = lcsidelane.leadfol_find(veh, guess, side)
            setattr(veh, lcsidefol, newlcsidefol)
            getattr(newlcsidefol, opsidelead).add(veh)
    
        
def get_guess(lcfol, lclead, veh, lcsidefol, newlcsidelane):
    #need to find new lcside follower for veh
    guess = getattr(lcfol, lcsidefol)
    anchor = newlcsidelane.anchor
    if guess == None or guess.lane.anchor is not anchor: 
        if lclead != None:
            guess = getattr(lclead, lcsidefol)
            if guess == None or guess.lane.anchor is not anchor: 
                guess = anchor
        else:
            guess = anchor
    return guess 

class simulation: 
    def __init__(self, inflow_lanes, merge_lanes, vehicles = None, all_vehicles = None, vehid = 0, timeind = 0, dt = .25): 
        self.inflow_lanes = inflow_lanes
        self.merge_lanes = merge_lanes
        self.vehicles = set() if vehicles == None else vehicles
        self.all_vehicles = set() if all_vehicles == None else all_vehicles
        self.vehid = vehid
        self.timeind = timeind
        self.dt = dt
        
        for curlane in inflow_lanes: 
            cf_parameters, lc_parameters, kwargs = curlane.new_vehicle()
            curlane.newveh = vehicle(self.vehid, curlane, cf_parameters, lc_parameters, **kwargs)
            self.vehid += 1
        
    
    def step(self, timeind):
        lc_actions = {}
        
        for veh in self.vehicles: 
            veh.set_cf(self.timeind, self.dt)
            
        for veh in self.vehicles: 
            veh.set_lc(lc_actions, self.timeind, self.dt)
            
        self.vehid, remove_vehicles = update_net(self.vehicles, lc_actions, self.inflow_lanes, self.merge_lanes, self.vehid, self.timeind, self.dt)
        
        self.timeind += 1
        self.all_vehicles.update(remove_vehicles)
        
    def simulate(self, timesteps):
        for i in range(timesteps):
            self.step(self.timeind)
        
    def reset(): #TODO - ability to put simulation back to earlier time
        pass

def get_eql_helper(veh, x, input_type = 'v', eql_type = 'v', spdbounds = (0, 1e4), hdbounds = (0, 1e4), tol = .1):
    #veh - vehicle object - needs cf_parameters attribute and eqlfun method
        #eqlfun has call signature of (cf_parameters, x) and returns a float
    #x - input, can be either a spsed or headway 
    #input_type = 'v' if 'v' we input a speed and return a headway; if 's' we input headway and return a speed
    #eql_type = 's' - if 'v', eqlfun takes in velocity and outputs headway. if 's', it takes in headway and outputs velocity
    #spdbounds = (0, 1e4), hdbounds = (0, 1e4) - x is projected onto the relevant bounds. 
    #if eqlfun is inverted, the appropriate bounds are used 
    #tol = .1 - if need to numerically invert the eqlfun, this is the tolerance used for the solver
    
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
        ans = sc.root_scalar(inveql, bracket = bracket, xtol = tol, method = 'brentq')
        if ans.converged: 
            return ans.root
        else: 
            raise RuntimeError('could not invert provided equilibrium function')
        
def inv_flow_helper(veh, x, leadlen = None, output_type = 'v', congested = True, eql_type = 'v', spdbounds = (0, 1e4), hdbounds = (0, 1e4), tol = .1, ftol = .01):
    if leadlen == None: 
        lead = veh.lead 
    if lead != None: 
        leadlen = lead.len
    else:
        leadlen = veh.len

    if eql_type == 'v': 
        bracket = spdbounds
    else:
        bracket = hdbounds
        
    def maxfun(y):
        return -veh.get_flow(y, leadlen = leadlen, input_type = eql_type)
    res = sc.minimize_scalar(maxfun, bracket = bracket, options = {'xatol':ftol}, method = 'bounded', bounds = bracket)
    
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
                return veh.get_eql(res['x'], input_type = eql_type)
    else:
        raise RuntimeError('could not find maximum flow')
        
    if eql_type == 'v':
        def invfun(y): 
            return x - y/(veh.eqlfun(veh.cf_parameters, y) + leadlen)
    elif eql_type == 's':
        def invfun(y):
            return x - veh.eqlfun(veh.cf_parameters, y)/(y+leadlen)
    
    ans = sc.root_scalar(invfun, bracket = invbounds, xtol = tol, method = 'brentq')
    
    if ans.converged:
        if output_type == eql_type: 
            return ans.root
        elif output_type == 's':
            return ans.root/x - leadlen
        elif output_type == 'v':
            return (ans.root+leadlen)*x
    else: 
        raise RuntimeError('could not invert provided equilibrium function')
    
def set_lc_helper(veh, chk_lc = 1, get_fol = True):
    #calculates new headways to pass to lane changing model
    #veh - vehicle object
    #chk_lc = 1 - probability of checking lane changing model
    #get_fol = True - if True, we calculate the headways for the follower
    
    #returns 
    #call_model - True if we are to pass to LC model, False otherwise
    #(lside, rside, newlfolhd, newlhd, newrfolhd, newrhd, newfolhd)
    #where lside/rside are boolean indicating which sides to call model on, rest are headways
    
    #first determine what situation we are in and which sides we need to check
    l, r  = veh.l, veh.r
    if l == None: 
        if r == None: 
            return False, None
        elif r == 'discretionary':
            lside, rside = False, True
            chk_cond = False if veh.lcside != None else True
        else: 
            lside, rside = False, True
            chk_cond = False
    elif l == 'discretionary': 
        if r == None:
            lside,rside = True, False
            chk_cond = False if veh.lcside != None else True
        elif r == 'discretionary': 
            if veh.lcside != None: 
                chk_cond = False
                if veh.lcside == 'l': 
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
        if r == None: 
            lside, rside = True, False
            chk_cond = False
        elif r == 'discretionary': 
            lside, rside = True, False
            chk_cond = False
            
    if chk_cond: #decide if we want to evaluate lc model or not - applies to discretionary state when there is no cooperation or tactical positioning
        if chk_lc >= 1:
            pass
        elif np.random.rand() > chk_lc: 
            return False, None 
        
    #next we compute quantities to send to LC model for the required sides 
    if lside: 
        newlfolhd, newlhd = get_new_hd(veh.lfol, veh, veh.llane) #better to just store left/right lanes
    else:
        newlfolhd = newlhd = None
    
    if rside: 
        newrfolhd, newrhd = get_new_hd(veh.rfol, veh, veh.rlane)
    else:
        newrfolhd = newrhd = None
        
    #if get_fol option is given to wrapper, it means model requires the follower's quantities as well 
    if get_fol: 
        fol, lead = veh.fol, veh.lead
        if fol.cf_parameters == None: 
            newfolhd = None 
        elif lead == None: 
            newfolhd = None
        else: 
            newfolhd = fol.lane.get_headway(fol, lead)
        
    return True, (lside, rside, newlfolhd, newlhd, newrfolhd, newrhd, newfolhd)

def get_new_hd(lcsidefol, veh, lcsidelane):
    #does headway calculation for new potential follower lfol (works for either side)
    #helper functino for set_lc_helper
    lcsidelead = lcsidefol.lead
    if lcsidelead == None: 
        newlcsidehd = None
    else: 
        newlcsidehd = lcsidelane.get_headway(veh, lcsidelead)
    if lcsidefol.cf_parameters == None:
        newlcsidefolhd = None
    else: 
        newlcsidefolhd = lcsidefol.lane.get_headway(lcsidefol, veh)
    
    return newlcsidefolhd, newlcsidehd

class vehicle: 
    #TODO implementation of adjoint method for cf, relax, shift parameters
    def __init__(self, vehid,curlane, p, lcp,
                 lead = None, fol = None, lfol = None, rfol = None, llead = None, rlead = None,
                 length = 3, eql_type = 'v',
                 relaxp = 12,  shiftp = None, coopp = .2, 
                 routep = None, route = None,
                 accbounds = None, maxspeed = 1e4, hdbounds = None): 
        self.vehid = vehid
        self.len = length
        self.lane = curlane
        self.road = curlane.road['name'] if curlane != None else None
        #model parameters
        self.cf_parameters = p
        self.lc_parameters = lcp
        
        #relaxation
        self.relaxp = relaxp
        self.in_relax = False
        self.relax = None
        self.relax_start = None
        
        #route parameters
        self.route_parameters = [30,120] if routep == None else routep
        self.route = [] if route == [] else route
        #TODO check if route is empty
        self.routemem = self.route.copy()
        
        #bounds
        if accbounds == None:
            self.minacc, self.maxacc = -7, 3
        else:
            self.minacc, self.maxacc = accbounds[0], accbounds[1]
        self.maxspeed = maxspeed    
        self.hdbounds = (0, 1e4) if hdbounds == None else hdbounds
        self.eql_type = eql_type
        
        #cooperative/tactical model
        self.shiftp = [.4,2] if shiftp == None else shiftp
        self.coopp = coopp
        self.lcside = None
        self.lc_urgency = None
        self.coop_veh = None
        
        
        #leader/follower relationships
        self.lead = lead
        self.fol = fol
        self.lfol = lfol
        self.rfol = rfol
        self.llead = llead
        self.rlead = rlead
        
        #memory
        self.endtime = None
        self.leadmem = []
        self.lanemem = []
        self.posmem = []
        self.speedmem = []
        self.relaxmem = []
    
            
    def initialize(self, pos, spd, hd, inittime):
        #state
        self.pos = pos
        self.speed = spd
        self.hd = hd

        #memory 
        self.inittime = inittime
        self.leadmem.append((self.lead, inittime))
        self.lanemem.append((self.lane, inittime))
        self.posmem.append(pos)
        self.speedmem.append(spd)
        
        #llane/rlane and l/r 
        self.llane = self.lane.get_connect_left(pos)
        if self.llane== None:
            self.l = None
        elif self.llane.roadname == self.road: 
            self.l = 'discretionary'
        else:
            self.l = None
        self.rlane = self.lane.get_connect_right(pos)
        if self.rlane== None:
            self.r = None
        elif self.rlane.roadname == self.road: 
            self.r = 'discretionary'
        else:
            self.r = None
        
        #set lane/route events - sets lane_events, route_events, cur_route attributes
        self.cur_route = make_cur_route(self.route_parameters, self.lane, self.route.pop(0))
        self.route_events = self.cur_route[self.lane].copy()
        set_lane_events(self)
        set_route_events(self)
        
        
    @staticmethod
    def cfmodel(p, state):
    #state = headway, velocity, lead velocity
    #p = parameters
    #returns acceleration 
        return p[3]*(1-(state[1]/p[0])**4-((p[2]+state[1]*p[1]+(state[1]*(state[1]-state[2]))/(2*(p[3]*p[4])**(1/2)))/(state[0]))**2)
            
    def get_cf(self, hd, spd, lead, curlane, timeind, dt, userelax): 
        if lead is None: 
            acc = curlane.call_downstream(self, timeind, dt)
            
        else:
            if userelax:
                currelax = self.relax[timeind - self.relax_start]
                hd += currelax #can add check to see if relaxed headway is too small
                acc = self.cfmodel(self.cf_parameters, [hd, spd, lead.speed])
                hd += -currelax
            else: 
                acc = self.cfmodel(self.cf_parameters, [hd, spd, lead.speed])
            
        return acc
    
    def set_cf(self, timeind, dt):
        self.acc = self.get_cf(self.hd, self.speed, self.lead, self.lane, timeind, dt, self.in_relax)
        
    @staticmethod
    def free_cf(p, spd):
        return p[3]*(1-(spd/p[0])**4)
    
    @staticmethod
    def eqlfun(p, v):
        #p = parameters
        #v - velocity
        s = ((p[2]+p[1]*v)**2/(1- (v/p[0])**4))**.5
        return s 
    
    def get_eql(self, x, input_type = 'v'):
        #if input type is 'v', x is a speed and we return a headway s such that (s,v) is an equilibrium solution
        #if input_type is 's', x is a headway and we return a speed
        return get_eql_helper(self, x, input_type, self.eql_type, (0, self.maxspeed), self.hdbounds)
    
    def get_flow(self, x, leadlen = None, input_type = 'v'):
        if leadlen == None: 
            lead = self.lead 
            if lead != None: 
                leadlen = lead.len
            else:
                leadlen = self.len
        if input_type == 'v':
            s = self.get_eql(x, input_type = input_type)
            return x / (s + leadlen)
        elif input_type == 's': 
            v = self.get_eql(x, input_type = input_type)
            return v / (s + leadlen)
        
    def inv_flow(self, x, leadlen = None, output_type = 'v', congested = True):
        return inv_flow_helper(self, x, leadlen, output_type, congested, self.eql_type, (0, self.maxspeed), self.hdbounds)
    
    @staticmethod
    def shift_eql(p, v, shiftp, state):
        #p = CF parameters
        #v = velocity 
        #shiftp = list of deceleration, acceleration parameters. eq'l at v goes to n times of normal, where n is the parameter
        #state = if state = 'decel' we use shiftp[0] else shiftp[1]
        if state == 'decel':
            temp = shiftp[0]**2
        else:
            temp = shiftp[1]**2
            
        return (temp - 1)/temp*p[3]*(1 - (v/p[0])**4)
    
    def set_lc(self, lc_actions, timeind, dt):
        call_model, args = set_lc_helper(self, self.lc_parameters[-1]*dt)
        if call_model: 
            hm.mobil(self, lc_actions, *args, timeind, dt)
        return
    
            
    def update(self, timeind, dt): 
        #bounds on acceleration
        acc = self.acc
        if acc > self.maxacc: 
            acc = self.maxacc
        elif acc < self.minacc: 
            acc = self.minacc
        
        #bounds on speed
        temp = acc*dt
        nextspeed = self.speed + temp
        if nextspeed < 0: 
            nextspeed = 0
            temp = -self.speed
        elif nextspeed > self.maxspeed:
            nextspeed = self.maxspeed
            temp = self.maxspeed - self.speed
        
        
        #update state
        self.pos += self.speed*dt + .5*temp*dt
        self.speed = nextspeed
        
        #update memory and relax
        self.posmem.append(self.pos)
        self.speedmem.append(self.speed)
        if self.in_relax:
            if timeind == self.relax_start + len(self.relax) - 1:
                self.in_relax = False
                self.relaxmem.append((self.relax_start, timeind, self.relax))
                
        
    def __hash__(self):
        return hash(self.vehid)
        
    def __eq__(self, other):
        # return self.vehid == other.vehid
        return self is other
    
    def __ne__(self, other):
        return not(self is other)
        
    def __repr__(self):
        #display string in interactive console
        return ('vehicle '+str(self.vehid)+' on lane '+str(self.lane)+' at position '+str(self.pos))
    
    def __str__(self):
        return self.__repr__()
    
    def __leadfol__(self):
        print('-------leader and follower-------')
        if self.lead == None:
            print('No leader')
        else:
            print('leader is '+str(self.lead))
        print('follower is '+str(self.fol))
        print('-------left and right followers-------')
        if self.lfol == None:
            print('no left follower')
        else:
            print('left follower is '+str(self.lfol))
        if self.rfol == None:
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
    
    def __chk_leadfol__(self, verbose = False):
        lfolpass = True
        if self.lfol != None: 
            if self.lfol is self:
                lfolpass = False
                lfolflag = 0
            if self not in self.lfol.rlead:
                lfolpass = False
                lfolflag = 1
            if self.lfol.lane.anchor is not self.llane.anchor:
                lfolpass = False
        rfolpass = True
        if self.rfol != None: 
            if self.rfol is self:
                rfolpass = False
                rfolflag = 0
            if self not in self.rfol.llead:
                rfolpass = False
                rfolflag = 1
            if self.rfol.lane.anchor is not self.rlane.anchor:
                rfolpass = False
        rleadpass = True
        for i in self.rlead: 
            if i.lfol is not self:
                rleadpass = False
        lleadpass = True
        for i in self.llead: 
            if i.rfol is not self:
                lleadpass = False
        leadpass = True
        if self.lead != None: 
            if self.lead.fol is not self: 
                leadpass = False
        folpass = True
        if self.fol.lead is not self:
            folpass = False
            
        if verbose: 
            print('lfol passing: '+str(lfolpass))
            print('rfol passing: '+str(rfolpass))
            print('fol passing: '+str(folpass))
            print('lead passing: '+str(leadpass))
            print('llead passing: '+str(lleadpass))
            print('rlead passing: '+str(rleadpass))
        
        return (lfolpass and rfolpass and rleadpass and lleadpass and leadpass and folpass)
        
            
    
            

def downstream_wrapper(speed_fun = None, method = 'speed', congested = True, 
                       mergeside = 'l', merge_anchor_ind = None, target_lane = None, selflane = None, shift = 1,
                       minacc = -2):
    #downstream function -> method of lane, takes in (veh, timeind, dt)
    #and returns action (acceleration) for the vehicle 
    
    #options - speed_fun 
    if method == 'speed': #specify a function speedfun which takes in time and returns the speed
        def call_downstream(self, veh, timeind, dt):
            speed = speed_fun(timeind)
            return (speed - veh.speed)/dt
        return call_downstream
    
    #options - none
    elif method == 'free': #use free flow method of the vehicle 
        def free_downstream(self, veh, *args):
            return veh.free_cf(veh.cf_parameters, veh.speed)
        return free_downstream
    
    #options - congested controls whether the flow is inverted into free or congested branch
    elif method == 'flow': #specify a function which gives the flow, we invert the flow to obtain speed
        def call_downstream(self, veh, timeind, dt):
            flow = speed_fun(timeind)
            speed = veh.inv_flow(flow, output_type = 'v', congested = congested)
            return (speed - veh.speed)/dt
        return call_downstream
    
    #options - minacc
    #selflane must be provided
    elif method == 'free merge': #use free flow method of the vehicle 
        endanchor = anchor_vehicle(selflane, None)
        endanchor.pos = selflane.end
        def free_downstream(self, veh, timeind, dt):
            hd = veh.lane.get_headway(veh, endanchor)
            acc = veh.get_cf(hd, veh.speed, endanchor, veh.lane, timeind, dt, veh.in_relax)
            if acc < minacc: 
                return acc
            return veh.free_cf(veh.cf_parameters, veh.speed)
        return free_downstream
    
     #this is meant to give a longitudinal update in congested conditions 
     #when on a bottleneck (on ramp or lane ending) and you have no leader 
     #not because you are leaving the network but because the lane is ending and you need to move over
     
     #options - mergeside = 'l' - whether vehicles are supposed to merge left or right 
     #merge_anchor_ind - index for merge anchor in target_lane
     #selflane - Your own lane, can be provided if you want vehicle to stop if it can't merge. It will act as if there is a vehicle at the end of the road, 
     #and if its deceleration is more than minacc, it will use its CF model to update.
     #shift = 1 - will shift the follower's speed
     #minacc = -2 - if selflane is not None, then once we have to decelerate stronger than this, we will start slowing down 
     #speed_fun, if we can't use the endanchor, a follower/merge anchor, we default to using the free_cf, unless you provide speed_fun, in which case we will use speed_fun. 
    elif method == 'merge':
        #first try to get a vehicle and use its shifted speed. By default use the l/rfol (controlled by mergeside)
        #can also try using the merge anchor (if merge_anchor_ind is not None) or another anchor's lead (if anchor is not None)
        #it has to be a vehicle (not an anchor vehicle) as we want its speedmem
        #if we fail to find such a vehicle and speed_fun is not None, we will use that; 
        #otherwise we will use the vehicle's free_cf method
        
        if mergeside == 'l': 
            folside = 'lfol'
        elif mergeside == 'r':
            folside = 'rfol'
        if selflane != None:
            endanchor = anchor_vehicle(selflane, None)
            endanchor.pos = selflane.end
        else:
            endanchor = None
        def call_downstream(self, veh, timeind, dt): 
            if endanchor != None:
                hd = veh.lane.get_headway(veh, endanchor)
                acc = veh.get_cf(hd, veh.speed, endanchor, veh.lane, timeind, dt, veh.in_relax)
                if acc < minacc: 
                    return acc
            fol = getattr(veh, folside) #first check if we can use your current change side follower
            #try to find a vehicle to use for shifted speed
            if merge_anchor_ind != None: 
                if fol == None: 
                    fol = target_lane.merge_anchors[merge_anchor_ind][0]
                if fol.cf_parameters == None:
                    fol = fol.lead
            elif fol == None:
                pass
            elif fol.cf_parameters == None:
                fol = fol.lead

            if fol != None: #fol must either be none or a vehicle (can't be anchor)
                speed = shift_speed(fol.speedmem, shift, dt)
            elif speed_fun != None:
                speed = speed_fun(timeind)
            else: 
                return veh.free_cf(veh.cf_parameters, veh.speed)
            return (speed - veh.speed)/dt
        
        return call_downstream

#anchor vehicles are present at the start of any track. Any track can have only one anchor for any constituent lanes. 
#a track is defined as a continuous series of lanes such that a vehicle could start at the first lane, and continue through 
#all lanes on the track without performing any lane changing actions. So anchors are basically the start of the track.
#besides defining what track any lane is in, anchor vehicles are used in the inflow methods, as they keep track of the vehicle order
#and allow vehicles to be inserted with the correct vehicle order. 
#they also more generally maintain the vehicle order at the beginning of tracks, so that the only way a vehicle can have a None attribute for 
#lfol/rfol is if there is no lane for the vehicle. They can also be used to calculate headways or distances relative to other vehicles. 
#they are identified as anchors because they have cf_parameters = None

class anchor_vehicle: 
    #anchor vehicles have cf_parameters as None 
    def __init__(self, curlane, inittime, lfol = None, rfol = None, lead = None, rlead = None, llead = None):
        self.cf_parameters = None 
        self.lane = curlane
        self.road = curlane.road['name']
        
        self.lfol = lfol #I think anchor vehicles just need the lead/llead/rlead attributes and none of the fol attributes
        self.rfol = rfol
        self.lead = lead
        self.rlead = set() if rlead == None else rlead
        self.llead = set() if llead == None else llead
            
        
        self.pos = curlane.start
        self.speed = 0
        self.hd = None
        self.len = 0
        
        self.leadmem = [[lead,inittime]]
        
    def __repr__(self):
        return ('anchor for lane '+str(self.lane))
    
    def __str__(self):
        return self.__repr__()
        
def get_inflow_wrapper(speed_fun, inflow_type = 'flow'):
    #to use the inflow functions provided, you need the following methods in the lane 
    # - get_inflow = 'flow' - accepts either timeseries of flow ('flow'), or timeseries of speed ('speed'). 
    #If giving speeds, the vehicle to be added needs a get_eql method
    # - generate_parameters (accepts no arguments, returns cf/lc_parameters, and all keyword arguments, 
    # for a new vehicle)
    
    #returns get_inflow function, which accepts timeind and returns the flow at that time
    
    #give flow series - simple 
    if inflow_type == 'flow':
        def get_inflow(self, timeind):
            return speed_fun(timeind), None
    #give speed series, we convert to equilibrium flow using the parameters of the next vehicle to be added 
    #note that if all vehicles have same parameters/length, this is exactly equivalent to the 'flow' method 
    #(where the speeds correspond to the flows for the eql soln being used)
    elif inflow_type == 'speed':
        def get_inflow(self, timeind):
            spd = speed_fun(timeind)
            lead = self.anchor.lead
            if lead is not None:
                leadlen = lead.len
            else:
                leadlen = self.newveh.len
            s = self.newveh.get_eql(spd, find = 's')
            return spd / (s + leadlen), spd
        
    #in congested type it is similar to the 'speed' method but uses the speed from the anchor.lead instead of 
    #a speed which is specified a priori. This is basically supposed to add a vehicle with 0 acceleration 
    elif inflow_type == 'congested':
        def get_inflow(self, timeind):
            lead = self.anchor.lead
            if lead is not None:
                leadlen = lead.len
                spd = lead.speed
            else:
                leadlen = self.newveh.len
                spd = speed_fun(timeind)
            s = self.newveh.get_eql(spd, find = 's')
            return spd / (s + leadlen), spd

    return get_inflow
        
def timeseries_wrapper(timeseries, starttimeind = 0):
    def out(timeind):
        return timeseries[timeind-starttimeind]
    return out
    
def eql_inflow_congested(curlane, inflow, c = .8, check_gap = True):
    #suggested by treiber for congested conditions, requires to invert the inflow to obtain 
    #the steady state headway. the actual headway on the road must be at least c * the steady state headway 
    #for the vehicle to be added. 
    #if check_gap is False, we don't have to invert the flow, we will always just add at the equilibrium speed
    #the vehicle is added with a speed obtained from the equilibrium speed with the current headway 
    
    lead = curlane.anchor.lead
    hd = curlane.get_headway(curlane.anchor, lead)
    if check_gap == True:
        se = curlane.newveh.inv_flow(inflow, leadlen = lead.len, output_type = 's') #headway corresponding to current flow
    else:
        se = -math.inf
    if hd > c*se: #condition met
        spd = curlane.veh.get_eql(hd, input_type = 's')
        return curlane.start, spd, hd
    else:
        return None
    
def eql_inflow_free(curlane, inflow):
    #suggested by treiber for free conditions, requires to invert the inflow to obtain 
    #the velocity 
    lead = curlane.anchor.lead
    hd = curlane.get_headway(curlane.anchor, lead)
    spd = curlane.newveh.inv_flow(inflow, leadlen = lead.len, output_type = 'v', congested = False) #speed corresponding to current flow
    return curlane.start, spd, hd

def shifted_speed_inflow(curlane, dt, shift = 1, accel_bound = -2):
    #gives the first speed based on the shifted speed of the lead vehicle (similar to newell model)
    #shift = 1 - shift in time, measured in real time 
    #accel_bound = -2 - if not None, the acceleration of the vehicle 
    #must be greater than the accel_bound. Otherwise, no such bound is enforced
    lead = curlane.anchor.lead
    hd = curlane.get_headway(curlane.anchor, lead)
    spd = shift_speed(lead.speedmem, shift, dt)
        
    if accel_bound is not None: 
        newveh = curlane.newveh
        acc = newveh.get_cf(hd, spd, lead, curlane, None, dt, False)
        if acc > accel_bound: 
            return 0, spd, hd
        else: 
            return None
    
    return curlane.start, spd, hd

def shift_speed(speedseries, shift, dt):
    #speedseries is timeseries with a constant discretization of dt
    #we want the measurement from shift time ago
    #outputs speed
    ind = int(shift // dt) 
    if ind+1 > len(speedseries):
        return speedseries[0]
    remainder = shift - ind*dt
    spd = (speedseries[-ind-1]*(dt - remainder) + speedseries[-ind]*remainder)/dt #weighted average
    return spd

def speed_inflow(curlane, speed_fun, timeind, dt, accel_bound = -2):
    #gives a speed based on speed_fun
    lead = curlane.anchor.lead
    hd = curlane.get_headway(curlane.anchor, lead)
    spd = speed_fun(timeind)
        
    if accel_bound is not None: 
        newveh = curlane.newveh
        acc = newveh.get_cf(hd, spd, lead, curlane, None, dt, False)
        if acc > accel_bound: 
            return 0, spd, hd
        else: 
            return None
    
    return curlane.start, spd, hd 
    


def increment_inflow_wrapper(speed_fun = None, method = 'ceql', accel_bound = -1, check_gap = True, shift = 1, c = .8):
    #method = 'eql' vehicles have 0 acceleration when being added. The speed is defined by the vehicles
    #equilibrium function. Thus to use this method, the vehicle to be added must have a get_eql method 
    #(and use a model which predicts acceleration)
    #if check_gap = True, the headway must be at least as big as the equilibrium headway corresponding to 
    #the current inflow. 
    #method = 'shifted' - uses shifted speed 
    def increment_inflow(self, vehicles, vehid, timeind, dt):
        inflow, spd = self.get_inflow(timeind)
        self.inflow_buffer += inflow * dt
        
        if self.inflow_buffer >= 1: 
            if self.anchor.lead is None: 
                if spd is None: 
                    # spd = speed_fun(timeind)
                    spd = self.newveh.inv_flow(inflow, congested = False)
                out = (self.start, spd, None)
            elif method == 'ceql': 
                out = eql_inflow_congested(self, inflow, c = c, check_gap = check_gap)
            elif method == 'feql':
                out = eql_inflow_free(self, inflow)
            elif method == 'shifted':
                out = shifted_speed_inflow(self, dt, shift = shift, accel_bound = accel_bound)
            elif method == 'speed':
                out = speed_inflow(self, speed_fun, timeind, dt, accel_bound = accel_bound)
            
            if out == None:  
                return vehid
            #add vehicle with the given initial conditions
            pos, speed, hd = out[:]
            newveh = self.newveh
            anchor = self.anchor
            lead = anchor.lead
            newveh.lead = lead
            
            #initialize state
            newveh.initialize(pos, speed, hd, timeind+1)
            
            #update leader/follower relationships
            #leader relationships
            if lead != None:
                lead.fol = newveh
            for rlead in anchor.rlead: 
                rlead.lfol = newveh
            newveh.rlead = anchor.rlead
            anchor.rlead = set()
            for llead in anchor.llead:
                llead.rfol = newveh
            newveh.llead = anchor.llead
            anchor.llead = set()
            
            #update anchor and follower relationships
            anchor.lead = newveh
            anchor.leadmem.append((newveh, timeind+1))
            newveh.fol = anchor
            
            llane = self.get_connect_left(pos)
            if llane != None:
                leftanchor = llane.anchor
                newveh.lfol = leftanchor
                leftanchor.rlead.add(newveh)
            else:
                newveh.lfol = None
                    
            rlane = self.get_connect_right(pos)
            if rlane != None:
                rightanchor = rlane.anchor
                newveh.rfol = rightanchor
                rightanchor.llead.add(newveh)
            else:
                newveh.rfol = None
            
            #update simulation
            self.inflow_buffer += -1
            vehicles.add(newveh)
            vehid = vehid + 1

        
            #create next vehicle
            cf_parameters, lc_parameters, kwargs = self.new_vehicle()
            self.newveh = vehicle(vehid, self, cf_parameters, lc_parameters, **kwargs)
            
            
        return vehid
        
    return increment_inflow
        
    
class lane: 
    def __init__(self, start, end, road, laneind, connect_left = None, connect_right =None,
                 downstream = None, increment_inflow = None, get_inflow = None, new_vehicle = None):
        
        self.laneind = laneind
        self.road = road
        self.roadname = road['name']
        #starting position/end (float)
        self.start = start
        self.end = end
        #connect_left/right has format of list of (pos (float), lane (object)) tuples where lane is the connection starting at pos 
        self.connect_left = connect_left if connect_left != None else [(0, None)]
        self.connect_right = connect_right if connect_right != None else [(0, None)]
        self.connect_to = None
        
        if downstream != None:
            self.call_downstream = downstream_wrapper(**downstream).__get__(self, lane)
            
        if get_inflow != None:
            self.get_inflow = get_inflow_wrapper(**get_inflow).__get__(self, lane)
            
        if new_vehicle != None:
            self.new_vehicle = new_vehicle
            
        if increment_inflow != None:
            self.inflow_buffer = 0
            # cf_parameters, lc_parameters, kwargs = self.new_vehicle()
            # self.newveh = vehicle(vehid, self, cf_parameters, lc_parameters, **kwargs)
            self.increment_inflow = increment_inflow_wrapper(**increment_inflow).__get__(self, lane)
        
    
    def get_headway(self, veh, lead): 
        #distance from front of vehicle to back of lead
        #assumes veh.road = self.road
        hd = lead.pos - veh.pos - lead.len
        if self.roadname != lead.road: 
            hd += self.roadlen[lead.road] #currently roads are dicts, and we hash them using their 'name' key
        return hd 
    
    def get_dist(self, veh, lead): 
        #distance from front of vehicle to front of lead
        #assumes veh.lane.road = self.road
        dist = lead.pos-veh.pos
        if self.roadname != lead.road: 
            dist += self.roadlen[lead.road]
        return dist
    
    def leadfol_find(self, veh, guess, side):
        #given guess vehicle which is 'close' to veh
        #returns the leader, follower in that order in the same track of lanes as guess 
        #side is the side of veh we are looking at - e.g. side = 'r' means we are looking to the right of veh
        
        #used to initialize the new lc side follower/leader when new lanes become available
        #because this is only used when a new lane becomes available, there will always be a follower returned
        #it is possible that the leader is None, or that there is a leader but it can't have veh as a follower. 
        
        if side == 'r':
            checkfol = 'lfol'
        else:
            checkfol = 'rfol'
        get_dist = self.get_dist
        hd = get_dist(veh,guess)
        if hd < 0: 
            nextguess = guess.lead 
            if nextguess == None:  #None -> reached end of network
                return nextguess, guess
            nexthd = get_dist(veh, nextguess)
            while nexthd < 0: 
                # counter += 1
                guess = nextguess 
                nextguess = guess.lead
                if nextguess == None:
                    return nextguess, guess
                nexthd = get_dist(veh, nextguess)
                
                
            if getattr(nextguess,checkfol) == None: 
                nextguess = None
            return nextguess, guess
        else: 
            nextguess = guess.fol
            if nextguess.cf_parameters == None:
                return guess, nextguess
            nexthd = get_dist(veh, nextguess)
            while nexthd > 0:
                # counter +=1
                guess = nextguess
                nextguess = guess.fol
                if nextguess.cf_parameters == None: #reached anchor -> beginning of network
                    return guess, nextguess
                nexthd = get_dist(veh, nextguess)
                
                
            if getattr(guess,checkfol) == None: 
                guess = None
            return guess, nextguess
    
        
    def get_connect_left(self, pos):
        #given position, returns the connection to left 
        #output is either lane object or None
        return connect_helper(self.connect_left, pos)

    def get_connect_right(self, pos):
        return connect_helper(self.connect_right,pos)
    
    
    def __hash__(self):
        return hash((self.roadname, self.laneind))
    
    def __eq__(self, other):
        # return self.roadname== other.roadname and self.laneind == other.laneind
        return self is other
    
    def __ne__(self, other):
        return not(self is other)
    
    def __repr__(self):
        return (self.roadname+' ('+str(self.laneind)+')')
    
    def __str__(self):
        return self.__repr__()
    
    
def connect_helper(connect, pos):
    out = connect[-1][1] #default to last lane for edge case or case when there is only one possible connection 
    for i in range(len(connect)-1):
        if pos < connect[i+1][0]:
            out = connect[i][1]
            break
    return out 


#def CF_wrapper(cfmodel): 
#    #acc_bounds controls [lower, upper] bounds on acceleration 
#    #assumes a second order model which has inputs of (p, state), where state
#    #is a list of all values needed, p is a list of parameters, and
#    #output is a float giving the acceleration 
#    def call_cf_helper(self, hd, spd, lead, curlane, timeind, dt, userelax): 
#        if lead is None: 
#            acc = curlane.call_downstream(self, timeind, dt)
#            
#        else:
#            if userelax:
#                currelax = self.relax[timeind - self.relax_start]
#                hd += currelax #can add check to see if relaxed headway is too small
#                acc = cfmodel(self.cf_parameters, [hd, spd, lead.speed])
#                hd += -currelax
#            else: 
#                acc = cfmodel(self.cf_parameters, [hd, spd, lead.speed])
#            
#        return acc
#    
#    return call_cf_helper
    
#def eql_wrapper(eqlfun, eql_type = 'v', tol = .1, spdbounds = (0, 1e4), hdbounds = (0, 1e4), **kwargs):
#    #eqlfun -> fun to wrap, needs call signature like (parameters, input, *args)
#    #eql_type = 's' - if 'v', eqlfun takes in velocity and outputs headway. if 's', it takes in headway and outputs velocity
#    #if eql_type = 'both', then eqlfun takes in an additional argument (parameters, input, input_type) and will return the other quantity
#    #bounds = (1e-10, 120) - bounds used when eql_type != find. Should define an interval that the soln is in 
#    #tol = .5 - if need to numerically invert the function, this is the tolerance used 
#    
#    if eql_type != 'both':
#        if eql_type == 'v': 
#            bracket = spdbounds
#        else:
#            bracket = hdbounds
#        def get_eql(self, x, input_type = 'v'):
#            if input_type == 'v': 
#                if x < spdbounds[0]: 
#                    x = spdbounds[0]
#                elif x > spdbounds[1]:
#                    x = spdbounds[1]
#            elif input_type == 's': 
#                if x < hdbounds[0]: 
#                    x = hdbounds[0]
#                elif x > hdbounds[1]:
#                    x = hdbounds[1]
#            
#            if input_type == eql_type: 
#                return eqlfun(self.cf_parameters, x)
#            elif input_type != eql_type: 
#                def inveql(y):
#                    return x - eqlfun(self.cf_parameters, y)
#                ans = sc.root_scalar(inveql, bracket = bracket, xtol = tol, method = 'brentq')
#                if ans['converged']: 
#                    return ans['root']
#                else: 
#                    raise RuntimeError('could not invert provided equilibrium function')
#    else:
#        def get_eql(self, x, input_type = 'v'):
#            return eqlfun(self.cf_parameters, x, input_type)
#    
#    return get_eql


#def FD_wrapper(eqlfun):
#    #returns flow based on provided equilibrium function 
#    def get_flow(self, x, leadlen = None, input_type = 'v'):
#        if leadlen == None: 
#            lead = self.lead 
#            if lead != None: 
#                leadlen = lead.len
#            else:
#                leadlen = self.len
#        if input_type == 'v':
#            s = self.get_eql(x, input_type = input_type)
#            return x / (s + leadlen)
#        elif input_type == 's': 
#            v = self.get_eql(x, input_type = input_type)
#            return v / (s + leadlen)
#    return get_flow
                
#def invFD_wrapper(eqlfun, eql_type = 'v', spdbounds = (0, 1e4), hdbounds = (0, 1e4), tol = .1, ftol = .01, invflowfun = None):
#    #same call signature as eql_wrapper, tol is for headway/speed tolerance, ftol is for flow tolerance 
#    if eql_type != 'both':
#        if eql_type == 'v': 
#            bracket = spdbounds
#        else:
#            bracket = hdbounds
#        def inv_flow(self, x, leadlen = None, output_type = 'v', congested = True):
#            if leadlen == None: 
#                lead = self.lead 
#            if lead != None: 
#                leadlen = lead.len
#            else:
#                leadlen = self.len
#                
#            def maxfun(y):
#                return -self.get_flow(y, leadlen = leadlen, input_type = eql_type)
#            
#            res = sc.minimize_scalar(maxfun, bracket = bracket, tol = ftol)
#            if res['success']:
#                if congested: 
#                    invbounds = (bracket[0], res['x'])
#                else:
#                    invbounds = (res['x'], bracket[1])
#                if -res['fun'] < x: 
#                    print('warning - inputted flow is too large to be achieved')
#                    if output_type == eql_type: 
#                        return res['x']
#                    else: 
#                        return self.get_eql(res['x'], input_type = eql_type)
#            else:
#                raise RuntimeError('could not find maximum flow')
#                
#                
#            if eql_type == 'v':
#                def invfun(y): 
#                    return x - y/(eqlfun(self.cf_parameters, y) + leadlen)
#            elif eql_type == 's':
#                def invfun(y):
#                    return x - eqlfun(self.cf_parameters, y)/(y+leadlen)
#            
#            ans = sc.root_scalar(invfun, bracket = invbounds, xtol = tol, method = 'brentq')
#            
#            if ans['converged']:
#                if output_type == eql_type: 
#                    return ans['root']
#                elif output_type == 's':
#                    return ans['root']/x - leadlen
#                elif output_type == 'v':
#                    return (ans['root']+leadlen)*x
#            else: 
#                raise RuntimeError('could not invert provided equilibrium function')
#    else: 
#        def inv_flow(self, x, leadlen = None, output_type = 'v', congested = True):
#            return invflowfun(x, leadlen, output_type, congested)
#    
#    return inv_flow
    

#def LC_wrapper(lcmodel, get_fol = True, **kwargs): #userelax_cur = True, userelax_new = False
#    #lcmodel - model to wrap. Assume it takes as input the vehicle, 
#        #new left follower headway, new left headway, new right follower headway, new right headway, 
#        #new follower headway (if get_fol is True), timeind, dt, *args, **kwargs
#    #get_fol - lc model uses the current follower if True 
#    #kwargs - keyword arguments which are passed to lcmodel 
#    
#    def call_lc(self, lc_actions, timeind, dt):
#        #first determine what situation we are in and which sides we need to check
#        l, r  = self.l, self.r
#        if l == None: 
#            if r == None: 
#                return
#            elif r == 'discretionary':
#                lside, rside = False, True
#                chk_cond = False if self.lcside != None else True
#            else: 
#                lside, rside = False, True
#                chk_cond = False
#        elif l == 'discretionary': 
#            if r == None:
#                lside,rside = True, False
#                chk_cond = False if self.lcside != None else True
#            elif r == 'discretionary': 
#                if self.lcside != None: 
#                    chk_cond = False
#                    if self.lcside == 'l': 
#                        lside, rside = True, False
#                    else: 
#                        lside, rside = False, True
#                else: 
#                    chk_cond = True
#                    lside, rside = True, True
#            else: 
#                lside, rside = False, True
#                chk_cond = False
#        else: 
#            if r == None: 
#                lside, rside = True, False
#                chk_cond = False
#            elif r == 'discretionary': 
#                lside, rside = True, False
#                chk_cond = False
#                
#        if chk_cond: #decide if we want to evaluate lc model or not - applies to discretionary state when there is no cooperation or tactical positioning
#            chk_lc = self.lc_parameters[-1]
#            if chk_lc >= 1:
#                pass
#            elif np.random.rand() > chk_lc: 
#                return 
#            
#        #next we compute quantities to send to LC model for the required sides 
#        if lside: 
#            newlfolhd, newlhd = call_lc_helper(self.lfol, self, self.llane) #better to just store left/right lanes
#        else:
#            newlfolhd = newlhd = None
#        
#        if rside: 
#            newrfolhd, newrhd = call_lc_helper(self.rfol, self, self.rlane)
#        else:
#            newrfolhd = newrhd = None
#            
#        #if get_fol option is given to wrapper, it means model requires the follower's quantities as well 
#        if get_fol: 
#            fol, lead = self.fol, self.lead
#            if fol.cf_parameters == None: 
#                newfolhd = None 
#            elif lead == None: 
#                newfolhd = None
#            else: 
#                newfolhd = fol.lane.get_headway(fol, lead)
#                
#            #do model call now for get_fol = True
#            lcmodel(self, lc_actions, lside, rside, newlfolhd, newlhd, newrfolhd, newrhd, newfolhd, timeind, dt, 
#                    **kwargs)
#        else: #model call signature when get_fol = False
#            lcmodel(self, lc_actions, lside, rside, newlfolhd, newlhd, newrfolhd, newrhd, timeind, dt, 
#                    **kwargs)
#            
#    return call_lc