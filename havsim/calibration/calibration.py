"""Refactors the functionality of the calibration.opt module.

The simulation module does an entire micro simulation. The calibration module supports simulations where
the lane changing times and vehicle orders fixed apriori to match trajecotry data. This allows direct
comparison with the trajectory data, removing the need to only calibrate to the macroscopic, aggreagted data.
Either single vehicles, or strings (platoons) of vehicles can be simulated.
"""

import numpy as np
import havsim.simulation as hs
from havsim.simulation.road_networks import get_headway
from havsim import helper
import havsim.helper as helper
from havsim import simulation
import math

# TODO finish implementing calibration for platoons of vehicle
    #handle assigning parameters for multiple vehicles
    #removing vehicles and downstream boundary
# TODO implement calibration for latitudinal models also

class CalibrationVehicle(hs.Vehicle):
    """Base CalibrationVehicle class for a second order ODE model.

    CalibrationVehicles implement rules to update their own positions only. There is currently no
    functionality for using their lane changing models. The use case for CalibrationVehicle is to simulate
    vehicle(s) with predetermined lane changes/lead vehicles, to compare directly to microscopic data.
    Compared to the Vehicle class, CalibrationVehicles have no lane/road, always have a leader (which can be
    either another CalibrationVehicle or LeadVehicle), and have no routes events/lane events.

    Attributes:
        vehid: unique vehicle ID for hashing (float)
        lane: None
        road: None
        cf_parameters: list of float parameters for cf model
        relax_parameters: float parameter(s) for relaxation model, or None
        relax: if there is currently relaxation, a list of floats or list of tuples giving the relaxation
            values.
        in_relax: bool, True if there is currently relaxation
        relax_start: time index corresponding to relax[0]. (int)
        relax_end: The last time index when relaxation is active. (int)
        minacc: minimum allowed acceleration (float)
        maxacc: maxmimum allowed acceleration(float)
        maxspeed: maximum allowed speed (float)
        hdbounds: tuple of minimum and maximum possible headway.
        eql_type: If 'v', the vehicle's eqlfun accepts a speed and returns a headway. Otherwise it
            accepts a headway and returns a speed.
        lead: leading vehicle, can be either a (subclassed) Vehicle or LeadVehicle
        inittime: time index of the first simulated time
        initpos: position at inittime
        initspd: speed at inittime
        leadveh: If there is a LeadVehicle, leadveh is a reference to it. Otherwise None.
        in_leadveh: True if the LeadVehicle is the current leader.
        leadmem: list of tuples, where each tuple is (lead vehicle, time) giving the time the ego vehicle
            first begins to follow the lead vehicle.
        posmem: list of floats giving the position, where the 0 index corresponds to the position at starttime
        speedmem: list of floats giving the speed, where the 0 index corresponds to the speed at starttime
        relaxmem: list of tuples where each tuple is (first time, last time, relaxation) where relaxation
            gives the relaxation values for between first time and last time
        pos: position (float)
        speed: speed (float)
        hd: headway (float)
        len: vehicle length (float)
        acc: acceleration (float)
        y: target given to loss function (e.g. the position time series from data)
    """
    def __init__(self, vehid, y, initpos, initspd, inittime, leadstatemem, leadinittime, length=3,
                 accbounds=None, maxspeed=1e4, hdbounds=None, eql_type='v'):
        """Inits CalibrationVehicle. Cannot be used for simulation until initialize is called.

        Args:
            vehid: unique vehicle ID for hashing, float
            y: target for loss function, e.g. a 1d numpy array of np.float64
            initpos: initial position, float
            initspd: initial speed, float
            inittime: first time of simulation, float
            leadstatemem: list of tuples of floats. Gives the LeadVehicle state at the corresponding
                time index.
            leadinittime: float of time index that 0 index of leadstatemem corresponds to
            length: float vehicle length
            accbounds: list of minimum/maximum acceleration. If None, defaults to [-7, 3]
            maxspeed: float of maximum speed.
            hdbounds: list of minimum/maximum headway. Defaults to [0, 10000].
            eql_type: 'v' If eqlfun takes in speed and outputs headway, 's' if vice versa. Defaults to 'v'.
        """
        self.vehid = vehid
        self.len = length
        self.y = y
        self.initpos = initpos
        self.initspd = initspd
        self.inittime = inittime

        self.road = None
        self.lane = None

        if accbounds is None:
            self.minacc, self.maxacc = -7, 3
        else:
            self.minacc, self.maxacc = accbounds[0], accbounds[1]
        self.maxspeed = maxspeed
        self.hdbounds = (0, 1e4) if hdbounds is None else hdbounds
        self.eql_type = eql_type

        if leadstatemem is not None:
            self.leadveh = LeadVehicle(leadstatemem, leadinittime)
        self.in_leadveh = False

    def set_relax(self, relaxamounts, timeind, dt):
        """Applies relaxation given the relaxation amounts."""
        rp = self.relax_parameters
        if rp is None:
            return
        relaxamount_s, relaxamount_v = relaxamounts
        hs.relaxation.relax_helper_vhd(rp, relaxamount_s, relaxamount_v, self, timeind, dt)

    def update(self, timeind, dt):
        """Update for longitudinal state. Updates LeadVehicle if applicable."""
        super().update(timeind, dt)

        if self.in_leadveh:  # only difference is we update the LeadVehicle if applicable
            self.leadveh.update(timeind+1)

    def loss(self):
        """Calculates loss."""
        print(len(self.posmem))
        print(len(self.y[:len(self.posmem)]))
        if len(self.posmem) > len(self.y[:len(self.posmem)]):
            return  sum(np.square(np.array(self.posmem[:len(self.y[:len(self.posmem)])]) - self.y[:len(self.posmem)]))/len(self.posmem[:len(self.y[:len(self.posmem)])])
        return sum(np.square(np.array(self.posmem) - self.y[:len(self.posmem)]))/len(self.posmem)

    def initialize(self, parameters):
        """Resets memory, applies initial conditions, and sets the parameters for the next simulation."""
        # initial conditions
        self.lead = None
        self.pos = self.initpos
        self.speed = self.initspd
        # reset relax
        self.in_relax = False
        self.relax = None
        self.relax_start = None
        # memory
        self.leadmem = []
        self.posmem = [self.pos]
        self.speedmem = [self.speed]
        self.relaxmem = []
        # parameters
        self.cf_parameters = parameters[:-1]
        self.maxspeed = parameters[0]-.1
        self.relax_parameters = parameters[-1]


class LeadVehicle:
    """Used for simulating a vehicle which follows a predetermined trajectory - it has no models.

    When a CalibrationVehicles natural leaders are not simulated, the CalibrationVehicle has a LeadVehicle
    which holds the trajectory of the leader. LeadVehicles act as if they were a Vehicle, but have no models
    or parameters, and instead have their state updated from predefined memory.
    """
    def __init__(self, leadstatemem, inittime):
        """
        leadstatemem - list of tuples, each tuple is a pair of (position, speed)
        inittime - leadstatemem[0] corresponds to time inittime
        """
        self.leadstatemem = leadstatemem
        self.inittime = inittime
        self.road = None

    def update(self, timeind, *args):
        """Update position/speed."""
        #get rid of this later, why is the statemem index not correct?
        if timeind - self.inittime >= len(self.leadstatemem):
            self.pos , self.speed = self.pos , self.speed
        else:
            self.pos, self.speed = self.leadstatemem[timeind - self.inittime]

    def set_len(self, length):
        """Set len so headway can be computed correctly."""
        self.len = length


class Calibration:
    """Does a simulation of a single CalibrationVehicle, and returns the loss.

    Attributes:
        all_vehicles: list of all vehicles in the simulation
        all_add_events: list of all add events in the simulation
        all_lc_events: list of all lead change events in the simulation
        starttime: first time index
        endtime: last time index
        dt: timestep
        addtime: the next time index when an add event occurs
        lctime: the next time index when a lead change event occurs
        timeind: current time index of simulation
        vehicles: set of all vehicles currently in simulation
        add_events: sorted list of remaining add events
        lc_events: sorted list of remaining lead change events
        lc_event: function that can apply a lc_events
    """
    def __init__(self, vehicles, add_events, lc_events, dt, meas, lc_event_fun = None, endtime = None):
        """Inits Calibration.

        Args:
            vehicles: list of all Vehicle objects to be simulated
            add_events: list of add events, sorted in time
            lc_events: list of lead change (lc) events, sorted in time
            dt: timestep, float
            lc_event_fun: can give a custom function for handling lc_events, otherwise we use the default
            endtime: last time index which is simulated. The starttime is inferred from add_events.
        """

        self.all_vehicles = vehicles
        self.all_add_events = add_events
        self.all_lc_events = lc_events
        if lc_event_fun is None:
            self.lc_event = lc_event
        else:
            self.lc_event = lc_event_fun


        self.starttime = add_events[-1][0]
        self.endtime = endtime
        self.dt = dt
        self.meas = meas



    def step(self):
        """Logic for a single simulation step. Main logics are in update_calibration."""
        for veh in self.vehicles:
            veh.set_cf(self.timeind, self.dt)


        self.addtime, self.lctime = update_calibration(self.vehicles, self.add_events, self.lc_events,
                                                       self.addtime, self.lctime, self.timeind, self.dt,
                                                       self.lc_event, self.meas)

        self.timeind += 1

    def simulate(self, parameters, meas):
        """Does a full simulation and returns the loss.

        Args:
            parameters: list of parameters for the vehicles.

        Returns:
            loss (float).
        """
        param_size = int(len(parameters) / len(self.all_vehicles))
        # reset all vehicles, and assign their parameters
        for veh_index in range(len(self.all_vehicles)):
            veh = self.all_vehicles[veh_index]
            param_start_index = int(veh_index * param_size)
            veh.initialize(parameters[param_start_index:param_start_index+param_size])  # TODO need some method to assign parameters to vehicles

        # initialize events, add the first vehicle(s), initialize vehicles headways/LeadVehicles
        self.vehicles = set()
        self.add_events = self.all_add_events.copy()
        self.lc_events = self.all_lc_events.copy()
        self.addtime = self.add_events[-1][0]
        self.lctime = self.lc_events[-1][0] if len(self.lc_events)>0 else math.inf
        self.timeind = self.starttime
        self.addtime = update_add_event(self.vehicles, self.add_events, self.addtime, self.timeind-1, self.dt,
                                        self.lc_event, meas)
        for veh in self.vehicles:
            if veh.in_leadveh:
                veh.leadveh.update(self.timeind)
            veh.hd = get_headway(veh, veh.lead)


        # do simulation by calling step repeatedly
        for i in range(self.endtime - self.starttime):
            self.step()

        # calculate and return loss
        loss = 0
        for veh in self.all_vehicles:
            loss += veh.loss()

        return loss


def update_calibration(vehicles, add_events, lc_events, addtime, lctime, timeind, dt, lc_event, meas):
    """Main logic for a single step of the Calibration simulation.

    At the beginning of the timestep, vehicles/states/events are assumed to be fully updated. Then, in order,
        -call each vehicle's cf model (done in Calibration.step).
        -check for lead change events in the next timestep, and apply them if applicable.
        -update all vehicle's states, except headway, for the next timestep.
        -check for add events, and add the vehicles if applicable.
        -update all vehicle's headways.
    Then, when the timestep is incremented, all vehicles/states/events are fully updated, and the iteration
    can continue.

    Args:
        vehicles: set of vehicles currently in simulation
        add_events: sorted list of current add events
        lc_events: sorted list of current lead change events
        addtime: next time an add event occurs
        lctime: next time a lead change event occurs
        timeind: time index
        dt: timestep
        lc_event: function to apply a single entry in lc_events
    """
    lctime = update_lc_event(lc_events, lctime, timeind, dt, lc_event, meas)

    for veh in vehicles:
        veh.update(timeind, dt)

    #removing vecs that have an end position above 1475
    remove_list = remove_vehicles(vehicles, 1420)
    for remove_vec in remove_list:
        vehicles.remove(remove_vec)


    addtime = update_add_event(vehicles, add_events, addtime, timeind, dt, lc_event, meas)

    for veh in vehicles:
        x = veh.lead
        if x != None:
            veh.hd = get_headway(veh, veh.lead)

    return addtime, lctime

def remove_vehicles(vehicles, endpos):
    """See if vehicle needs to be removed from simulation"""
    remove_list = []
    for veh in vehicles:
        if veh.pos > endpos:
            remove_list.append(veh)
            if veh.fol is not None:
                veh.fol.lead = None
    # pop from vehicles
    return remove_list

def update_lc_event(lc_events, lctime, timeind, dt, lc_event, meas):
    """Check if we need to apply the next lc event, apply it and update lctime if so.

    Args:
        lc_events: sorted list of current lead change events
        lctime: next time a lead change event occurs
        timeind: time index
        dt: timestep
    """
    if lctime == timeind+1:
        lc_event(lc_events.pop(), timeind, dt, meas)
        lctime = lc_events[-1][0] if len(lc_events)>0 else math.inf
        if lctime == timeind+1:
            while lctime == timeind+1:
                lc_event(lc_events.pop(), timeind, dt, meas)
                lctime = lc_events[-1][0] if len(lc_events)>0 else math.inf
    return lctime


def update_add_event(vehicles, add_events, addtime, timeind, dt, lc_event, meas):
    """Check if we need to apply the next add event, apply it and update addtime if so.

    Args:
        add_events: sorted list of current add events
        addtime: next time an add event occurs
        timeind: time index
        dt: timestep
    """
    if addtime == timeind+1:
        add_event(add_events.pop(), vehicles, timeind, dt, lc_event, meas)
        addtime = add_events[-1][0] if len(add_events)>0 else math.inf
        if addtime == timeind+1:
            while addtime == timeind+1:
                add_event(add_events.pop(), vehicles, timeind, dt, lc_event, meas)
                addtime = add_events[-1][0] if len(add_events)>0 else math.inf
    return addtime


def make_calibration(vehicles, meas, platooninfo, dt, vehicle_class = None, calibration_class = None,
                     event_maker = None, lc_event_fun = None):
    """Sets up a Calibration object.

    Extracts the relevant quantities (e.g. LeadVehicle, initial conditions, loss) from the data
    and creates the add/lc event.

    Args:
        vehicles: list/set of vehicles to add to the Calibration
        meas: from havsim.calibration.algs.makeplatoonlist
        platooninfo: from havsim.calibration.algs.makeplatoonlist
        dt: timestep
        vehicle_class: subclassed Vehicle to use - if None defaults to CalibrationVehicle
        calibration_class: subclassed Calibration to use - if None defaults to Calibration
        event_maker: specify a function to create custom (lc) events
        lc_event_fun: specify function which handles custom lc events
    """
    # TODO - to make this consistent with deep_learning, y should be over the times t_n+1 - min(T_nm1+1, T_n)
    # and also this should use the helper function get_lead_data
    if vehicle_class is None:
        vehicle_class = CalibrationVehicle
    if calibration_class is None:
        calibration_class = Calibration
    if event_maker is None:
        event_maker = make_lc_event
    if lc_event_fun is None:
        lc_event_fun = lc_event

    # initialize
    vehicle_list = []
    addevent_list = []
    lcevent_list = []
    id2obj = {}
    if type(vehicles) == list:
        vehicles = set(vehicles)
    for veh in vehicles:
        # make lead memory - a list of position/speed tuples for any leaders which are not simulated
        leads = set(platooninfo[veh][4])
        #they are being simulated so need to get rid of this?
        needleads = leads.difference(vehicles)
        if len(needleads) > 0:
            leadstatemem = []
            leadinittime = None  # initial time lead is used
            leadinfo = helper.makeleadinfo([veh],platooninfo,meas)
            for j in leadinfo[0]:
                curlead, start, end = j
                if curlead in needleads:
                    if not leadinittime:
                        leadinittime = start
                    leadt_n = platooninfo[curlead][0]
                    poslist = list(meas[curlead][start-leadt_n:end-leadt_n+1,2])
                    spdlist = list(meas[curlead][start-leadt_n:end-leadt_n+1,3])
                    leadstatemem.extend(zip(poslist,spdlist))
                elif leadinittime is not None:
                    temp = [0]*(end-start+1)
                    leadstatemem.extend(temp)
        else:
            leadstatemem, leadinittime = None, None

        # get initial values, y for veh
        t_nstar, inittime, endtime = platooninfo[veh][0:3]
        initlead, initpos, initspd, length = meas[veh][inittime-t_nstar,[4,2,3,6]]
        y = meas[veh][inittime-t_nstar:endtime+1-t_nstar,2]
        # create vehicle object
        newveh = vehicle_class(veh, y, initpos, initspd, inittime, leadstatemem, leadinittime,
                                    length=length)
        vehicle_list.append(newveh)
        id2obj[veh] = newveh

    # create events
    addevent_list, lcevent_list = event_maker(vehicles, id2obj, meas, platooninfo, dt,
                                                addevent_list, lcevent_list)

    addevent_list.sort(key = lambda x: x[0], reverse = True)  # sort events in time
    lcevent_list.sort(key = lambda x: x[0], reverse = True)
    # make calibration object
    return calibration_class(vehicle_list, addevent_list, lcevent_list, dt, meas,endtime=endtime,
                             lc_event_fun = lc_event_fun)


def make_lc_event(vehicles, id2obj, meas, platooninfo, dt, addevent_list, lcevent_list):
    for veh in vehicles:
        curveh = id2obj[veh]
        leadinfo = helper.makeleadinfo([veh],platooninfo,meas)[0]
        for count, j in enumerate(leadinfo):
            curlead, start, end = j
            leadt_nstar = platooninfo[curlead][0]
            # even though the change occurs at time start, we want to calculate the relaxation using
            # the differences in headway at time start - 1. This leads to 4 combinations, first, whether
            # the new leader is simulated or not, and second, whether the new lead is available at start-1
            if curlead in vehicles:  # curlead is simulated (in the same calibration object)
                if start-1 < leadt_nstar:  # handle edge case where t_nstar = start
                    leadstate = (meas[curlead][0,2]-meas[curlead][0,3]*dt,
                                 meas[curlead][0,3])
                else:
                    leadstate = (None,)
                curlead, curlen = id2obj[curlead], None
            else:
                curlen = meas[curlead][0,6]  # curlead is already simulated, stored in curveh.leadstatemem
                if start-1 < leadt_nstar:  # handle edge case where t_nstar = start
                    leadstate = (meas[curlead][0,2]-meas[curlead][0,3]*dt,
                                 meas[curlead][0,3])
                else:
                    leadstate = (meas[curlead][start-1-leadt_nstar,2], meas[curlead][start-1-leadt_nstar,3])

            if count == 0:  # add event adds vehicle to simulation
                t_nstar, t_n = platooninfo[veh][0:2]
                if t_n > t_nstar:  # apply merger rule
                    userelax = True
                else:
                    userelax = False
                # make the add event
                curevent = (start, 'lc', curveh, curlead, curlen, userelax, leadstate)
                curevent = (start, 'add', curveh, curevent)
                addevent_list.append(curevent)
            else:  # lc event changes leader, applies relax
                curevent = (start, 'lc', curveh, curlead, curlen, True, leadstate)
                lcevent_list.append(curevent)

    return addevent_list, lcevent_list


def add_event(event, vehicles, timeind, dt, lc_event, meas):
    """Adds a vehicle to the simulation and applies the first lead change event.

    Add events occur when a vehicle is added to the Calibration.
    Add events are a tuple of
        start (float) - time index of the event
        'add' (str) - identifies event as an add event
        curveh - CalibrationVehicle object to add to simulation
        lc_event - a lead change event which sets the first leader for curveh.

    Args:
        event: the add event
        vehicles: set of current vehicles in simulation which is modified in place
        timeind: time index
        dt: timestep
    """
    unused, unused, curveh, lcevent = event
    vehicles.add(curveh)
    lc_event(lcevent, timeind, dt, meas)

# TODO need to add new lane to event
def lc_event(event, timeind, dt, meas):
    """Applies lead change event, updating a CalibrationVehicle's leader.

    Lead change events occur when a CalibrationVehicle's leader changes. In a Calibration, it is
    assumed that vehicles have fixed lc times and fixed vehicle orders.
    Lead change events are a tuple of
        start (float) - time index of the event
        'lc' (str) - identifies event as a lane change event
        curveh - CalibrationVehicle object which has the leader change at time start
        newlead - The new leader for curveh. If the new leader is being simulated, curlead is a
            CalibrationVehicle, otherwise curlead is a float corresponding to the vehicle ID
        leadlen - If newlead is a float (i.e. the new leader is not simulated), curlen is the length of
            curlead. Otherwise, curlen is None (curlead.len gives the length for a CalibrationVehicle)
        userelax - bool, whether to apply relaxation
        leadstate - if the new leader is not simulated, leadstate is a tuple which gives the position/speed
            to use for computing the relaxation amount

    Args:
        event: lead change event
        timeind: time index
        dt: timestep
    """
    unused, unused, curveh, newlead, leadlen, userelax, leadstate = event

    # calculate relaxation amount
    if userelax:
        # get olds/oldv
        if curveh.lead is None:  # rule for merges
            olds, oldv = curveh.get_eql(curveh.speed), curveh.speed
        else:  # normal rule
            olds, oldv = curveh.hd, curveh.lead.speed

        # get news/newv
        uselen = newlead.len if leadlen is None else leadlen
        if leadstate[0] is None:
            newpos, newv = newlead.pos, newlead.speed
        else:
            newpos, newv = leadstate
        news = newpos - uselen - curveh.pos

        # apply relaxation
        relaxamounts = (olds - news, oldv - newv)
        curveh.set_relax(relaxamounts, timeind, dt)

    update_lead(curveh, newlead, leadlen, timeind)  # update leader

    #Adding lane to curveh/ Where are we getting the lane from?
    unused, unused, exitspeeds, unused = helper.boundaryspeeds(meas, [], [3],.1,.1)
    downstream = {'method': 'speed', 'time_series':exitspeeds}
    lane = simulation.Lane(None, None, None, None, downstream=downstream)
    curveh.lane = lane



def update_lead(curveh, newlead, leadlen, timeind):
    """Updates leader for curveh.

    Args:
        curveh: Vehicle to update
        newlead: if a float, the new leader is the LeadVehicle for curveh. Otherwise, newlead is the
            Vehicle object of the new leader
        leadlen: if newlead is a float, leadlen gives the length of the leader so LeadVehicle can be updated.
        timeind: time index of update (change happens at timeind+1)
    """
    if leadlen is None:  # newlead is simulated
        curveh.lead = newlead
        newlead.fol = curveh
        curveh.leadmem.append([newlead, timeind+1])
        curveh.in_leadveh = False
    else:  # LeadVehicle
        curveh.lead = curveh.leadveh
        curveh.lead.set_len(leadlen)  # must set the length of LeadVehicle
        curveh.leadmem.append([newlead, timeind+1])
        curveh.in_leadveh = True
