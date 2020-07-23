"""Refactors the functionality of the calibration.opt module.

The simulation module does an entire micro simulation. The calibration module is supposed to either
calibrate only the longitudinal, or only the latitudinal model. This allows direct comparison with the
data on a microscopic level.
"""

import numpy as np
from havsim.simulation.simulation import Vehicle, get_headway, relax_helper_vhd, relax_helper
import havsim.calibration.helper as helper
import math

# TODO finish implementing calibration for platoons of vehicles (handle downstream boundary, removing, etc.)
    #handle assigning parameters for multiple vehicles
    #removing vehicles and downstream boundary
# TODO implement calibration for latitudinal models only

class CalibrationVehicle(Vehicle):
    def __init__(self, vehid, y, initpos, initspd, inittime, leadstatemem, leadinittime, length=3,
                 accbounds=None, maxspeed=1e4, hdbounds=None, eql_type='v'):
        """Inits CalibrationVehicle."""
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
        relax_helper_vhd(rp, relaxamount_s, relaxamount_v, self, timeind, dt)

    def update(self, timeind, dt):
        """Update for longitudinal state. Updates LeadVehicle if applicable."""
        super().update(timeind, dt)

        if self.in_leadveh:
            self.leadveh.update(timeind+1)

    def loss(self):
        """Calculates loss."""
        return sum(np.square(np.array(self.posmem) - self.y))/len(self.posmem)

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
    """Used for simulating a vehicle which follows a predetermined trajectory - it has no models."""
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
        self.pos, self.speed = self.leadstatemem[timeind - self.inittime]

    def set_len(self, length):
        """Set len so headway can be computed correctly."""
        self.len = length


class Calibration:
    def __init__(self, vehicles, add_events, lc_events, dt, endtime = None):
        self.all_vehicles = vehicles
        self.all_add_events = add_events
        self.all_lc_events = lc_events

        self.starttime = add_events[-1][0]
        self.endtime = endtime
        self.dt = dt


    def step(self):
        for veh in self.vehicles:
            veh.set_cf(self.timeind, self.dt)

        self.addtime, self.lctime = update_calibration(self.vehicles, self.add_events, self.lc_events,
                                                       self.addtime, self.lctime, self.timeind, self.dt)

        self.timeind += 1

    def simulate(self, parameters):
        # reset all vehicles, and assign their parameters
        for veh in self.all_vehicles:
            veh.initialize(parameters)

        # initialize events, add the first vehicle(s), initialize vehicles headways/LeadVehicles
        self.vehicles = set()
        self.add_events = self.all_add_events.copy()
        self.lc_events = self.all_lc_events.copy()
        self.addtime = self.add_events[-1][0]
        self.lctime = self.lc_events[-1][0] if len(self.lc_events)>0 else math.inf
        self.timeind = self.starttime
        self.addtime = update_add_event(self.vehicles, self.add_events, self.addtime, self.timeind-1, self.dt)
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


def update_calibration(vehicles, add_events, lc_events, addtime, lctime, timeind, dt):
    lctime = update_lc_event(lc_events, lctime, timeind, dt)

    for veh in vehicles:
        veh.update(timeind, dt)

    addtime = update_add_event(vehicles, add_events, addtime, timeind, dt)

    for veh in vehicles:
        veh.hd = get_headway(veh, veh.lead)

    return addtime, lctime


def update_lc_event(lc_events, lctime, timeind, dt):
    if lctime == timeind+1:
        lc_event(lc_events.pop(), timeind, dt)
        lctime = lc_events[-1][0] if len(lc_events)>0 else math.inf
        if lctime == timeind+1:
            while lctime == timeind+1:
                lc_event(lc_events.pop(), timeind, dt)
                lctime = lc_events[-1][0] if len(lc_events)>0 else math.inf
    return lctime


def update_add_event(vehicles, add_events, addtime, timeind, dt):
    if addtime == timeind+1:
        add_event(add_events.pop(), vehicles, timeind, dt)
        addtime = add_events[-1][0] if len(add_events)>0 else math.inf
        if addtime == timeind+1:
            while addtime == timeind+1:
                add_event(add_events.pop(), vehicles, timeind, dt)
                addtime = add_events[-1][0] if len(add_events)>0 else math.inf
    return addtime


def make_calibration(vehicles, meas, platooninfo, dt, vehicle_class):
    vehicle_list = []
    addevent_list = []
    lcevent_list = []
    id2obj = {}
    if type(vehicles) == list:
        vehicles = set(vehicles)

    for veh in vehicles:
        # make lead memory - a list of position/speed tuples for any leaders which are not simulated
        leads = set(platooninfo[veh][4])
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
            leadstatemem, leadinittime = None

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
    for veh in vehicles:
        curveh = id2obj[veh]
        leadinfo = helper.makeleadinfo([veh],platooninfo,meas)[0]
        for count, j in enumerate(leadinfo):
            curlead, start, end = j
            leadt_nstar = platooninfo[curlead][0]
            # even though the change occurs at time start, we want to calculate the relaxation using
            # the differences in headway at time start - 1. This leads to 4 combinations, first, whether
            # the vehicle is simulated or not, and second, whether the vehicle is available at start-1
            if curlead in vehicles:  # curlead is simulated (in the same calibration object)
                if start-1 < leadt_nstar:  # handle edge case where t_nstar = start
                    leadstate = (meas[curlead][0,2], meas[curlead][0,3])
                    leadstate[0] += -leadstate[1]*dt
                else:
                    leadstate = (None,)
                curlead, curlen = id2obj[curlead], None
            else:
                curlen = meas[curlead][0,6]  # curlead is already simulated, stored in curveh.leadstatemem
                if start-1 < leadt_nstar:  # handle edge case where t_nstar = start
                    leadstate = (meas[curlead][0,2], meas[curlead][0,3])
                    leadstate[0] += -leadstate[1]*dt
                else:
                    leadstate = (meas[curlead][start-leadt_nstar,2], meas[curlead][start-leadt_nstar,3])

            if count == 0:  # add event adds vehicle to simulation
                t_nstar, t_n = platooninfo[veh][0:2]
                # if t_n > t_nstar and meas[veh][t_n-t_nstar-1,7]==7 and meas[veh][t_n-t_nstar,7]==6:
                if t_n > t_nstar:
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

    addevent_list.sort(key = lambda x: x[0], reverse = True)  # sort events in time
    lcevent_list.sort(key = lambda x: x[0], reverse = True)

    # make calibration object
    return Calibration(vehicle_list, addevent_list, lcevent_list, dt, endtime=endtime)


def add_event(event, vehicles, timeind, dt):
    """Adds a vehicle to the simulation and applies the first lead change event."""
    unused, unused, curveh, lcevent = event
    vehicles.add(curveh)
    lc_event(lcevent, timeind, dt)


def lc_event(event, timeind, dt):
    """Applies lead change event, updating a CalibrationVehicle's leader.

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


def update_lead(curveh, newlead, leadlen, timeind):
    """Updates leader for curveh. newlead is CalibrationVehicle (simulated) or float (use LeadVehicle)."""
    if leadlen is None:  # newlead is simulated
        curveh.lead = newlead
        curveh.leadmem.append([newlead, timeind+1])
        curveh.in_leadveh = False
    else:  # LeadVehicle
        curveh.lead = curveh.leadveh
        curveh.lead.set_len(leadlen)  # must set the length of LeadVehicle
        curveh.leadmem.append([newlead, timeind+1])
        curveh.in_leadveh = True