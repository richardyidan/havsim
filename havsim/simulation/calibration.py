"""Refactors the functionality of the calibration.opt module."""

import numpy as np
from havsim.simulation.simulation import Vehicle
import havsim.calibration.helper as helper

class CalibrationVehicle(Vehicle):
    def __init__(self, vehid, y, initpos, initspd, leadstatemem, leadinittime, length=3,
                 accbounds=None, maxspeed=1e4, hdbounds=None, eql_type='v'):
        self.vehid = vehid
        self.len = length
        self.y = y
        self.initpos = initpos
        self.initspd = initspd

        self.road = None
        self.lane = None

        self.in_relax = False
        self.relax = None
        self.relax_start = None

        if accbounds is None:
            self.minacc, self.maxacc = -7, 3
        else:
            self.minacc, self.maxacc = accbounds[0], accbounds[1]
        self.maxspeed = maxspeed
        self.hdbounds = (0, 1e4) if hdbounds is None else hdbounds
        self.eql_type = eql_type

        if leadstatemem is not None:
            self.leadveh = LeadVehicle(leadstatemem, leadinittime)

    def loss(self):
        return sum(np.square(np.array(self.posmem) - np.array(self.y)))

    def reset(self):
        pass

    def initialize(self):
        pass

    def dbc(self):
        RuntimeError('not implemented')  # TODO implement boundary conditions for platoons


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
        self.pos, self.spd = self.leadstatemem[timeind - self.inittime]

    def set_len(self, length):
        """Set len so headway can be computed correctly."""
        self.len = length


class Calibration:
    def __init__(vehicles):
        pass

    def step(self, timeind):
        for veh in self.vehicles:
            veh.set_cf(self.timeind, self.dt)



def make_calibration(vehicles, meas, platooninfo, dt):
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
        t_nstar, inittime, endtime = platooninfo[veh][0,1,2]
        initlead, initpos, initspd, length = meas[veh][inittime-t_nstar,[4,2,3,6]]
        y = meas[veh][inittime-t_nstar:endtime+1-t_nstar]

        # create vehicle object
        newveh = CalibrationVehicle(veh, y, initpos, initspd, leadstatemem, leadinittime, length=length)
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

            if count == 0:  # add event adds vehicle to simulation, checks for merge
                t_nstar, t_n = platooninfo[veh][0,1]
                if t_n > t_nstar and meas[veh][t_n-t_nstar-1,7]==7 and meas[veh][t_n-t_nstar,7]==6:
                    userelax, leadstate = True, (None,)
                else:
                    userelax, leadstate = False, (None,)
                # make the add event
                curevent = (start, 'lc', curveh, curlead, curlen, userelax, leadstate)
                curevent = (start, 'add', curveh, curevent)
                addevent_list.append(curevent)
            else:  # lc event changes leader, applies relax
                curevent = (start, 'lc', curveh, curlead, curlen, userelax, leadstate)
                lcevent_list.append(curevent)

    addevent_list.sort(key = lambda x: x[0])  # sort events in time
    lcevent_list.sort(key = lambda x: x[0])


def add_event(event):


    pass


def lc_event(event):
    """Applies lead change event, updating a CalibrationVehicle's leader.

    Lead change events are a tuple of
        start (float) - time index of the event
        'lc' (str) - identifies event as a lane change event
        curveh - CalibrationVehicle object which has the leader change at time start
        curlead - The new leader for curveh. If the new leader is being simulated, curlead is a
            CalibrationVehicle, otherwise curlead is a float corresponding to the vehicle ID
        curlen - If curlead is a float (i.e. the new leader is not simulated), curlen is the length of
            curlead. Otherwise, curlen is None (curlead.len gives the length for a CalibrationVehicle)
        userelax - bool, whether to apply relaxation
        leadstate - if the new leader is not simulated, leadstate is a tuple which gives the position/speed
            to use for computing the relaxation amount
    """
    unused, unused, curveh, curlead, curlen, userelax, oldlead = event

    #calculate relaxation amount
    if userelax:
        if not curveh.leadmem:  # relaxation for merge
            olds, oldv = curveh.get_eql(curveh.speed), curveh.speed
        elif oldlead[0] is None:  # old leader is simulated
            olds, oldv  = curveh.hd, curveh.lead.speed
        else:  # old leader is not simulated
            olds, oldv = curveh.hd + oldlead[0], oldlead[1]


    pass

def optimize():
    pass

"""
Note on relaxation

This relaxation is calculated slightly differently than that for the simulation module. This is because

"""


# class LeadVehicle:
#     """Used for simulating a vehicle which follows a predetermined trajectory - it has no models."""
#     def __init__(self, posmem, speedmem, length, inittime):
#         """
#         posmem - list of positions
#         speedmem - list of speeds
#         length - vehicle length
#         inittime - time corresponding to the 0 index of memory
#         """
#         self.posmem = posmem
#         self.speedmem = speedmem
#         self.len = length
#         self.inittime
#         self.cf_parameters = None

#     def update(self, timeind, dt):
#         """position/speed are determined by posmem/speedmem"""
#         temp = timeind - self.inittime
#         self.pos = self.posmem[temp]
#         self.speed = self.speedmem[temp]

#     def set_cf(self, *args):
#         pass

#     def set_relax(self, *args):
#         pass

#     def loss(self, *args):
#         pass

#     def reset(self):
#         pass

#     def initialize(self):
#         pass

