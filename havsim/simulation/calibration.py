"""Refactors the functionality of the calibration.opt module."""

import numpy as np
from havsim.simulation import Vehicle
import havsim.calibration.helper as helper

class CalibrationVehicle(Vehicle):
    def __init__(self, vehid, y, initpos, initspd, initlead, length=3, accbounds=None,
                  maxspeed=1e4, hdbounds=None, eql_type='v'):
        self.vehid = vehid
        self.len = length
        # self.cf_parameters = cf_parameters
        self.y = y
        self.initpos = initpos
        self.initspd = initspd

        # self.relax_parameters = relax_parameters
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

        # self.lead = lead
        # self.fol = fol

    def reset(self):
        pass

    def initialize(self):
        pass

    def dbc(self):
        RuntimeError('not implemented')  # TODO implement boundary conditions for platoons


class LeadVehicle:
    """Used for simulating a vehicle which follows a predetermined trajectory - it has no models."""
    def __init__(self, posmem, speedmem, length, inittime):
        """
        posmem - list of positions
        speedmem - list of speeds
        length - vehicle length
        inittime - time corresponding to the 0 index of memory
        """
        self.posmem = posmem
        self.speedmem = speedmem
        self.len = length
        self.inittime
        self.cf_parameters = None

    def update(self, timeind, dt):
        """position/speed are determined by posmem/speedmem"""
        temp = timeind - self.inittime
        self.pos = self.posmem[temp]
        self.speed = self.speedmem[temp]

    def set_cf(self, *args):
        pass

    def set_relax(self, *args):
        pass

    def loss(self, *args):
        pass

    def reset(self):
        pass

    def initialize(self):
        pass


class Calibration:
    def __init__(vehicles):
        pass

    def step(self, timeind):
        for veh in self.vehicles:
            veh.set_cf(self.timeind, self.dt)



def make_calibration(vehicles, meas, platooninfo):
    vehicle_list = []
    event_list = []
    id2obj = {}
    if type(vehicles) == list:
        vehicles = set(vehicles)

    for veh in vehicles:
        #make lead memory
        leads = set(platooninfo[veh][4])
        needleads = leads.difference(vehicles)
        if len(needleads) > 0:
            leadmem = []
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
                    leadmem.extend(zip(poslist,spdlist))
                elif leadinittime is not None:
                    temp = [0]*(end-start+1)
                    leadmem.extend(temp)

        # get initial values, y for veh
        t_nstar, inittime, endtime = platooninfo[veh][0,1,2]
        initlead, initpos, initspd, length = meas[veh][inittime-t_nstar,[4,2,3,6]]
        y = meas[veh][inittime-t_nstar:endtime+1-t_nstar]

        #create vehicle object
        newveh = CalibrationVehicle(veh, y, initpos, initspd, length=length)
        vehicle_list.append(newveh)
        id2obj[veh] = newveh

    # create events
    for veh in vehicles:
        curveh = id2obj[veh]
        leadinfo = helper.makeleadinfo([veh],platooninfo,meas)[0]
        # first make the add event, which includes handling the first leader
        curlead, start, end = leadinfo[0]
        if curlead in vehicles:  # curlead is simulated in the same calibration object
            curlead = id2obj[curlead]
            curlen = curlead.len
        else:
            curlen = meas[curlead][0,6]  # curlead is already simulated, stored in curveh.leadmem
            curlead = None
        # check if curveh is a merging vehicle
        t_nstar, t_n = platooninfo[veh][0,1]
        if t_n > t_nstar and meas[veh][t_n-t_nstar-1,7]==7 and meas[veh][t_n-t_nstar,7]==6:
            userelax = True
        else:
            userelax = False

        # make the add event
        curevent = (start, 'lc', curveh, curlead, curlen, userelax)
        curevent = (start, 'add', curveh, curevent)
        event_list.append(curevent)

        # make the lead change events
        for j in leadinfo[1:]:
            curlead, start, end = j
            if curlead in vehicles:  # curlead is simulated in the same calibration object
                curlead = id2obj[curlead]
                curlen = curlead.len
            else:
                curlen = meas[curlead][0,6]  # curlead is already simulated, stored in curveh.leadmem
                curlead = None
            curevent = (start, 'lc', curveh, curlead, curlen, True)
            event_list.append(curevent)

    event_list.sort(key = lambda x: x[0])  # sort events in time

def optimize():
    pass

