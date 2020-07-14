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

        self.initlead = initlead
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

def make_calibration(vehicles, meas, platooninfo):
    leadvehs = {}
    vehicle_list = []
    event_list = []

    for veh in vehicles:
        # see what lead vehicles veh needs, if any. Results are stored in leadvehs where keys are the leaders,
        # values are a list of starting, ending times
        leads = set(platooninfo[veh][4])
        needleads = leads.difference(vehicles)
        if len(needleads) > 0:
            leadinfo = helper.makeleadinfo([veh],platooninfo,meas)
            for j in leadinfo[0]:
                curlead, start, end = j
                if curlead in needleads:
                    if curlead in leadvehs:
                        leadvehs[curlead].extend([start, end])
                    else:
                        leadvehs[curlead] = [start, end]

        # get initial values, y for veh
        t_nstar, inittime, endtime = platooninfo[veh][0,1,2]
        initlead, initpos, initspd, length = meas[veh][inittime-t_nstar,[4,2,3,6]]
        y = meas[veh][inittime-t_nstar:endtime+1-t_nstar]

        #make the CalibrationVehicle object for veh
        newveh = CalibrationVehicle(veh, y, initpos, initspd, initlead, length = length)
        vehicle_list.append(newveh)

        #make the add/remove events for veh
        event_list.append((inittime, 'add', newveh))
        event_list.append((endtime, 'remove', newveh))

        #make the lane changing events for veh

        event_list.append()


        if len(needleads) > 0:
            leadid = 'lead'+str(int(veh))  # hash value
            leadpos, leadspd, leadlen = [], [], []  # list of positions, speeds, lengths
            leadinittime = None  # initial time lead is used
            leadinfo = helper.makeleadinfo([veh],platooninfo,meas)
            for j in leadinfo[0]:
                curlead, start, end = j
                if curlead in needleads:
                    if not leadinittime:
                        leadinittime = start
                    leadt_n = platooninfo[curlead][0]
                    leadpos.extend(list(meas[curlead][start-leadt_n:end-leadt_n+1,2]))
                    leadspd.extend(list(meas[curlead][start-leadt_n:end-leadt_n+1,3]))
                    leadlen.extend(list(meas[curlead][start-leadt_n:end-leadt_n+1,6]))
                elif leadinittime is not None:
                    temp = [0]*(end-start+1)
                    leadpos.extend(temp), leadspd.extend(temp), leadlen.extend(temp)
            leadendtime = leadinittime + len(leadpos)-1







    pass

def optimize():
    pass

