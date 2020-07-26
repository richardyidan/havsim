
"""
@author: rlk268@cornell.edu
"""
import havsim.simulation.calibration as hc
import time
import scipy.optimize as sc
import matplotlib.pyplot as plt
import math
import pickle
from havsim.calibration.algs import makeplatoonlist
import havsim.simulation.models as hm
import havsim.simulation.simulation as hs

# load data
try:
    with open('C:/Users/rlk268/OneDrive - Cornell University/important misc/datasets/trajectory data/mydata.pkl', 'rb') as f:
        rawdata, truedata, data, trueextradata = pickle.load(f) #load data
except:
    with open('/home/rlk268/data/mydata.pkl', 'rb') as f:
        rawdata, truedata, data, trueextradata = pickle.load(f) #load data

meas, platooninfo = makeplatoonlist(data,1,False)

# categorize vehicles
veh_list = meas.keys()
merge_list = []
lc_list = []
nolc_list = []
for veh in veh_list:
    t_nstar, t_n = platooninfo[veh][0:2]
    if t_n > t_nstar and meas[veh][t_n-t_nstar-1,7]==7 and meas[veh][t_n-t_nstar,7]==6:
        merge_list.append(veh)
    elif len(platooninfo[veh][4]) > 1:
        lc_list.append(veh)
    elif len(platooninfo[veh][4]) == 1:
        nolc_list.append(veh)

# define training loop
def training_ga(veh_id_list, bounds, meas, platooninfo, dt, vehicle_object, workers = 2):
    """Runs differential evolution to fit parameters for a list of CalibrationVehicle's"""
    #veh_id_list = list of float vehicle id, bounds = bounds for optimizer (list of tuples),
    #vehicle_object = (possibly subclassed) CalibrationVehicle object
    out = []
    for veh_id in veh_id_list:
        cal = hc.make_calibration([veh_id], meas, platooninfo, dt, vehicle_object)
        ga = sc.differential_evolution(cal.simulate, bounds = bounds, workers = workers)
        out.append(ga)

    return out

#%%
"""
Run 1: IDM with no accident-free relax, no max speed bound, no acceleration bound (only for merge, lc)
"""
bounds = [(20,120),(.1,5),(.1,35),(.1,20),(.1,20),(.1,75)]
relax_lc_res = training_ga(lc_list, bounds, meas, platooninfo, .1, hc.CalibrationVehicle)
relax_merge_res = training_ga(merge_list, bounds, meas, platooninfo, .1, hc.CalibrationVehicle)

with open('IDMrelax.pkl','wb') as f:
    pickle.dump((relax_lc_res,relax_merge_res), f)

"""
Run 2: Like Run 1, but with relax disabled. (for all vehicles)
"""
#subclass calibrationvehicle as necessary
class NoRelaxIDM(hc.CalibrationVehicle):
    def set_relax(self, *args):
        pass

    def initialize(self, parameters):  # just need to set parameters correctly
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
        self.cf_parameters = parameters

bounds = [(20,120),(.1,5),(.1,35),(.1,20),(.1,20)]
norelax_lc_res = training_ga(lc_list, bounds, meas, platooninfo, .1 ,NoRelaxIDM)
norelax_merge_res = training_ga(merge_list, bounds, meas, platooninfo, .1, NoRelaxIDM)
norelax_nolc_res = training_ga(nolc_list, bounds, meas, platooninfo, .1, NoRelaxIDM)

with open('IDMnorelax.pkl','wb') as f:
    pickle.dump((norelax_lc_res,norelax_merge_res,norelax_nolc_res),f)

"""
Run 3: OVM with no accident-free relax, no max speed bound, no acceleration bound (only for merge, lc)
"""
# make OVM calibrationvehicle
class OVMCalibrationVehicle(hc.CalibrationVehicle):
    def cf_model(self, p, state):
        return hm.OVM(p, state)

    def get_cf(self, hd, spd, lead, curlane, timeind, dt, userelax):
        if lead is None:
            acc = curlane.call_downstream(self, timeind, dt)

        else:
            if self.in_relax:
                # accident free formulation of relaxation
                # ttc = hd / (self.speed - lead.speed)
                # if ttc < 1.5 and ttc > 0:
                if False:  # disable accident free
                    temp = (ttc/1.5)**2
                    # currelax, currelax_v = self.relax[timeind-self.relax_start]  # hd + v relax
                    # currelax, currelax_v = currelax*temp, currelax_v*temp
                    currelax = self.relax[timeind - self.relax_start]*temp
                else:
                    # currelax, currelax_v = self.relax[timeind-self.relax_start]
                    currelax = self.relax[timeind - self.relax_start]

                # acc = self.cf_model(self.cf_parameters, [hd + currelax, spd, lead.speed + currelax_v])
                acc = self.cf_model(self.cf_parameters, [hd + currelax, spd, lead.speed])
            else:
                acc = self.cf_model(self.cf_parameters, [hd, spd, lead.speed])

        return acc

    def eqlfun(self, p, s):
        return hm.OVM_eql(p, s)

    def set_relax(self, relaxamounts, timeind, dt):
        rp = self.relax_parameters
        if rp is None:
            return
        relaxamount_s, relaxamount_v = relaxamounts
        hs.relax_helper(rp, relaxamount_s, self, timeind, dt)

    def initialize(self, parameters):
        super().initialize(parameters)
        self.maxspeed = parameters[0]*(1-math.tanh(-parameters[2]))-.1
        self.eql_type = 's'  # you are supposed to set this in __init__

bounds = [(20,120),(.001,.1),(.1,2),(.1,5),(0,3), (.1,75)]
relax_lc_res_ovm = training_ga(lc_list, bounds, meas, platooninfo, .1, OVMCalibrationVehicle)
relax_merge_res_ovm = training_ga(merge_list, bounds, meas, platooninfo, .1, OVMCalibrationVehicle)

with open('OVMrelax.pkl', 'wb') as f:
    pickle.dump((relax_lc_res_ovm, relax_merge_res_ovm),f)


"""
Run 4: Like Run 3, but with relax disabled. (for all vehicles)
"""

class NoRelaxOVM(OVMCalibrationVehicle):
    def set_relax(self, *args):
        pass

    def initialize(self, parameters):
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
        self.cf_parameters = parameters

bounds = [(20,120),(.001,.1),(.1,2),(.1,5),(0,3)]
norelax_lc_res_ovm = training_ga(lc_list, bounds, meas, platooninfo, .1, NoRelaxOVM)
norelax_merge_res_ovm = training_ga(merge_list, bounds, meas, platooninfo, .1, NoRelaxOVM)
norelax_nolc_res_ovm = training_ga(nolc_list, bounds, meas, platooninfo, .1, NoRelaxOVM)

with open('OVMnorelax.pkl', 'wb') as f:
    pickle.dump((norelax_lc_res_ovm, norelax_merge_res_ovm, norelax_nolc_res_ovm),f)


#%%
"""
Run 5: Newell with no accident free
"""
class NewellCalibrationVehicle(hc.CalibrationVehicle):
    """Implementation of Newell model in Differential form, example of 1st order ODE implementation."""
    def cf_model(self, p, state):
        """p = parameters, state = headway"""
        return (state - p[1])/p[0]

    def get_cf(self, hd, lead, curlane, timeind, dt, userelax):
        if lead is None:
            acc = curlane.call_downstream(self, timeind, dt)

        else:
            if self.in_relax:
                currelax = self.relax[timeind - self.relax_start]
                spd = self.cf_model(self.cf_parameters, hd+currelax)
            else:
                spd = self.cf_model(self.cf_parameters, hd)
        return spd

    def set_cf(self, timeind, dt):
        self.speed = self.get_cf(self.hd, self.lead, self.lane, timeind, dt, self.in_relax)

    def eqlfun(self, p, v):
        return p[0]*v+p[1]

    def set_relax(self, relaxamounts, timeind, dt):
        rp = self.relax_parameters
        if rp is None:
            return
        relaxamount_s, relaxamount_v = relaxamounts
        hs.relax_helper(rp, relaxamount_s, self, timeind, dt)

    def update(self, timeind, dt):
        # bounds on speed must be applied for 1st order model
        curspeed = self.speed
        if curspeed < 0:
            curspeed = 0
        elif curspeed > self.maxspeed:
            curspeed = self.maxspeed
        # update state
        self.pos += curspeed*dt
        self.speed = curspeed
        # update memory
        self.posmem.append(self.pos)
        self.speedmem.append(self.speed)
        if self.in_relax:
            if timeind == self.relax_end:
                self.in_relax = False
                self.relaxmem.append((self.relax_start, self.relax))

        if self.in_leadveh:  # only difference is we update the LeadVehicle if applicable
            self.leadveh.update(timeind+1)


    def initialize(self, parameters):
        super().initialize(parameters)  # before the first cf call, the speed is initialized as initspd.
        # this handles the edge case for if a vehicle tries to access the speed before the first cf call.
        # after the first cf call, in this case the speed will simply be the speed from the previous timestep
        self.speedmem = []  # note that speedmem will be 1 len shorter than posmem for a 1st order model
        self.maxspeed = parameters[2]


bounds = [(.1,10),(0,100),(40,120),(.1,75)]
relax_lc_res_newell = training_ga(lc_list, bounds, meas, platooninfo, .1, NewellCalibrationVehicle)
relax_merge_res_newell = training_ga(merge_list, bounds, meas, platooninfo, .1, NewellCalibrationVehicle)

with open('Newellrelax.pkl','wb') as f:
    pickle.dump([relax_lc_res_newell, relax_merge_res_newell], f)


"""
Run 6: Like Run 5, but with no relax
"""
class NoRelaxNewell(NewellCalibrationVehicle):
    def set_relax(self, *args):
        pass

    def initialize(self, parameters):
        super().initialize(parameters)
        self.cf_parameters = parameters
        self.relax_parameters = None

bounds = [(.1,10),(0,100),(40,120)]
norelax_lc_res_newell = training_ga(lc_list, bounds, meas, platooninfo, .1, NoRelaxNewell)
norelax_merge_res_newell = training_ga(merge_list, bounds, meas, platooninfo, .1, NoRelaxNewell)
norelax_nolc_res_newell = training_ga(nolc_list, bounds, meas, platooninfo, .1, NoRelaxNewell)

with open('Newellnorelax.pkl','wb') as f:
    pickle.dump([norelax_lc_res_newell, norelax_merge_res_newell, norelax_nolc_res_newell], f)
