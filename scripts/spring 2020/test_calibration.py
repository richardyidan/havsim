
"""
Tests the simulation.calibration code. Compare to scripts dothecalibration, relaxtest in
"/scripts/2018 AY .../useful misc/"
Need data loaded in meas/platooninfo format
"""

import havsim.simulation.calibration as hc
import time
import scipy.optimize as sc
import matplotlib.pyplot as plt
import havsim.simulation.models as hm
import havsim.simulation.simulation as hs
import math

# for testing OVM
class OVMCalibrationVehicle(hc.CalibrationVehicle):
    """Optimal Velocity Model Implementation."""
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


# for Newell
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
        
        
class SKA_IDM(hc.CalibrationVehicle):
    """IDM with a relaxation model based on Schakel, Knoop, van Arem (2012).
    
    In the original paper, they give a full microsimulation model, and the relaxation is integrated in the 
    sense that the 'desire' parameter controls both the gap acceptance as well as the relaxation amount.
    In this implementation, the relaxation amount is its own parameter, thus it has two relax parameters,
    the first being the desire which controls the relaxation amount, and the second being the rate
    of change for the relaxation.
    """
    def initialize(self, parameters):
        super().initialize(parameters)
        self.cf_parameters = parameters[:-2]
        self.relax_parameters = parameters[-2:]
        self.relax_end = math.inf
        self.max_relax = parameters[1]
    
    def set_relax(self, relaxamounts, timeind, dt):
        # in_relax is always False, and we implement the relaxation by just 
        # changing the time headway (cf_parameter[1]) appropriately
        self.relax_start = 'r'  # give special value 'r' in case we need to be adjusting the time headway
        temp = dt/self.relax_parameters[1]
        self.cf_parameters[1] = (self.relax_parameters[0] - self.max_relax*temp)/(1-temp)  # handle first 
        # relaxation value correctly (because it will be updated once before being used)
        
    def update(self, timeind, dt):
        super().update(timeind, dt)
        
        if self.relax_start == 'r':
            temp = dt/self.relax_parameters[1]
            self.cf_parameters[1] += (self.max_relax-self.cf_parameters[1])*temp
        
    

use_model = 'SKA'   # change to one of IDM, OVM, Newell
curplatoon = [2910]  # test vehicle to calibrate
if __name__ == '__main__':
    if use_model == 'IDM':
        pguess =  [40,1,1,3,10,25] #[80,1,15,1,1,35]
        mybounds = [(20,120),(.1,5),(.1,35),(.1,20),(.1,20),(.1,75)]
        cal = hc.make_calibration(curplatoon, meas, platooninfo, .1, hc.CalibrationVehicle)
    elif use_model == 'Newell':
        pguess = [1,40,100,5]
        mybounds = [(.1,10),(0,100),(40,120),(.1,75)]
        cal = hc.make_calibration(curplatoon, meas, platooninfo, .1, NewellCalibrationVehicle)
    elif use_model == 'OVM':
        pguess = [10*3.3,.086/3.3, 1.545, 2, .175, 5 ]
        mybounds = [(20,120),(.001,.1),(.1,2),(.1,5),(0,3), (.1,75)]
        cal = hc.make_calibration(curplatoon, meas, platooninfo, .1, OVMCalibrationVehicle)
    elif use_model == 'SKA':
        pguess =  [40,1,1,3,10,.5,25] #[80,1,15,1,1,35]
        mybounds = [(20,120),(.1,5),(.1,35),(.1,20),(.1,20),(.1,5),(.1,75)]
        cal = hc.make_calibration(curplatoon, meas, platooninfo, .1, SKA_IDM)
    start = time.time()
    cal.simulate(pguess)
    print('time to compute loss is '+str(time.time()-start))

    start = time.time()
    # bfgs = sc.fmin_l_bfgs_b(cal.simulate, pguess, bounds = mybounds, approx_grad=1)  # BFGS
    # print('time to calibrate is '+str(time.time()-start)+' to find mse '+str(bfgs[1]))
    bfgs = sc.differential_evolution(cal.simulate, bounds = mybounds, workers = 2)  # GA
    print('time to calibrate is '+str(time.time()-start)+' to find mse '+str(bfgs['fun']))

    plt.plot(cal.all_vehicles[0].speedmem)
    plt.ylabel('speed')
    plt.xlabel('time index')

