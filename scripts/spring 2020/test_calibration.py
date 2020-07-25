
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
        self.maxspeed = parameters[0]*(1-math.tanh(-parameters[2]))
        self.eql_type = 's'  # you are supposed to set this in __init__
        

# parameters
pguess =  [80,1,15,1,1,35] #IDM  #[40,1,1,3,10,25]
# mybounds = [(20,120),(.1,3),(.1,40),(.1,10),(.1,10),(.1,60)]
mybounds = [(20,120),(.1,5),(.1,35),(.1,20),(.1,20),(.1,75)]

# pguess = [10*3.3,.086/3.3, 1.545, 2, .175, 5 ] #OVM
# mybounds = [(20,120),(.001,.1),(.1,2),(.1,5),(0,3), (.1,75)] #OVM

curplatoon = [2040]
cal = hc.make_calibration(curplatoon, meas, platooninfo, .1, hc.CalibrationVehicle)
# cal = hc.make_calibration(curplatoon, meas, platooninfo, .1, OVMCalibrationVehicle)

start = time.time()
cal.simulate(pguess)
print('time to compute loss is '+str(time.time()-start))

start = time.time()
# bfgs = sc.fmin_l_bfgs_b(cal.simulate, pguess, bounds = mybounds, approx_grad=1)
bfgs = sc.differential_evolution(cal.simulate, bounds = mybounds)
print('time to calibrate is '+str(time.time()-start)+' to find mse '+str(bfgs['fun']))
# print('time to calibrate is '+str(time.time()-start)+' to find mse '+str(bfgs[1]))

plt.plot(cal.all_vehicles[0].speedmem)

