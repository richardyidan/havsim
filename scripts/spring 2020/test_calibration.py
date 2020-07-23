
"""
Tests the simulation.calibration code. Compare to scripts dothecalibration, relaxtest in
"/scripts/2018 AY .../useful misc/"
Need data loaded in meas/platooninfo format
"""

import havsim.simulation.calibration as hc
import time
import scipy.optimize as sc
import matplotlib.pyplot as plt
# parameters
pguess =  [40,1,1,3,10,25] #IDM
mybounds = [(20,120),(.1,5),(.1,35),(.1,20),(.1,20),(.1,75)]
curplatoon = [399]
cal = hc.make_calibration(curplatoon, meas, platooninfo, .1, hc.CalibrationVehicle)

start = time.time()
cal.simulate(pguess)
print('time to compute loss is '+str(time.time()-start))

start = time.time()
# bfgs = sc.fmin_l_bfgs_b(cal.simulate, pguess, bounds = mybounds, approx_grad=1)
# print('time to calibrate is '+str(time.time()-start)+' to find mse '+str(bfgs[1]))

plt.plot(cal.all_vehicles[0].speedmem)

