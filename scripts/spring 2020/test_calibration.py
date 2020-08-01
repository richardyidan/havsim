
"""
Tests the simulation.calibration code. Compare to scripts dothecalibration, relaxtest in
"/scripts/2018 AY .../useful misc/"
Need data loaded in meas/platooninfo format
"""

import havsim.simulation.calibration as hc
import time
import scipy.optimize as sc
import matplotlib.pyplot as plt
import havsim.simulation.calibration_models as hm
import math

use_model = 'Newell'   # change to one of IDM, OVM, Newell
curplatoon = [lc_list[101]]  # test vehicle to calibrate
use_method = 'GA' # GA or BFGS
if __name__ == '__main__':
    if use_model == 'IDM':
        pguess =  [40,1,1,3,10,25] #[80,1,15,1,1,35] #
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
    elif use_model == '2IDM':
        pguess =  [40,1,1,3,10,25,25]
        mybounds = [(20,120),(.1,5),(.1,35),(.1,20),(.1,20),(.1,75), (.1, 75)]
        cal = hc.make_calibration(curplatoon, meas, platooninfo, .1, Relax2IDM)
    elif use_model == 'ShapeIDM':
        pguess =  [80,1,15,1,1,35, -.5] #[40,1,1,3,10,25,.5]
        mybounds = [(20,120),(.1,5),(.1,35),(.1,20),(.1,20),(.1,75), (-1,1)]
        cal = hc.make_calibration(curplatoon, meas, platooninfo, .1, RelaxShapeIDM)

    start = time.time()
    cal.simulate(pguess)
    print('time to compute loss is '+str(time.time()-start))

    start = time.time()
    if use_method == 'BFGS':
        bfgs = sc.fmin_l_bfgs_b(cal.simulate, pguess, bounds = mybounds, approx_grad=1)  # BFGS
        print('time to calibrate is '+str(time.time()-start)+' to find mse '+str(bfgs[1]))
    elif use_method == 'GA':
        bfgs = sc.differential_evolution(cal.simulate, bounds = mybounds, workers = 1)  # GA
        print('time to calibrate is '+str(time.time()-start)+' to find mse '+str(bfgs['fun']))

    plt.plot(cal.all_vehicles[0].speedmem)
    plt.ylabel('speed')
    plt.xlabel('time index')

