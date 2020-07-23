
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
    else:
        nolc_list.append(veh)

# define training loop
def training(veh_id, plist, bounds, meas, platooninfo, dt, vehicle_object):
    cal = hc.make_calibration([veh_id], meas, platooninfo, dt, vehicle_object)
    bestmse = math.inf
    best = None
    for guesses in plist:
        guess = guesses
        bfgs = sc.fmin_l_bfgs_b(cal.simulate, guess, bounds = bounds, approx_grad=1)



"""
Run 1: IDM with no accident-free relax, no max speed bound, no acceleration bound (only for merge, lc)
"""


"""
Run 2: Like Run 1, but with relax disabled. (for all vehicles)
"""


"""
Run 3: OVM with no accident-free relax, no max speed bound, no acceleration bound (only for merge, lc)
"""

"""
Run 4: Like Run 3, but with relax disabled. (for all vehicles)
"""
