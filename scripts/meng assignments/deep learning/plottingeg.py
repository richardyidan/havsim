
"""
@author: rlk268@cornell.edu
"""

#assume you have data loaded already, have access to meas and platooninfo

from havsim.plotting import plotvhd_v2
import matplotlib.pyplot as plt
from havsim.calibration.algs import makeplatoonlist
import pickle
from sklearn.cluster import KMeans
import havsim
import numpy as np


path_highd26 = '/Users/nathanbala/Desktop/meng_project/data/highd26.pkl'
with open(path_highd26, 'rb') as f:
   data = pickle.load(f)[0]

meas, platooninfo = makeplatoonlist(data,1, False)


with open("simhighd_info_relax.pickle", 'rb') as f:
    data = pickle.load(f)

high_error = []
low_error = []

lange_change_error = []
stay_error = []
for i in data:
    t_nstar, t_n, T_nm1, T_n = platooninfo[i][:4]
    curmeas = meas[i][t_n-t_nstar:T_nm1+1-t_nstar,[2,3,7]]
    curr_lanes = curmeas[:,2]
    curr = np.unique(curr_lanes)
    data_tup = data[i]
    rmse = data_tup[0]
    if rmse != None:
        error = rmse.numpy()
        lane_change = data_tup[1]
        if lane_change == True:
            lange_change_error.append(error)
        else:
            stay_error.append(error)

print(np.mean(lange_change_error))
print(np.mean(stay_error))





plotvhd_v2(meas,None, platooninfo, [207.0])

plotvhd_v2(meas,None, platooninfo, [2868.0, 2870.0, 2880.0, 2886.0, 2888.0,])
