
"""
@author: rlk268@cornell.edu
"""
import havsim
from havsim.plotting import animatevhd, animatetraj, plotvhd, plotColorLines, plotflows
import numpy as np
import matplotlib.pyplot as plt

meas, platooninfo = havsim.calibration.algs.makeplatoonlist(data, 1, False)

#%%
mergelist = []
for i in platooninfo.keys():
    t_nstar, t_n = platooninfo[i][:2]
    if meas[i][0,7] == 7:
        if meas[i][t_n-t_nstar,7]==6:
            mergelist.append(i)

lane6vehlist = []
for i in platooninfo.keys():
    if 6 in np.unique(meas[i][:,7]):
        lane6vehlist.append(i)   
sortedvehlist = havsim.calibration.algs.sortveh3(lane6vehlist, 6, meas, platooninfo)

#%%
#animatevhd(meas, None, platooninfo, sortedvehlist[113:115], interval = 30, lane = 6)
veh = mergelist[i]
plotvhd(meas, None, platooninfo, [mergelist[5]], draw_arrow = True, plot_color_line = True)

plt.figure()
t_nstar, t_n = platooninfo[veh][:2]
plt.plot(meas[veh][0:t_n-t_nstar,3])

#%%
plt.figure()
plotflows(meas,[[800,1200]],[0,10*60*14.5],30*10,type = 'FD',lane = 6, method = 'area')