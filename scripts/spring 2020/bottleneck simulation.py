
"""
@author: rlk268@cornell.edu
"""
from havsim.simulation.simulation import vehicle, lane
from havsim.calibration.helper import boundaryspeeds, getentryflows
from havsim.plotting import calculateflows
import numpy as np
import matplotlib.pyplot as plt

#%%get boundary conditions (careful with units)
#option 1 - 
#could get them directly from data 
entryflows, unused = getentryflows(meas, [3],.1,.25) 
unused, unused, exitspeeds, unused = boundaryspeeds(meas, [], [3],.1,.1)

#option 2 - use calculateflows, which has some aggregation in it and uses a different method to compute flows
q,k = calculateflows(meas, [[200,600],[1000,1400]], [0, 9900], 30*10, lane = 6) 

#option 3 - can also just make boudnary conditions based on what the FD looks like
tempveh = vehicle(-1, None, [33, 1.2, 2, 1.1, 1.5], None)
spds = np.arange(0,33,.01)
flows = np.array([tempveh.get_flow(i) for i in spds])
density = np.divide(flows,spds)
plt.plot(density,flows)

#%%


#make road network - want to make an api for this in the future 
road = {'name': 'main road', 'len': 800, 'laneinds':2, 0: None, 1: None}
road['connect to'] = {'exit': (800, 'continue', (0,1), None, None)}
onramp = {'name': 'on ramp', 'len': 200, 'laneinds':1, 0: None}
onramp['connect to'] = {'main road': ((100,200), 'merge', 0, 'l', road)}
#lane0 = lane(0,800, road1, 0, connect_right)