
"""
@author: rlk268@cornell.edu
"""
import numpy as np
import matplotlib.pyplot as plt
from havsim.simulation.simulation import vehicle
from havsim.simulation.models import IDM, IDM_eql

class lane:
    def __init__(self, speedfun, connect_left = [(0, None)], connect_right = [(0, None)]):
#        self.call_downstream = downstream_wrapper(speed_fun = speedfun, method = 'speed').__get__(self, lane)
        self.timeseries = speedfun
        self.road = None
        self.events = []
        self.connect_left = connect_left
        self.connect_right = connect_right
    def get_headway(self, veh, lead):
        return lead.pos-veh.pos - lead.length
    
    def call_downstream(self, veh, timeind, dt):
        speed = self.timeseries[timeind]
        return (speed - veh.speed)/dt
    
    def get_connect_left(self, pos):
        #given position, returns the connection to left 
        #output is either lane object or None
        return connect_helper(self.connect_left, pos)

    def get_connect_right(self, pos):
        return connect_helper(self.connect_right,pos)
    
def connect_helper(connect, pos):
    out = connect[-1][1] #default to last lane for edge case or case when there is only one possible connection 
    for i in range(len(connect)-1):
        if pos < connect[i+1][0]:
            out = connect[i][1]
            break
    return out 

#number of timesteps and timestep length
simlen = 2000 
dt = .25
nveh = 100 #number of vehicles
#seed initial disturbance/create 'network'
timeseries = [30 for i in range(simlen)]
timeseries[:200] = [1/10*(.1*i-10)**2 +20 for i in range(200)]
timeseries = np.asarray(timeseries) - 10
#speedfun = timeseries_wrapper(timeseries)
#curlane = lane(speedfun)
curlane = lane(timeseries)
#only parameters needed are for cf call
p =  [33.33, 1.1, 2, .9, 1.5]
initspeed = timeseries[0]
length = 2
# build simulation
vehicles = set()
curpos = 0
veh = vehicle(-1, curlane, p, None, curpos, initspeed, None, 0, length = length, cfmodel = IDM, eqlfun = IDM_eql)
#eql_hd = veh.get_eql(initspeed)
eql_hd = IDM_eql(p, initspeed) 
vehicles.add(veh)
vehlead = veh
for i in range(nveh -1):
    curpos += -eql_hd - length
    veh = vehicle(i, curlane, p, None, curpos, initspeed, eql_hd, 0, length = length, cfmodel = IDM, eqlfun = IDM_eql, lead = vehlead)
    vehicles.add(veh)
    vehlead = veh
    
def test_cf(vehicles, simlen, dt):
    for i in range(simlen):
        for veh in vehicles: 
            veh.call_cf( i, dt)
        
        for veh in vehicles: 
            veh.update(i, dt)
        for veh in vehicles: 
            if veh.lead is not None: 
                veh.hd = veh.lane.get_headway(veh, veh.lead)
#%%
                
test_cf(vehicles, simlen, dt)
plt.figure()
for veh in vehicles: 
    plt.plot(veh.posmem)
    



