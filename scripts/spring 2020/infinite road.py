
"""
@author: rlk268@cornell.edu
"""
import numpy as np
import matplotlib.pyplot as plt
from havsim.simulation.simulation import vehicle
from havsim.simulation.models import IDM, IDM_eql

class lane:
    def __init__(self, speedfun):
#        self.call_downstream = downstream_wrapper(speed_fun = speedfun, method = 'speed')
        self.timeseries = speedfun
        self.road = None
    def get_headway(self, veh, lead):
        return lead.pos-veh.pos - lead.length
    
    def call_downstream(self, veh, timeind, dt):
        speed = self.timeseries[timeind]
        return (speed - veh.speed)/dt
    

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
veh = vehicle(-1, curlane, p, None, length = length, cfmodel = IDM, eqlfun = IDM_eql)
#eql_hd = veh.get_eql(veh, initspeed) - length #eql_hd = IDM_eql(p, initspeed) - length 
eql_hd = IDM_eql(p, initspeed) - length 
curpos = 0
veh.pos, veh.speed = curpos, initspeed
veh.posmem.append(curpos), veh.speedmem.append(initspeed)
vehicles.add(veh)
vehlead = veh
for i in range(nveh -1):
    curpos += -eql_hd
    veh = vehicle(i, curlane, p, None, length = length, cfmodel = IDM, eqlfun = IDM_eql)
    veh.pos, veh.speed = curpos, initspeed
    veh.posmem.append(curpos), veh.speedmem.append(initspeed)
    veh.lead = vehlead
    veh.hd = curlane.get_headway(veh,veh.lead)
    vehicles.add(veh)
    vehlead = veh
    
def test_cf(vehicles, simlen, dt):
    for i in range(simlen):
        for veh in vehicles: 
            veh.action = veh.call_cf(veh, veh.lead, veh.lane, i, dt, veh.in_relax)
        
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
    



