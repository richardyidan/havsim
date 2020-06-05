
"""
Bottleneck simulation
"""
from havsim.simulation.simulation import vehicle, lane, downstream_wrapper, anchor_vehicle, simulation
from havsim.calibration.helper import boundaryspeeds, getentryflows
from havsim.plotting import calculateflows
import numpy as np
import matplotlib.pyplot as plt
from havsim.simulation.models import IDM_parameters
#%%get boundary conditions (careful with units)
# #option 1 -
# #could get them directly from data
# entryflows, unused = getentryflows(meas, [3],.1,.25)
# unused, unused, exitspeeds, unused = boundaryspeeds(meas, [], [3],.1,.1)

# #option 2 - use calculateflows, which has some aggregation in it and uses a different method to compute flows
# q,k = calculateflows(meas, [[200,600],[1000,1400]], [0, 9900], 30*10, lane = 6)

# #option 3 - can also just make boudnary conditions based on what the FD looks like
# tempveh = vehicle(-1, None, [33, 1.2, 2, 1.1, 1.5], None)
# spds = np.arange(0,33,.01)
# flows = np.array([tempveh.get_flow(i) for i in spds])
# density = np.divide(flows,spds)
# plt.plot(density,flows)

#%%

#define boundary conditions
def onramp_parameters(*args):
    cf_p, lc_p  = IDM_parameters()
    kwargs = {'route':['main road', 'exit'], 'maxspeed': cf_p[0]-1e-6}
    return cf_p, lc_p, kwargs
def mainroad_parameters(*args):
    cf_p, lc_p  = IDM_parameters()
    kwargs = {'route':['exit'], 'maxspeed': cf_p[0]-1e-6}
    return cf_p, lc_p, kwargs
def onramp_inflow(timeind, *args):
    if timeind % 1000 > 800:
        return .15+np.random.rand()/25
    else:
        return .08 + np.random.rand()/25
def mainroad_inflow(*args):
    return .58
get_inflow1 = {'speed_fun':onramp_inflow}
get_inflow2 = {'speed_fun':mainroad_inflow}
increment_inflow = {'method': 'shifted'}
downstream1 ={'method':'free'}

#make road network with boundary conditions - want to make an api for this in the future
road = {'name': 'main road', 'len': 900, 'laneinds':2, 0: None, 1: None}
road['connect to'] = {'exit': (900, 'continue', (0,1), None, None)}
onramp = {'name': 'on ramp', 'len': 200, 'laneinds':1, 0: None}
onramp['connect to'] = {'main road': ((100,200), 'merge', 0, 'l', road)}
lane0 = lane(0,800, road, 0, downstream = downstream1, increment_inflow = increment_inflow, get_inflow = get_inflow2, new_vehicle = mainroad_parameters)
lane1 = lane(0,800, road, 1, downstream = downstream1, increment_inflow = increment_inflow, get_inflow = get_inflow2, new_vehicle = mainroad_parameters)
road[0] = lane0
road[1] = lane1
lane2 = lane(0,200,onramp,0, increment_inflow = increment_inflow, get_inflow = get_inflow1, new_vehicle = onramp_parameters)
downstream2 = {'method':'merge', 'merge_anchor_ind':0, 'target_lane': lane1, 'selflane':lane2}
lane2.call_downstream = downstream_wrapper(**downstream2).__get__(lane2, lane)
onramp[0] = lane2

#road 1 connect left/right and roadlen
roadlenmain = {'on ramp':400, 'main road':0}
lane0.roadlen = roadlenmain
lane1.roadlen = roadlenmain
lane0.connect_right = [(0, lane1)]
lane1.connect_left = [(0, lane0)]
lane1.connect_right.append((500,lane2))
lane1.connect_right.append((600,None))
#road 2 connect left/right and roadlen
roadlenonramp = {'main road':-400, 'on ramp':0}
lane2.roadlen = roadlenonramp
lane2.connect_left.append((100, lane1))
#anchors
lane0.anchor = anchor_vehicle(lane0, 0)
lane1.anchor = anchor_vehicle(lane1,0)
lane2.anchor = anchor_vehicle(lane2,0)
lane1.merge_anchors = [[lane1.anchor, 500]]
lane2.merge_anchors = [[lane2.anchor,100]]
#add lane events
lane0.events = [{'event':'exit', 'pos':800}]
lane1.events = [{'event':'update lr', 'left': None, 'right':'add','right anchor':0, 'pos':500}, {'event':'update lr', 'left':None, 'right':'remove','pos':600},
                {'event':'exit','pos':800}]
lane2.events = [{'event':'update lr', 'left':'add', 'left anchor':0, 'right': None, 'pos':100}]

#make simulation
merge_lanes = [lane1, lane2]
inflow_lanes = [lane0, lane1, lane2]
sim = simulation(inflow_lanes, merge_lanes)

#call
sim.simulate(3000)


