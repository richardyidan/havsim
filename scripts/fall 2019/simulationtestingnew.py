
"""
@author: rlk268@cornell.edu

test the new shiny simulation module 
"""

import havsim
from havsim.simulation.simulation import *
from havsim.simulation.models import IDM_b3, IDM_b3_eql
import matplotlib.pyplot as plt 


#p = [33.33, 1.5, 2, 1.1, 1.5] #4th parameter can change between .9 for absolutely unstable, 1.1 for convectively unstable. 
p = [33.33, 1.2, 2, 1.1, 1.5]
length = 2
initstate, auxinfo, modelinfo, L = eq_circular(p, length, IDM_b3, IDM_b3_eql, 41, L = None, v = 15, perturb = 0)
#change velocity of all vehicles so we're further from equilibrium
for i in initstate.keys():
    initstate[i][1] += 2

sim, curstate, auxinfo, modelinfo = simulate_cir(initstate, auxinfo,modelinfo, L, 2000, .25)

#%%
p2 = [33.33, .5, 2, 1.5, 1.5] #IDK just guess some parameters for IDM
obj, testsim, curstate2, auxinfo2, modelinfo2 = simcir_obj(p2,curstate, auxinfo, modelinfo, L, 5000, [8], IDM_b3, sv_obj, objonly = False, dt = .25)


#%%
def plothelper(sim, cur = 41):
#    plt.close('all')
    x, v = [], []
    for i in range(len(sim[cur])):
        x.append(sim[cur][i][0])
        v.append(sim[cur][i][1])
    plt.figure()
    plt.plot(x)
    plt.figure()
    plt.plot(v)
    
plothelper(sim, cur = 7)

def headwayhelper(sim, auxinfo, L, cur = 41):
    slist = []
    for j in range(len(sim[cur])): #penality for collisions
        lead = auxinfo[cur][1]
        leadx = sim[lead][j][0]
        leadlen = auxinfo[lead][0]
        s = leadx - leadlen - sim[cur][j][0]
        slist.append(s)
    plt.figure()
    plt.plot(slist)
    return 
headwayhelper(sim,auxinfo, cur = 7)