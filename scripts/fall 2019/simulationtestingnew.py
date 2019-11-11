
"""
@author: rlk268@cornell.edu

test the new shiny simulation module 
"""

import havsim
from havsim.simulation.simulation import eq_circular, simulate_cir
from havsim.simulation.models import IDM_b3, IDM_b3_eql
import matplotlib.pyplot as plt 


#p = [33.33, 1.5, 2, 1.1, 1.5] #4th parameter can change between .9 for absolutely unstable, 1.1 for convectively unstable. 
p = [33.33, 1.2, 2, 1.1, 1.5]
length = 2
initstate, auxinfo, modelinfo, L = eq_circular(p, length, IDM_b3, IDM_b3_eql, 41, L = None, v = 15)
#change velocity of all vehicles so we're further from equilibrium
for i in initstate.keys():
    initstate[i][1] += 2

sim, curstate, auxinfo, modelinfo = simulate_cir(initstate, auxinfo,modelinfo, L, 25000, .25)

#%%
plt.close('all')
x, v, cur = [], [], 8
for i in range(len(sim[cur])):
    x.append(sim[cur][i][0])
    v.append(sim[cur][i][1])
plt.plot(x)
plt.figure()
plt.plot(v)