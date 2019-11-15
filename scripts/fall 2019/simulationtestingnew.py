
"""
@author: rlk268@cornell.edu

test the new shiny simulation module 
"""

import havsim
from havsim.simulation.simulation import *
from havsim.simulation.models import IDM_b3, IDM_b3_eql, sv_obj, IDM_b3_sh
import matplotlib.pyplot as plt 
import scipy.optimize as sc
#%%
#create initial conditions of stop and go traffic on circular road 
#p = [33.33, 1.5, 2, 1.1, 1.5] #4th parameter can change between .9 for absolutely unstable, 1.1 for convectively unstable. 
p = [33.33, 1.2, 2, 1.1, 1.5]
initstate, auxinfo, roadinfo = eq_circular(p, IDM_b3, IDM_b3_eql, 41, length = 2, L = None, v = 15, perturb = 1)
sim, curstate, auxinfo = simulate_cir(initstate, auxinfo,roadinfo, update_cir, timesteps = 25000, dt = .25)


#%%
p2 = [33.33, 1.2, 2, 1.1, 1.5, 15] #IDK just guess some parameters for IDM
vlist = {i: curstate[i][1] for i in curstate.keys()}
indlist = [min(vlist, key=vlist.get)]
#verify objective function works - also get baseline of original scenario
obj, testsim, curstate2, auxinfo2, modelinfo2 = simcir_obj(p,curstate, auxinfo, roadinfo,indlist, IDM_b3, sv_obj, timesteps = 5000, objonly = False, dt = .25)
#optimize
args = (curstate, auxinfo, roadinfo,indlist, IDM_b3_sh, sv_obj, update_cir, 5000, .25, True)
bounds = [(15,35), (.5, 1.8), (1, 3), (.5, 3), (.5, 3), (10, 20)]
res = sc.minimize(simcir_obj, p2, args = args, method = 'l-bfgs-b', bounds = bounds)
#%%

objopt, testsim, curstate2, auxinfo2, modelinfo2 = simcir_obj(res['x'],curstate, auxinfo, roadinfo,indlist, IDM_b3_sh, sv_obj, timesteps = 5000, objonly = False, dt = .25)

#%%
def plothelper(sim, cur = 40):
    plt.close('all')
    x, v, s = [], [], []
    for i in range(len(sim[cur])):
        x.append(sim[cur][i][0])
        v.append(sim[cur][i][1])
        s.append(sim[cur][i][2])
    plt.figure()
    plt.plot(x)
    plt.figure()
    plt.plot(v)
    plt.figure()
    plt.plot(s)
    
plothelper(testsim, cur = 21)

#def headwayhelper(sim, auxinfo, L, cur = 41):
#    slist = []
#    for j in range(len(sim[cur])): #penality for collisions
#        lead = auxinfo[cur][1]
#        leadx = sim[lead][j][0]
#        leadlen = auxinfo[lead][0]
#        s = leadx - leadlen - sim[cur][j][0]
#        slist.append(s)
#    plt.figure()
#    plt.plot(slist)
#    return 
#headwayhelper(sim,auxinfo, cur = 7)

#%%
def v_metric(testsim):
    v = []
    for i in testsim.keys(): 
        for j in testsim[i]:
            v.append(j[1])
    print(np.mean(v))
    return
v_metric(testsim)
v_metric(sim)