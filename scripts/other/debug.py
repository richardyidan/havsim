# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 20:31:23 2019

@author: rlk268
"""
#import jax.numpy as jnp 
#import pickle
#with open('dataautodiff.pkl','rb') as f:
#    meas,platooninfo = pickle.load(f)
#meas2 = {}
#for i in meas.keys():
#    meas2[i] = jnp.asarray(meas[i])
#    
#with open('datauatodiff2.pkl','wb') as f: 
#    pickle.dump([meas2,platooninfo],f)
    

#from calibration import interp1ds, binaryint
#test = [1,2,3,4,5]
#test2 = [2,3,4,5,6]
#test3 = [3.25,4,4.75]
#out = interp1ds(test,test2,test3)

#eql(IDM_b3,30+speedoffset,p, length)


#egobj([10,1.6,2,1.3,1.5],*args)
#data = []
#for i in universe: 
#    data.append(i.dx)
#with open('simeg33.33-1.5-2-1.1-1.5IDM_b3.pkl','wb') as f:
#    pickle.dump([data],f)

#import havsim.calibration.helper as helper
#platoon4 = [995, 998, 1013, 1023]
#testplatoon2 = [platoon4, [956]]
#cir = helper.cirdep_metric(testplatoon2, platooninfo, k=0.9, type='num', meas = meas)

#%%
#cleaning some stuff up for qiwu
#import numpy as np 
#import pickle 
#
#testdata = np.random.rand(50) #note jax handles random numbers in a special way but here we are just treating this as some arbitrary data so it should be fine
#testdata2 = {}
#for i in range(6):
#    testdata2[i] = np.random.rand(50)
#
#testdata3 = [[1,5,10],[2,10,15],[5,15,25]]
#testdata4 = [25,5]
#testdata5 = 20
#
#x1 = [1]  #eg1
#x2 = np.arange(1,10, dtype = np.float) #eg2
#p = 2*np.random.rand(20)+.5
#pfinal = [10,2,3]
#p7 = [2,1,15]
#
#with open('autodiffeg.pkl','wb') as f:
#    pickle.dump([x1,x2,p,pfinal, testdata,testdata2,testdata3,testdata4,testdata5,X,Y,times,p7], f)

#%%
#for i in list(meas.keys()): 
#    cur = helper.makeleadinfo([i],platooninfo,meas)
#    temp = [j[0] for j in cur[0]]
#    if len(np.unique(temp)) < len(temp):
#        print(i)

#meas, platooninfo,platoons = makeplatoonlist(data,n=5)
#helper.c_metric(384,platoons[0],T,platooninfo,type='follower')

#%%
from havsim.calibration.algs import makedepgraph, oldmakedepgraph
totfollist = [672.0, 503.0, 582.0, 423.0, 455.0, 518.0, 435.0, 533.0, 470.0, 439.0, 413.0, 511.0]
leaders = [492.0, 521.0, 460.0, 380.0, 420.0, 409.0, 400.0, 411.0, 406.0, 394.0, 312.0, 657.0, 613.0, 384.0]

Glen = math.inf
Gedge= math.inf
for i in totfollist:
    curG, depth = makedepgraph([i],leaders,platooninfo,math.inf)
    if len(curG.nodes()) < Glen or len(curG.edges())<Gedge:
        G = curG
        print('best new G is '+str(i)+' source with '+str(len(G.nodes()))+' nodes')
        
oldG = oldmakedepgraph(totfollist,leaders,platooninfo)