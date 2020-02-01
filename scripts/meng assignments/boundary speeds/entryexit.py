
"""
@author: rlk268@cornell.edu
"""

import pickle
import numpy as np 
import copy 
from havsim.calibration.algs import makeplatoonlist
from havsim.plotting import platoonplot


#ngsim data 
with open('C:/Users/rlk268\OneDrive - Cornell University/important misc/pickle files/meng/reconngsim.pkl', 'rb') as f: 
    data = pickle.load(f)
    
#highd data 
with open('C:/Users/rlk268\OneDrive - Cornell University/important misc/pickle files/meng/highd26.pkl', 'rb') as f: 
    highd = pickle.load(f)
    

meas, platooninfo = makeplatoonlist(data,1,False) #form meas for ngsim data
#note time discretization = .1 seconds for ngsim, .04 seconds for highd

#%% toy example 
#create toy example as just a small portion of the full data 
platoon  = [875.0, 903.0, 908.0, 913.0, 922.0] #dataset only with these vehicles 
toymeas = {}
for i in platoon: 
    toymeas[i] = meas[i].copy()
toymeas[875][:,4] = 0
toymeas[922][:,5] = 0
#plot all the vehicles, you can click on lines to see which vehicle IDs the trajectories correspond to
platoonplot(meas,None,platooninfo,platoon = platoon,colorcode=False,lane = 3)

#expected output: 
#exit speeds are just the full trajectory of vehicle 875, the first 3 observations of 875 are actually in a different lane, 
#entry speeds 


#%%
#call signature for solution 

def boundaryspeeds(meas,entrylanes,exitlanes,timeind,outtimeind):
    
    
    return entryspeeds, entrytimes,exitspeeds, exittimes 

