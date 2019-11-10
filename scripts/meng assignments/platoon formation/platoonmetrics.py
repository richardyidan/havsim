
"""
@author: rlk268@cornell.edu

create metric for evaluating how good platoons are 
"""
import pickle 
import numpy as np 
from havsim.calibration.algs import makeplatoonlist, makeplatoonlist_s

#load data 
with open('C:/Users/rlk268/OneDrive - Cornell University/important misc/datasets/trajectory data/mydata.pkl', 'rb') as f: #replace with path to pickle file 
    rawdata, truedata, data, trueextradata = pickle.load(f) #load data 

#%%
    
#existing platoon formation algorithm
meas, platooninfo, platoons = makeplatoonlist(data, n = 5)
#existing platoon formation algorithm in a single lane 
unused, unused, laneplatoons = makeplatoonlist(data,n=5,lane=2,vehs=[582,1146])
#platoon formation based on sorting    
unused, unused, sortedplatoons = makeplatoonlist_s(data,n=5,lane=6)
    

