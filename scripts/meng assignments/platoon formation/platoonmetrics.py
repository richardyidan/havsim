
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
    
#%%
#note that havsim.calibration.helper.makeleadfolinfo can be used to get the leaders 
#for a platoon, which may be convenient. 

from havsim.calibration.helper import makeleadfolinfo

testplatoon = [381.0, 391.0, 335.0, 326.0, 334.0]
leadinfo, folinfo, unused = makeleadfolinfo(testplatoon, platooninfo, meas)

#leadinfo[2] = [[316.0, 1302, 1616], [318.0, 1617, 1644]]
#this means second vehicle in the platoon (testplatoon[1], which is 391)
#follows vehicle 316 from time 1302 to 1616, and it follows 318 from 1617 to 1644. 
#folinfo has the same information but for followers instead of leaders. 

"""
TO DO 
Implement functions which calculate the metrics for a given platoon
Calculate manually what the chain metric should be for the platoon [[], 391, 335, 326] for k = 1 and k = 0. Show your work. 
"""