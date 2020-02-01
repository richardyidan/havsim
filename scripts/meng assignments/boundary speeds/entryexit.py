
"""
@author: rlk268@cornell.edu
"""

import pickle
import numpy as np 
import copy 
from havsim.calibration.algs import makeplatoonlist
from havsim.plotting import platoonplot, animatetraj


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
platoon  = [875.0, 903.0, 908.0] #dataset only with these vehicles 
toymeas = {}
for i in platoon: 
    toymeas[i] = meas[i].copy()
toymeas[875][:,4] = 0
toymeas[908][:,5] = 0
#plot all the vehicles, you can click on lines to see which vehicle IDs the trajectories correspond to
platoonplot(meas,None,platooninfo,platoon = platoon,colorcode=False,lane = 3)
#you can view the vehicle trajectories in this animation if you want
ani = animatetraj(meas,platooninfo,platoon = platoon)

#expected output: 

#for lane 4 - note the first 3 observations of vehicle 875 are in lane 4, so these 3 observations define the entry/exit speeds 
#for lane 4, which are defined only for those 3 times. 

#for lane 3 - 
#exit speeds will be vehicle 875 starting from its third observation, vehicle 903 from 3083 - 3104, vehicle 908 from 3105 - 3139
#entry speeds are going to be the speeds of vehicle 875 from times 2555 - 2573, 
#speeds of vehicle 903 from times 2574 - 2620, vehicle 908 from 2621 - 2660

#please verify what I've written above is correct and manually form the output for testing purposes 
#hint: you can figure out what the times are supposed to be since we know 875 defines the exit speeds until it's last observation, 
#then as soon as 875 last observation time is passed, 903 takes over the exit speeds, which continues until its last observation, etc. 

"""
Record what the output should be for boundaryspeeds(toymeas,[3,4],[3,4],.1,.1)
"""

#%%
#call signature for solution 

def boundaryspeeds(meas,entrylanes,exitlanes,timeind,outtimeind):

    return entryspeeds, entrytimes,exitspeeds, exittimes 
#%%
    #actual use cases
boundaryspeeds(meas, [2,3,4,5,6,7], [2,3,4,5,6], .1, .25)