# -*- coding: utf-8 -*-
"""
\\TO DO\\ - clean up all tis explanation in green and the code which has some unused parts 

This file can load in the reconstructed, original, and re-extracted NGSim I-80 data. 
#first 15 minutes of the dataset, another 30 minutes are available but only in the original format, and not the reextracted or reconstructed. 
#data - reconstructed dataset, not that the study area is trimmed so that the length of each trajectory is shorter(~1400-1450 feet versus 1600)
#rawdata - original dataset
#truedata - reconstructed dataset all vehicles which are in the original dataset
#trueextradata - reconstructed dataset vehicles which are not in the original dataset


NOTE: the reconstructed data doesn't include the motorcycles. Those aren't included because they lack the leader/follower IDs. In the future, one can consider reobtaining all the 
leader/follower IDs, which would allow the motorcycles to be added again. That might also be useful for a few vehicles which have gaps in their lead IDs, which prevent them from being 
calibrated for the entire length of their observations. 
NOTE: YOU DO NOT NEED TO ACTUALLY RUN THE FUNCTION IF YOU ALREADY HAVE MYDATA.PKL.  
It takes some time to load in all the data, and get it in the correct format (np.array) and fix a few inconsistencies in data
this loads in the raw, reconstructed, and actual NGSim data. the reextracted data is in a different format, and only for cam 6: see the below
--------------------------------------------------------------------------------------------------------------
DATA REQUIREMENTS 
[3,4,9,8,6,2,5] for reconstructed. [5,11,14,15,8,13,12] for raw/reextracted
motorcycles can sometimes be a problem: they have IDs: 
[12,292,564, 644, 1093, 1370, 1376, 1389, 1526, 1881, 2323, 2945, 3001, 3178, 3349 ]
the rawdata has all of those. the reextracteddata is missing 5 of the motorcycles, so they only have: 
[292,564,644,1093,1376,1526,1881,2323,2945,3349]

need vehicle ID, frame ID (i.e. time), position, speed (can be derived from position and time), acceleration (can be derived from position and time),
lane ID (number to indicate which lane a vehicle is in), lead vehicle ID (can be derived by position, time and lane ID), follower ID (can be derived by lead ID),
vehicle length. 

Also we require all the data to be sorted so that all the vehicle IDs are together, and all the measurements are in sequential order, i.e. sorted in time. 
(so in other words, you need to sort the data according to both the vehicle ID and frame ID)

            0 - vehicle ID 
            1 - time
            2- position, 
            3 - speed, 
            4 - leader, 
            5 - follower. 
            6 - length, 
            7 - lane, 
            8 - acceleration


--------------------------------------------------------------------------------------------------------------------
reconstructed data (created by montanino and punzo)

0 - vehicle ID,
1 - frame ID,
2 - Lane ID.
3 - local y,
4 - mean speed,
5 - mean accel,
6 - vehicle length,
7 - vehicle class ID,
8 - follower ID,
9 - leader ID

the numerical differentiation is done using backward differences i.e. data[i+1,4] = (data[i+1,3]- data[i,3])/.1 and same for acceleration (for reconstructed data)
note the motorcycle data follower and leader ID are not given; we take those directly from the raw data 

Note that the reconstructed leader/follower ID have an issue where entries that should be '0' are instead '1'. This is fixed in the function call. 
Also we convert the units to be in feet. 
--------------------------------------------------------------------------------------------------------------------
raw data (feet / second) (created by US DOT)

0 -vehicle id
1 - frame id
2 - total frames,
3 - global time,
4  - local x,
5 - local y
6 - global x,
7 - global y,
8 - v length,
9 -  v width,
10 - v class, (1 - motorcycle, 2 - auto, 3 - truck)
11 - v vel,
12 - v acc,
13 -  lane id, (1 - leftmost, 6 - rightmost, 7 - onramp, 9 - right shoulder)
14 - preceding, (leader)
15 - following, (follower)
16 - space headway,
17 - time headway 
--------------------------------------------------------------------------------------------------------------------
re-extracted data (created by coifman and Li)
same format as the NGSim data, but missing several columns. It reextracts the REAR BUMPER! NOT THE FRONT BUMPER!
the function below will merge the raw and re-extracted data, preserving extra information from the original experiment but updating the positions with the
more accurate manually extracted positions. The output of the function has the position of the front bumper according to coifman and li. 

There is some code at the end of the script that can combine trueextradata with the other sources of data. This can potentially be useful as you can use trueextradata
as leaders, thus allowing you to simulate (and calibrate) more of the vehicles in the raw/reconstructed/reextracted datasets. Note that the vehicles in trueextradata cannot
be fully simulated as their lengths are unknown. However, their lanes and their leader/follower IDs are known. 

VehID 4000-4999 are observed before the NGSim platoon, 5000-5999 are observed after the NGSim platoon. 
For my purposes, we need the length of the lead vehicle to simulate the follower, and coifman and li did not reextract the length of the vehicle. 
However, since they extracted the rear bumper, we can actually use vehID 4000-4999 as extra lead vehicles, allowing us to potentially simulate
more of the NGsim vehicles. The bottom of this script will append vehicles 4000-4999 to the data. note that these vehicles cannot be simulated, they can ONLY be used as leaders.
It is easy to flag these vehicles as being special, since they have a length of 0. 

Note that although the measurements are very accurate, they are only extracted for camera 6 as of yet (in the future, perhaps all cameras will be extracted)
this means that the re-extracted data only details the movement over about 250 feet of road, whereas the raw data has movement over about 1600 feet of road. 
the reconstructed data truncates the raw data a bit, so that it details the movement on about 1200-1300 feet of road. 

The re-extracted data is differentiated between vehicles in the original experiment, and extra vehicles that were not extracted in the original (i.e. rawdata) experiment
what I call truedata (called truedata2 inside the function) contains vehicles only in the original experiment. 
what is returned as trueextradata are those vehicles (ID > 3999) which were not extracted in the original experiment (and are therefore not in the reconstructed data either)
inside the function, truedata contains all of the re-extracted data.

---------------------------------------------------------------------------------------------------------------------

It's not clear how the numerical differentiation is done for the raw ngsim data. but its close to the above (backward differences) most of the time. 
in reextracted data, only local y, v vel, v acc, lane id, preceding, and following are affected
it's not clear how the numerical differentiation is done for the reextracted ngsim data either. so you might want to make the numerical differentiation for reextracted and raw data the same as for the reconstructed data 

#reextracted data is missing 5 motorcycles, so it has 2047 vehicles. raw/reconstructed data has 2052 vehicles. raw data has longer local y ranges, and therefore has significantly more rows. 

Takes in no inputs (assume the data has the original names, and are in current directory)
outputs %%%%%%%%%%%%%%%%%%%%%%%%%%
 rawdata, truedata, data, extradata - raw, reextracted, reconstructed, and extra reextracted data, respectively 
 reextracted data reextracts rear bumper rather than front bumper 
 --------------------------------------------------------------
 
 there are arguably some weird parts of the data, like how the leaders and followers are determined (some seeming inconsistencies e.g. use makefollowerchain and look at followerchain output. 
 in makefollowerchain, where we don't use lane changing, the T_n for the lead vehicle should be the T_nm1 for the following vehicle. But this isn't always the case? Why? Well, apparently because
 the data says that the follower no longer has a leader vehicle, even though that function will only look at vehicles in the same lane. So really there is a leader.
 
Another thing that is arguably impercise is how which lane a vehicle is in is determined. e.g. at what frame should I be switched to lane 2 from lane 3? That is a choice. Should it be when I stop moving in 
x position? Should it be when we pass the halfway point? Should it say I changed lanes when I start moving to the other lane? Those choices impact the calibrated dynamics. 
Trying to clean the data is pretty out of the scope of this package, but these are things one should keep in mind. Calibration results are very dependent 
on the data you feed into it (and this is something that we perhaps don't think about as much as we should).
"""


#first load in the raw NGSim data from text format 
#when loading in a raw text format, assume that the values are padded with spaces and that we know the number of entries in each row of the data


import numpy as np 
import re
import csv 
import pickle
import copy 

from ..havsim.calibration.opt import makeplatoonlist



def loadngsim():
    
    #in general this will load in any .txt format data assuming we know the number of entries in each row
    numentries = 18
    rawdata = []
    #text = 'asdfaff'
    file = open('trajectories-0400-0415.txt', 'r') #raw data from fhwa
    
    for line in file: 
        for s in re.findall(r'\d+(?:\.\d+)?',line):  #find all ints and floats in the string
            rawdata.append(float(s)) #append eveyrthing into a long array as floats
    
    l = len(rawdata)
    rawdata = np.asarray(rawdata)
    rawdata = np.reshape(rawdata, (int(l/numentries),numentries))
    #############################################################################
    
    #next we will load in the actual data, which is in similar format as the raw data but needs to be combined with the raw data
    #in general this code will load in any .csv format data assuming there is no header to the csv (if there is, skip the header by uncommenting the next(readdata) lines) and we know the number of entries in each row 
    
    with open('Coifmandata.csv', newline='') as f:
        rows = 215243 #if you don't know this quantity you can use the commented code below to find it 
        readdata = csv.reader(f)
        numentries = 18
        next(readdata) #skip first row 
        
        #rows = sum(1 for i in readdata) #get number of rows #note you need to get rows before you do the above 
        #readdata = csv.reader(f)
        #next(readdata) #skip first row 
        
        truedata = np.zeros((rows,numentries))
        counter = 0
        for row in readdata: 
            truedata[counter,:] = [float(i) for i in row]
            counter+=1
    ############################################################################
    #now we will load in the reconstructed data 
    
    data = []
    numentries = 10
    file = open('DATA (NO MOTORCYCLES).txt', 'r') #reconstructed data from montanino and punzo, this has a different format than the above two 
    for line in file: 
        for s in re.findall(r'\d+(?:\.\d+)?',line):  #find all ints and floats in the string
            data.append(float(s)) #append eveyrthing into a long array as floats
            
#    file = open('MOTORCYCLES.txt', 'r') #reconstructed motorcycle data
#    for line in file: 
#        for s in re.findall(r'\d+(?:\.\d+)?',line):  #find all ints and floats in the string
#            data.append(float(s)) #append eveyrthing into a long array as floats      
    
    l = len(data)
    data = np.asarray(data)
    data = np.reshape(data, (int(l/numentries),numentries))
    for i in range(len(data)):
        data[i,3] = data[i,3]*3.28084 #convert from meters to feet
        data[i,4] = data[i,4]*3.28084 #convert from meters to feet
        data[i,5] = data[i,5]*3.28084 #convert from meters to feet
        data[i,6] = data[i,6]*3.28084
        #############################
    #just ignore the motorcycles since they mess things up 
#    with open('RECONSTRUCTED trajectories-400-0415_MOTORCYCLES.csv', newline='') as f:
#        rows = 3220 #if you don't know this quantity you can use the commented code below to find it 
#        readdata = csv.reader(f)
#        next(readdata) #skip first row 
#        
##        rows = sum(1 for i in readdata)
##        readdata = csv.reader(f)
##        next(readdata)
#        
#        motorcycledata = np.zeros((rows,8))
#        counter = 0
#        for row in readdata: 
#            motorcycledata[counter,:] = [float(i) for i in row]
#            counter+=1
#
#    ############################################################################
    #more stuff for the motorcycles we will ignore
#    #now we will find the leader and follower IDs for the motorcycles and put them into motorcycledata before merging the reconstructed motorcycle data with the reconstructed vehicle data 
#    rows = 3220
#    newmotorcycledata = np.zeros((rows,10))
#    motorID = np.unique(motorcycledata[:,0]) #unique motorcycle IDs
#    
#    counter =0
#    for i in motorID:
#        curdata = rawdata[rawdata[:,0]==i] #get all raw data for current motorcycle ID 
#        currecdata = motorcycledata[motorcycledata[:,0]==i] #get all reconstructed data for current motorcycle ID
#        curframes = currecdata[:,1] #all frames for reconstructed data 
#        
#        for j in range(len(curframes)): 
#            curframe = curframes[j] #current frame for current vehicle ID
#            currow = curdata[curdata[:,1]==curframe]
#            newmotorcycledata[counter,:] = np.append(currecdata[j,:],currow[0,[15,14]])
#            newmotorcycledata[counter,3] = newmotorcycledata[counter,3]*3.28084 #convert local y to feet...speed/accel/len already in feet for motorcycles 
#            counter += 1
#    
#    data = np.append(data, newmotorcycledata, axis = 0) #append the full motorcycle data with original vehicle data 
    

    
    ###############################################################################
    
    #we need to combine truedata and rawdata because of the format truedata is in 
    truedata2 = [] #true data for vehicles in original dataset
    trueextradata = [] #true data for vehicles not in original dataset
    #note that the reextracted vehicles not in original dataset don't include a vehicle length 
    
    rawunique, rawind, rawct = np.unique(rawdata[:,0],return_index=True,return_counts=True)
    trueunique, trueind,truect = np.unique(truedata[:,0],return_index=True,return_counts=True)
    maxraw = max(rawunique)
    
    for z in range(len(rawunique)): 
        i = rawunique[z]
        curdata = rawdata[rawind[z]:rawind[z]+rawct[z],:] #get all raw data for current vehicle ID
        t_nstar = curdata[0,1] #t_nstar for curdata
        try:
            curtrueind = list(trueunique).index(i) #truedata is missing vehicle 12 (motorcycle). It's also missing motorcycle 1370, 1389, 3001, 3178
        except: 
            continue
        curtruedata = truedata[trueind[curtrueind]:trueind[curtrueind]+truect[curtrueind],:] #all true data for current vehicle ID
        curframes = curtruedata[:,1]
        
        for j in range(len(curframes)): 
            curframe = curframes[j] #current frame for current vehicle ID
            
            currow = curdata[int(curframe-t_nstar),:].copy() #corresponding row of raw data
            
            currow[5] = curtruedata[j,5]+currow[8] #new position is reextracted position plus vehicle length 
            
            currow[[11,12,13,14,15]] = curtruedata[j,[11,12,13,14,15]].copy() #load in other reextracted quantities
            
            for k in currow: 
                truedata2.append(k)
                
        #reshape all entries into appropriately sized matrix
                
    l = len(truedata2)
    #print(l)
    truedata2 = np.asarray(truedata2)
    truedata2 = truedata2.reshape((int(l/18),18))
    
    #if want to look at the new extrated vehicles (note that the length of the vehicle is unknown for these)
    
    for i in range(len(truedata)):
        if truedata[i,0] <=maxraw: 
            continue
        else: 
            trueextradata = np.append(trueextradata,truedata[i,:])
        
    l = len(trueextradata)
    #print(l)
    trueextradata = np.asarray(trueextradata)
    trueextradata = trueextradata.reshape((int(l/18),18))

    
    
    
    #now we will fix a problem with leader/follower ID for reconstructed data 
    #problem with reconstructed data where montanino gave an ID of 1 to indicate no leader/follower. This is a problem because there is a vehicle ID 1. Therefore 
    #the value of 1 should be interpreted as vehicle ID 1. However, most of the time the value of 1 in the reconstructed data actually means no follower (ID should be 0)
    #if montanino says 1, raw says 0, we have to assume that montanino means no leader/follower, and change the 1 to  a zero
    #if montanino says not 1, and raw says 0, we can assume that the difference in vehicle ID is caused by the reconstructed data fixing some issues in the raw data. so do nothing.
    
    #start the new code which will fix the problem
    for i in range(len(data)):
        if data[i,9]==1: #if it says 1 change it to zero
            data[i,9]=0
        if data[i,8]==1:
            data[i,8]=0
    #will make it so vehicle 1 can act as a leader
    curveh = data[data[:,0]==1] #get vehicle 1; could be generalized with an outer loop here if you wanted to do multiple vehicles 
    fol = np.unique(curveh[:,8]) #unique followers of curveh
    for i in fol: 
        if i==0:
            continue
        curfol = data[data[:,0]==i]
        mytimes = curveh[:,1]
        for j in range(len(mytimes)):
            if curveh[j,8]==i: #if the curveh has i as a follower
                temp = curfol[curfol[:,1]==mytimes[j]] #grab time of the curfol we want; need to use dummy variable temp because of numpy
                temp[0,9]=1 #then the curfol should have the curveh as a leader at the same time
                curfol[curfol[:,1]==mytimes[j]] = temp #overwrite old value of curfol
        
        data[data[:,0]==i] = curfol #overwrite old data values for the vehicle 
        
        
    #vehicle 564 has a problem because it's a motorcycle. The below code fixes it for 564 but we will actually just ignore all the motorcycles 
#    curveh = data[data[:,0]== 564]
#    t_nstar = curveh[0,1]
#    curveh[2163-int(t_nstar):,9] = 0
#    curveh[:,8] = 0 
#    data[data[:,0]==564] = curveh
     
    
    #in the rawdata/truedata there are some vehicles with IDs greater than 4000; those we actualy don't want 
    #there are also some IDs of 999. Those we should get rid of as well. 
    for i in range(len(rawdata)):
        if rawdata[i,14]>=4000:
            rawdata[i,14] = 0
        if rawdata[i,14]==999:
            rawdata[i,14] = 0 
        if rawdata[i,15]==999:
            rawdata[i,15] = 0 
        if rawdata[i,15]>=4000:
            rawdata[i,15] = 0 
            
    for i in range(len(truedata2)):
        if truedata2[i,14]>=4000:
            truedata2[i,14] = 0
        if truedata2[i,15]>=4000:
            truedata2[i,15] = 0 
        if truedata2[i,14]==999:
            truedata2[i,14] = 0 
        if truedata2[i,15]==999:
            truedata2[i,15] = 0 
            
    
    ##########################################################################################################
        
    
    #redifferentiate data
    data = re_differentiate(data)
    #other datasets seem to just get more loopy when you redifferentiate
#    rawdata = re_differentiate(rawdata, [5,11,14,15,8,13,12])
#    truedata2 = re_differentiate(truedata2, [5,11,14,15,8,13,12]) #we won't redifferentiate the true data because stuff looks weird if we do that 
#    trueextradata = re_differentiate(trueextradata, [5,11,14,15,8,13,12])
    
    #reorder data
    data = data[:,[0,1,3,4,9,8,6,2,5]]
    truedata2 = truedata2[:,[0,1,5,11,14,15,8,13,12]]
    rawdata = rawdata[:,[0,1,5,11,14,15,8,13,12]]
    trueextradata = trueextradata[:,[0,1,5,11,14,15,8,13,12]]
    return rawdata, truedata2, data, trueextradata




def re_differentiate(data,dataind=[3,4,9,8,6,2,5]):
    #redifferentiates the data according to forward difference scheme
    vehlist,vehind,vehct = np.unique(data[:,0], return_index=True, return_counts=True)
    for z in range(len(vehlist)): #reconstructed data first 
        curveh = data[vehind[z]:vehind[z]+vehct[z],:]
        curlen = len(curveh)
        curveh[0,dataind[1]] = (curveh[1,dataind[0]] - curveh[0,dataind[0]])/.1 #difference first speed 
        for j in range(curlen-2): #need to put a minus 2 here. i.e. last two acceleration, last speed not differenced. 
            curveh[j+1,dataind[1]] = (curveh[j+2,dataind[0]] - curveh[j+1,dataind[0]])/.1 #difference next speed so now we can difference acceleration as well 
            curveh[j,dataind[-1]] = (curveh[j+1,dataind[1]] - curveh[j,dataind[1]])/.1 #difference acceleration 
#            if curveh[j,1]+1 != curveh[j+1,1]:
#                print('warning: data missing observation')
    return data



#either save or load the data 
       
#
#rawdata, truedata, data, trueextradata = loadngsim() #execute above function 
##################################################################################

#with open('mydata.pkl','wb') as f:
#    pickle.dump([rawdata, truedata, data, trueextradata], f) #save data 

with open('mydata.pkl', 'rb') as f:
    rawdata, truedata, data, trueextradata = pickle.load(f) #load data 
    
####################################################################################
    
#code to put trueextradata into truedata2 and rawdata. Can automatically flag trueextra vehicles as leaders by using the fact that they have 0 length. (length not reextracted)    


#trueextradata = trueextradata[trueextradata[:,0]<5000] #only get the vehicles between 4000-4999 because those are the ones we can use as leaders
#
#truedata = np.append(truedata, trueextradata, axis =0) #append trueextradata to truedata
#
#data = np.append(data, trueextradata[:,[0,1,13,5,11,12,8,10,15,14]], axis=0) #append trueextradata to reconstructed data. note trueextradata is only over cam6 range (~1250-1570 local y)
#
#rawdata = np.append(rawdata,trueextradata,axis=0) #append trueextradata to raw data 



meas, platooninfo = makeplatoonlist(data,1,False)