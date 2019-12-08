
"""
helper functions for calibration module; these functions don't implement any core functionality; they simply need to 'work'

@author: rlk268@cornell.edu
"""
import numpy as np 
import heapq
import math
from collections import defaultdict

def checksequential(data, dataind = 1, pickfirst = False):	
#	checks that given data are all sequential in time (i.e. each row of data advances one frameID)
	#If the data are not sequential, it finds all different sequential periods, and returns the longest one
	#if pickfirst is true, it will always pick the first interval.
	#This function is used by both makeplatooninfo and makefollowerchain to check if we have a continuous period of having a leader. 
	#This essentially is using the np.nonzero function 
	#note - assumes the data is arranged so that higher row indices correspond to later times 
	#input-
	#    data: nparray, with quantity of interest in column dataind i.e. data[:,dataind]
	#    data is arranged so that data[:,dataind] has values that increase by 1 [0,1,2, ...] BUT THERE MAY BE JUMPS e.g. [1 3] or [1 10000] are jumps. [1 2 3 4] is sequential
	#    we "check the data is sequential" by finding the longest period with no jumps. return the sequential data, and also the indices for sequential data (indjumps)
	#    data[indjumps[0]:indjumps[1],:] would give the sequential data.  
	#    dataind: the column index of the data that we will be checking is sequential
	#    
	#    pickfirst = False: if True, we will always pick the first sequential period, even if it isn't the longest. Otherwise, we always pick the longest. 
	#    
	#
	#output - 
	#    sequentialdata: the sequential part of the data. i.e. the data with only the sequential part returned
	#    indjumps: the indices of the jumps
	#    note that indjumps[seqind]:indjumps[seqind+1] gives correct slice indexing. So the actual times indices are indjumps[seqind], indjumps[seqind+1]-1
	
	
    #returns data which is sequential and the slice indices used to give this modified data
    #note that indjumps[seqind]:indjumps[seqind+1] gives correct slice indexing. So the actual times indices are indjumps[seqind], indjumps[seqind+1]-1
    l = data.shape[0] #data must be an np array with rows as observations! cannot be a list/regular array or have columns as observations 
    
    if l ==0: #check if data is empty. if this happens we return the special value of [0,0] for indjumps. [0,0] cannot be returned any other way. 
        indjumps = [0,0]
        return data, indjumps 
    if (-data[0,dataind]+data[-1,dataind]+1) == l: #very quick check if data is totally sequential we can return it immediately 
        indjumps = [0,l]
        return data, indjumps
    if l <= 10: 
        print('warning: very short measurements') #this is just to get a feel for how many potentially weird simulated vehicles are in a platoon. 
        #if you get a bunch of print out it might be because the data has some issue or there may be a bug in other code
    timejumps = data[1:l,dataind] - data[0:l-1,dataind]
    timejumps = timejumps - 1 #timejumps is nonzero only if there is a gap in the datastream
    indjumps = np.nonzero(timejumps) #non-zero indices of timejumps
    lenmeas = np.append(indjumps, [l-1]) - np.insert(indjumps,0,-1) #array containing number of measurements in each sequential period
    seqind = np.argmin(-lenmeas) #gets index of longest sequential period
    indjumps = indjumps[0] + 1 #prepare indjumps so we can get the different time periods from it easily 
    indjumps = np.append(indjumps, l)
    indjumps = np.insert(indjumps,0,0)
    
    if pickfirst: #pick first = true always returns the first sequential period, regardless of length. defaults to false
        data = data[indjumps[0]:indjumps[1],:]
        return data, indjumps[[0, 1]]
        
    data = data[indjumps[seqind]:indjumps[seqind+1],:]     
    #i don't know why I return only indjumps with specific seqind instead of just all of indjumps. But that's how it is so I will leave it 
    return data, indjumps[[seqind, seqind+1]] 

def sequential(data, dataind = 1, slices_format = True):
    
    #returns indices where data is no longer sequential after that index. So for example for indjumps = [5,10,11] we have data[0:5+1], data[5+1:10+1], data[10+1:11+1], data[11+1:]
    #as being sequential 
    #if slices_format = True, then it returns the indjumps so that indjumps[i]:indjumps[i+1] gives the correct slices notation
    l = data.shape[0] #data must be an np array with rows as observations! cannot be a list/regular array or have columns as observations 
    
    if l ==0: #check if data is empty. if this happens we return the special value of [0,0] for indjumps. [0,0] cannot be returned any other way. 
        #this should probably return None instead 
        indjumps = [0,0]
        return indjumps 
    if (-data[0,dataind]+data[-1,dataind]+1) == l: #very quick check if data is totally sequential we can return it immediately 
        indjumps = [0,l]
        return indjumps
    
    timejumps = data[1:l,dataind] - data[0:l-1,dataind]
    timejumps = timejumps - 1 #timejumps is nonzero only if there is a gap in the datastream
    indjumps = np.nonzero(timejumps) #non-zero indices of timejumps
    
    
    if slices_format:
        indjumps = indjumps[0] + 1 #prepare indjumps so we can get the different time periods from it easily 
        indjumps = np.append(indjumps, l)
        indjumps = np.insert(indjumps,0,0)
    
    return indjumps

def indtotimes(indjumps,data, dataind = 1):
    #takes indjumps, data and returns the times corresponding to the boundaries of the sequential periods
    #there are len(indjumps) - 1 total blocks, and each block is characterized as a tuple of two values of the beginning and ending time + 1
    
    #if you want to go from times back to inds it's easy, first you subtract off the start time, then you add the starting indices of the start time (included as third in the tuple )
    out = []
    for i in range(len(indjumps)-1):
        starttime = data[indjumps[i],dataind]
        endtime = data[indjumps[i+1]-1,dataind]+1
        temp = (starttime, endtime, indjumps[i]) #indjumps[i] included because 
        out.append(temp)
    
    return out 

def interp1ds(X,Y,times):
    #given time series data X, Y (such that each (X[i], Y[i]) tuple is an observation), 
    #and the array times, interpolates the data onto times. 
    
    #X, Y, and times all need to be sorted in terms of increasing time. X and times need to have a constant time discretization
    #runtime is O(n+m) where X is len(n) and times is len(m)
    
    #uses 1d interpolation. This is similar to the functionality of scipy.interp1d. Name is interp1ds because it does linear interpolation in 1d (interp1d) on sorted data (s)
    
    #e.g. X = [1,2,3,4,5]
    #Y = [2,3,4,5,6]
    #times = [3.25,4,4.75]
    
    #out = [4.25,5,5.75]
    
    if times[0] < X[0] or times[-1] > X[-1]:
        print('Error: requested times are outside measurements')
        return None

    Xdt = X[1] - X[0]
    timesdt = times[1]-times[0]
    change = timesdt/Xdt
    
    m = binaryint(X,times[0])
    out = np.zeros(len(times))
    curind = m + (times[0]-X[m])/Xdt
    
    leftover = curind % 1
    out[0] = Y[m] + leftover*(Y[m+1]-Y[m])
    
    for i in range(len(times)-1):
        curind = curind + change #update index
        
        leftover = curind % 1 #leftover needed for interpolation
        ind = int(curind // 1) #cast to int because it is casted to float automatically 
        
        out[i+1] = Y[ind] + leftover*(Y[ind+1]-Y[ind])
    
    return out
    

def binaryint(X,time): 
    #finds index m such that the interval X[m], X[m+1] contains time. 
    #X = array 
    #time = float
    lo = 0 
    hi = len(X)-2
    m = (lo + hi) //  2
    while (hi - lo) > 1: 
        if time < X[m]:
            hi = m
        else: 
            lo = m
        m = (lo + hi) // 2
    return lo 


def makeleadinfo(platoon, platooninfo, sim, *args): 
    #gets ALL lead info for platoon 
    #requires platooninfo as input as well
    #leaders are computed ONLY OVER SIMULATED TIMES (t_n - T_nm1). 
    #requires platooninfo so it can know the simulated times. 
    #also requires sim to know the leader at each observation
    
    #EXAMPLE: 
#    platoon = [[],5,7] means we want to calibrate vehicles 5 and 7 in a platoon
#    
#    leadinfo = [[[1,1,10],[2,11,20]],[[5,10,500]]] Means that vehicle 5 has vehicle 1 as a leader from 1 to 10, 2 as a leader from 11 to 20. 
#    vehicle 7 has 3 as a leader from 10 to 500 (leadinfo[i] is the leader info for platoons[i].)
    leadinfo = []
    for i in platoon: #iterate over each vehicle in the platoon
        curleadinfo = [] #for each vehicle, we get these and then these are appeneded at the end so we have a list of the info for each vehicle in the platoon
        
        t_nstar, t_n, T_nm1, T_n = platooninfo[i][0:4] #get times for current vehicle 
        leadlist = sim[i][t_n-t_nstar:T_nm1-t_nstar+1,4] #this gets the leaders for each timestep of the current vehicle\
        curlead = leadlist[0] #initialize current leader
        curleadinfo.append([curlead, t_n]) #initialization 
        for j in range(len(leadlist)):
            if leadlist[j] != curlead: #if there is a new leader
                curlead = leadlist[j] #update the current leader
                curleadinfo[-1].append(t_n+j-1) #last time (in frameID) the old leader is observed
                curleadinfo.append([curlead,t_n+j]) #new leader and the first time (in frameID) it is observed. 
        curleadinfo[-1].append(t_n+len(leadlist)-1) #termination
        
        leadinfo.append(curleadinfo)
    
    return leadinfo

def makefolinfo(platoon, platooninfo, sim, *args, allfollowers = True, endtime = 'Tn'):
    #same as leadinfo but it gives followers instead of leaders. 
    #followers computed ONLY OVER SIMULATED TIMES + BOUNDARY CONDITION TIMES (t_n - T_nm1 + T_nm1 - T_n)
    #allfollowers = True -> gives all followers, even if they aren't in platoon
    #allfollowers = False -> only gives followers in the platoon (needed for adjoint calculation, adjoint variables depend on followers, not leaders.)
    #endtime = 'Tn' calculates followers between t_n, T_n, otherwise calculated between t_n, T_nm1, 
    #so give endtime = 'Tnm1' and it will not compute followers over boundary condition times 
    
    #EXAMPLE
    ##    platoons = [[],5,7] means we want to calibrate vehicles 5 and 7 in a platoon
##    allfollowers = False-
##    folinfo = [[[7,11,20]], [[]]] Means that vehicle 5 has vehicle 7 as a follower in the platoon from 11 to 20, and vehicle 7 has no followers IN THE PLATOON
    #all followers = True - 
#    [[[7,11,20]], [[8, 11, 15],[9,16,20]]] #vehicle 7 has 8 and 9 as followers 
    folinfo = []
    for i in platoon:
        curfolinfo = []
        t_nstar, t_n, T_nm1, T_n= platooninfo[i][0:4]
        if endtime == 'Tn':
            follist = sim[i][t_n-t_nstar:T_n-t_nstar+1,5] #list of followers
        else: 
            follist = sim[i][t_n-t_nstar:T_nm1-t_nstar+1,5]
        curfol = follist[0]
        unfinished = False
        
        if allfollowers and curfol != 0: 
            curfolinfo.append([curfol,t_n])
            unfinished = True
        else:
            if curfol in platoon: #if the current follower is in platoons we initialize
                curfolinfo.append([curfol,t_n])
                unfinished = True
            
        for j in range(len(follist)): #check what we just made to see if we need to put stuff in folinfo
            if follist[j] != curfol: #if there is a new follower
                curfol = follist[j]
                if unfinished: #if currrent follower entry is not finished
                    curfolinfo[-1].append(t_n+j-1) #we finish the interval
                    unfinished = False
                #check if we need to start a new fol entry
                if allfollowers and curfol != 0:
                    curfolinfo.append([curfol,t_n+j])
                    unfinished = True
                else:
                    if curfol in platoon: #if new follower is in platoons
                        curfolinfo.append([curfol,t_n+j]) #start the next interval
                        unfinished = True
        if unfinished: #if currrent follower entry is not finished
            curfolinfo[-1].append(t_n+len(follist)-1) #finish it 
        
        folinfo.append(curfolinfo)
        
        
        
    return folinfo

def makeleadfolinfo(platoons, platooninfo, sim, *args, relaxtype = 'both', mergertype = 'avg', merge_from_lane = 7, merge_lane = 6):
    #new makeleadfolinfo function which integrates the previous versions 
    #inputs - 
    #platoons : platoon you want to calibrate
    #platooninfo - output from makeplatooninfo
    #meas - measurements in usual format 
    #relaxtype = 'pos', 'neg', 'both', 'none'  - choose between positive, negative, and pos/negative relaxation amounts added. 'none' is no relax. 
    #mergertype = 'avg', 'last', 'none', 'remove'- 'avg' calculates the relaxation amount using average headway, 'last' uses the last known headway ; 'avg' works a lot better
    #if 'none' will not get merger relaxation amounts, but NOTE that some mergers are actually treated as lane changes and these are still kept. 
    #if 'remove' will actually go through and remove those mergers treated as lane changes (this happens when you had a leader in the on-ramp, and then merged before your leader)
    #merge_from_lane = 7 - if using merger anything other than 'none', you need to specify the laneID corresponding to the on-ramp
    #merge_lane = 6 - if using merger anything other than 'none' you need to specify the laneID you are merging into
    
    #outputs - 
    #leadinfo - list of lists with the relevant lead info (lists of triples leadID, starttime, endtime )
    #leadinfo lets you get the lead vehicle trajectory of a leader in a vectorized way. 
    #folinfo - same as leadinfo, but for followers instead of leaders. 
    #rinfo - gets times and relaxation amounts. Used for implementing relaxation phenomenon, which prevents
    #unrealistic behaviors due to lane changing, and improves lane changing dynamics 
 
##EXAMPLE: 
##    platoons = [[],5,7] means we want to calibrate vehicles 5 and 7 in a platoon
##    
##    leadinfo = [[[1,1,10],[2,11,20]],[[5,10,500]]] Means that vehicle 5 has vehicle 1 as a leader from 1 to 10, 2 as a leader from 11 to 20. 
##    vehicle 7 has 3 as a leader from 10 to 500 (leadinfo[i] is the leader info for platoons[i]. leadinfo[i] is a list of lists, so leadinfo is a list of lists of lists.)
##    
##    folinfo = [[[7,11,20]], [[]]] Means that vehicle 5 has vehicle 7 as a follower in the platoon from 11 to 20, and vehicle 7 has no followers in the platoon
    
    #legacy info-
    #makeleadfolinfo_r - 'pos' 'none'
    #makeleadfolinfo_r2 - 'neg', 'none'
    #makeleadfolinfo_r3 - 'both', 'none'
    #makeleadfolinfo_r4 - 'both', 'avg'
    #makeleadfolinfo_r5 - 'both', 'last'
    #makeleadfolinfo_r6 - 'both', 'remove'
    
    #in the original implementation everything was done at once which I guess saves some work but makes it harder to change/develop.
    #in this refactored version everything is modularized which is a lot nicer but slower. These functions are for calibration, and for calibration
    #all the time (>99.99%) is spent simulating, the time you spend doing makeleadfolinfo is neglible. Hence this design makes sense. 
    
    leadinfo = makeleadinfo(platoons, platooninfo, sim)
    folinfo = makefolinfo(platoons, platooninfo, sim, allfollowers = False)
    rinfo = makerinfo(platoons, platooninfo, sim, leadinfo, relaxtype = relaxtype, mergertype = mergertype, merge_from_lane = merge_from_lane, merge_lane = merge_lane)
    
    return leadinfo, folinfo, rinfo

def makerinfo(platoons, platooninfo, sim, leadinfo, relaxtype = 'both',mergertype = 'avg', merge_from_lane = 7, merge_lane = 6):
    
    if relaxtype =='none':
        return []
    
    rinfo = []
    for i in platoons: 
        currinfo = []
        t_nstar, t_n, T_nm1, T_n = platooninfo[i][0:4]
        leadlist = sim[i][t_n-t_nstar:T_nm1-t_nstar+1,4] #this gets the leaders for each timestep of the current vehicle\
        curlead = leadlist[0]
        
        for j in range(len(leadlist)):
            if leadlist[j] != curlead:
                newlead = leadlist[j]
                oldlead = curlead 
                
                #####relax constant calculation 
                newt_nstar = platooninfo[newlead][0]
                oldt_nstar = platooninfo[oldlead][0]
                olds = sim[oldlead][t_n+j-1-oldt_nstar,2] - sim[oldlead][0,6] - sim[i][t_n+j-1-t_nstar,2] #the time is t_n+j-1; this is the headway
                news = sim[newlead][t_n+j-newt_nstar,2] - sim[newlead][0,6] - sim[i][t_n+j-t_nstar,2] #the time is t_n+j
                ########
                
                #pos/neg relax amounts
                gam = olds - news
                if relaxtype == 'both':
                    currinfo.append([t_n+j,gam])
                    
                    if mergertype == 'remove': 
                        if sim[i][t_n+j-t_nstar,7]==merge_lane and sim[i][t_n+j-1-t_nstar,7]==merge_from_lane:
                            currinfo.pop(-1)
                elif relaxtype == 'pos':
                    if gam > 0:
                        currinfo.append([t_n+j,gam])
                        
                        if mergertype == 'remove': 
                            if sim[i][t_n+j-t_nstar,7]==merge_lane and sim[i][t_n+j-1-t_nstar,7]==merge_from_lane:
                                currinfo.pop(-1)
                elif relaxtype == 'neg': 
                    if gam < 0:
                        currinfo.append([t_n+j,gam])
                        
                        if mergertype == 'remove': 
                            if sim[i][t_n+j-t_nstar,7]==merge_lane and sim[i][t_n+j-1-t_nstar,7]==merge_from_lane:
                                currinfo.pop(-1)
                        
                curlead = newlead
        rinfo.append(currinfo)
    #merger cases
    if mergertype == 'avg':
        rinfo = merge_rconstant(platoons,platooninfo,sim,leadinfo,rinfo,200, merge_from_lane, merge_lane)
    elif mergertype == 'last':
        rinfo = merge_rconstant2(platoons,platooninfo,sim,leadinfo,rinfo,200, merge_from_lane, merge_lane)
                    

    return rinfo


def merge_rconstant(platoons, platooninfo, sim, leadinfo, rinfo, relax_constant = 100,merge_from_lane= 7,merge_lane = 6, datalen =9,h=.1):
    for i in range(len(platoons)):
        curveh = platoons[i]
        t_nstar, t_n, T_nm1, T_n = platooninfo[curveh][0:4]
        lanelist = np.unique(sim[curveh][:t_n-t_nstar,7])
        
        
        if merge_from_lane in lanelist and merge_lane not in lanelist and sim[curveh][t_n-t_nstar,7]==merge_lane: #get a merge constant when a vehicle's simulation starts when they enter the highway #
            lead = np.zeros((T_n+1-t_n,datalen)) #initialize the lead vehicle trajectory 
            for j in leadinfo[i]:
                curleadid = j[0] #current leader ID 
                leadt_nstar = int(sim[curleadid][0,1]) #t_nstar for the current lead, put into int 
                lead[j[1]-t_n:j[2]+1-t_n,:] = sim[curleadid][j[1]-leadt_nstar:j[2]+1-leadt_nstar,:] #get the lead trajectory from simulation
            headway =  lead[:,2]-sim[curveh][t_n-t_nstar:,2]-lead[:,6]
            headway = headway[:T_nm1+1-t_n]
            #calculate the average headway when not close to lane changing events 
            headwaybool = np.ones(len(headway),dtype=bool)
            for j in rinfo[i]:
                headwaybool[j[0]-t_n:j[0]-t_n+relax_constant]=0
            
            headway = headway[headwaybool]
            if len(headway)>0: #if there are any suitable headways we can use then do it 
                preheadway = np.mean(headway)
                
                postlead = sim[curveh][t_n-t_nstar,4]
                postleadt_nstar = platooninfo[postlead][0]
                
                posthd = sim[postlead][t_n-postleadt_nstar,2]-sim[postlead][t_n-postleadt_nstar,6]-sim[curveh][t_n-t_nstar,2]
                
                curr = preheadway - posthd
                rinfo[i].insert(0,[t_n,curr])
            #another strategy to get the headway in the case that there aren't any places we can estimate it from 
                
            else:
                #it never reaches this point unless in a special case
                leadlist = np.unique(sim[curveh][:t_n-t_nstar,4])
                if len(leadlist)>1 and 0 in leadlist: #otherwise if we are able to use the other strategy then use that 
                
                    cursim = sim[curveh][:t_n-t_nstar,:].copy()
                    cursim = cursim[cursim[:,7]==merge_from_lane]
                    cursim = cursim[cursim[:,4]!=0]
                    
                    curt = cursim[-1,1]
                    curlead = cursim[-1,4]
                    leadt_nstar = platooninfo[curlead][0]
                    
                    prehd = sim[curlead][curt-leadt_nstar,2]-sim[curlead][curt-leadt_nstar,6]-cursim[-1,2] #headway before 
                    
                    postlead = sim[curveh][t_n-t_nstar,4]
                    postleadt_nstar = platooninfo[postlead][0]
                    
                    posthd = sim[postlead][t_n-postleadt_nstar,2]-sim[postlead][t_n-postleadt_nstar,6]-sim[curveh][t_n-t_nstar,2]
                    curr=prehd-posthd
                    
                    rinfo[i].insert(0,[t_n,curr])
                
                else: #if neither strategy can be used then we can't get a merger r constant for the current vehicle. 
                    continue 
        
    return rinfo



def merge_rconstant2(platoons, platooninfo, sim, leadinfo, rinfo, relax_constant = 100,merge_from_lane= 7,merge_lane = 6, datalen =9,h=.1):
    #this one doesn't seem to work very well. use merge_rconstant (called by _r4 by default)
    
    for i in range(len(platoons)):
        curveh = platoons[i]
        t_nstar, t_n, T_nm1, T_n = platooninfo[curveh][0:4]
        lanelist = np.unique(sim[curveh][:t_n-t_nstar,7])
        leadlist = np.unique(sim[curveh][:t_n-t_nstar,4])
        
        if merge_from_lane in lanelist and merge_lane not in lanelist and sim[curveh][t_n-t_nstar,7]==merge_lane: #get a merge constant when a vehicle's simulation starts when they enter the highway #
            if len(leadlist)>1 and 0 in leadlist: #if we can use the strategy using the last known headway
                
                cursim = sim[curveh][:t_n-t_nstar,:].copy()
                cursim = cursim[cursim[:,7]==merge_from_lane]
                cursim = cursim[cursim[:,4]!=0]
                
                curt = int(cursim[-1,1])
                curlead = int(cursim[-1,4])
                leadt_nstar = platooninfo[curlead][0]
                
                prehd = sim[curlead][curt-leadt_nstar,2]-sim[curlead][curt-leadt_nstar,6]-cursim[-1,2] #headway before 
                
                postlead = sim[curveh][t_n-t_nstar,4]
                postleadt_nstar = platooninfo[postlead][0]
                
                posthd = sim[postlead][t_n-postleadt_nstar,2]-sim[postlead][t_n-postleadt_nstar,6]-sim[curveh][t_n-t_nstar,2]
                curr=prehd-posthd
                
                rinfo[i].insert(0,[t_n,curr])
            
            else: #otherwise try the other strategy 
                lead = np.zeros((T_n+1-t_n,datalen)) #initialize the lead vehicle trajectory 
                for j in leadinfo[i]:
                    curleadid = j[0] #current leader ID 
                    leadt_nstar = int(sim[curleadid][0,1]) #t_nstar for the current lead, put into int 
                    lead[j[1]-t_n:j[2]+1-t_n,:] = sim[curleadid][j[1]-leadt_nstar:j[2]+1-leadt_nstar,:] #get the lead trajectory from simulation
                headway =  lead[:,2]-sim[curveh][t_n-t_nstar:,2]-lead[:,6]
                headway = headway[:T_nm1+1-t_n]
                #calculate the average headway when not close to lane changing events 
                headwaybool = np.ones(len(headway),dtype=bool)
                for j in rinfo[i]:
                    headwaybool[j[0]-t_n:j[0]-t_n+relax_constant]=0
                
                headway = headway[headwaybool]
                if len(headway)>0: #if it is possible to use this strategy
                    preheadway = np.mean(headway)
                    postlead = sim[curveh][t_n-t_nstar,4]
                    postleadt_nstar = platooninfo[postlead][0]
                    
                    posthd = sim[postlead][t_n-postleadt_nstar,2]-sim[postlead][t_n-postleadt_nstar,6]-sim[curveh][t_n-t_nstar,2]
                    
                    curr = preheadway - posthd
                    rinfo[i].insert(0,[t_n,curr])
                else: 
                    continue #nothing we can do             
    return rinfo


def obj_helper(plist,model,modeladjsys,modeladj, meas,sim,platooninfo,platoonlist,makeleadfolfun,platoonobjfn,args,manual=False):
    #this will get the objective of a list of platoons and in doing so get the simulated trajectories loaded into sim whereas sim is normally just unchanged 
    #if only using the optimization algorithm. 
    count = 0 
    
    for i in platoonlist: 
        if manual: 
            p = plist[count]
        else:
            p = plist[count][0] #this is supposed to be for directly passing in results of optimization for scipy.optimize.minimize routines
        leadinfo, folinfo, rinfo = makeleadfolfun(i,platooninfo,sim)
        
        obj = platoonobjfn(p,model,modeladj,modeladjsys,meas,sim,platooninfo,i,leadinfo,folinfo,rinfo,*args)
        count += 1
    return sim 


def arraytraj(meas,followerchain, presim = False, postsim = False,datalen=9):  
    #puts output from makefollerchain/makeplatooninfo into a dict where the key is frame ID, value is array of vehicle and their position, speed 
    #we can include the presimulation (t_nstar to t_n) as well as postsimulation (T_nm1 to T_n) but including those are optional 
    #this will put in the whole trajectory based off of the times in followerchain
    t = 1 #index for first time
    T = 2 #index for last time
    if presim: #if presim is true we want the presimulation, so we start with t_nstar
        t = 0
    if postsim: 
        T = 3
    t_list = [] #t_n list
    T_list = [] #T_n list
    for i in followerchain.values():
        t_list.append(i[t])
        T_list.append(i[T])
    
    T_n = int(max(T_list))  #T_{n} #get maximum T 
    t_1 = int(min(t_list)) #t_1 #get minimum T
    mytime = range(t_1, T_n+1) #time range of data
    platoontraj = {k: np.empty((0,datalen)) for k in mytime}  #initialize output
    #fix this part, also make something to use this output to fix the lead/follow stuff in ngsim data 
    for i in followerchain.keys():
        
        curmeas = meas[i] #i is the vehicle id, these are the current measuremeents
        t_nstar = followerchain[i][0]
        t_n = followerchain[i][t] #first time we are using for current vehicle
        T_n = followerchain[i][T] #last time we are using for current vehicle 
        curmeas = curmeas[t_n-t_nstar:T_n-t_nstar+1,:] #get only the times requested 
        curtime = range(t_n,T_n+1)
        count = 0
        for j in curtime:
            platoontraj[j] = np.append(platoontraj[j], np.reshape(curmeas[count,:],(1,datalen)), axis=0)
            count += 1
    return platoontraj, mytime


def platoononly(platooninfo, platoon):
    #makes it so platooninfo only contains entries for platoon
    #platoon can be a platoon in form [ 1, 2, 3, etc.] or a list of those. 
    ans = {}

    if type(platoon[0]) == list:  #list of platoons 
        useplatoon = []
        for i in platoon:
            useplatoon.extend(i[:])
    else: 
        useplatoon = platoon
        
    for i in useplatoon[:]:
        ans[i] = platooninfo[i]
    return ans



def greedy_set_cover(subsets, parent_set):
    #copy pasted from stack exchange
    parent_set = set(parent_set)
    max = len(parent_set)
    # create the initial heap. Note 'subsets' can be unsorted,
    # so this is independent of whether remove_redunant_subsets is used.
    heap = []
    for s in subsets:
        # Python's heapq lets you pop the *smallest* value, so we
        # want to use max-len(s) as a score, not len(s).
        # len(heap) is just proving a unique number to each subset,
        # used to tiebreak equal scores.
        heapq.heappush(heap, [max-len(s), len(heap), s])
    results = []
    result_set = set()
    while result_set < parent_set:
#        logging.debug('len of result_set is {0}'.format(len(result_set)))
        best = []
        unused = []
        while heap:
            score, count, s = heapq.heappop(heap)
            if not best:
                best = [max-len(s - result_set), count, s]
                continue
            if score >= best[0]:
                # because subset scores only get worse as the resultset
                # gets bigger, we know that the rest of the heap cannot beat
                # the best score. So push the subset back on the heap, and
                # stop this iteration.
                heapq.heappush(heap, [score, count, s])
                break
            score = max-len(s - result_set)
            if score >= best[0]:
                unused.append([score, count, s])
            else:
                unused.append(best)
                best = [score, count, s]
        add_set = best[2]
#        logging.debug('len of add_set is {0} score was {1}'.format(len(add_set), best[0]))
        results.append(add_set)
        result_set.update(add_set)
        # subsets that were not the best get put back on the heap for next time.
        while unused:
            heapq.heappush(heap, unused.pop())
    return results
    
def SEobj_pervehicle(meas,sim,platooninfo,curplatoon, dim=2, h=.1):
    #takes as input meas, sim, platooninfo, curplatoon, 
    #outputs a list of the objective function for each vehicle. 
    out = []
    for i in curplatoon:
        t_nstar, t_n, T_nm1, T_n = platooninfo[i][0:4]
        curloss = sim[i][t_n-t_nstar:T_nm1+dim-t_nstar,2] -  meas[i][t_n-t_nstar:T_nm1+dim-t_nstar,2]
        curloss = np.sum(np.square(curloss))*h
        out.append(curloss)
    
    return out

def convert_to_rmse(obj,platooninfo,curplatoon, dim = 2, h=.1, delay = 0):
    #converts an objective in squared distance error into RMSE in distance
    #note if comparing platoons to single vehicles this aggregates the entire platoon which is different then doing each vehicle individually then aggreating. 
    #need a keyword parameter here for the special case of newell
    #this currently is only for newell delay. 
    ans = 0 
    num_obs = 0 
    for i in curplatoon:
        if delay == 0 : #ODE
            t_n, T_nm1, T_n = platooninfo[i][1:4]
            if T_n >= T_nm1+dim-1: #the reason you have dim is because at T_nm1 you get position and speed, speed determines position at next step. 
                #so when dim = 2 you will get simulation to T_nm1 +1 (when position is being used for loss). 
                num_obs += T_n-t_n+1
            else:
                num_obs += T_nm1-t_n+dim
        else: #if using TT or DDE model you will have a time delay
            t_n, T_nm1, T_n = platooninfo[i][1:4]
            offset = math.ceil(delay/h)
            offsetend = math.floor(delay/h)
            if T_nm1 + offsetend >= T_n: 
                end = T_n 
            else: 
                end = T_nm1 + offsetend
            start = t_n+offset
            num_obs += end-start+1
            #worth noting that the loss for a TT/DDE is interpolated over both simulation and measurements and averaged, so if you compute the loss 
            #based only on the simulation you will get something slightly different (see adjoint method paper)
    ans = (obj/(num_obs*h))**.5
    return ans 

def convert_to_prmse(meas,sim,platooninfo,my_id,dim=2,h=.1,datalen=9):
    
    # %based rmse.
    #note a huge fan of this metric since it mainly depends on the headway instead of the actual accuracy. 
    
    leadinfo,folinfo,unused = makeleadfolinfo([[],my_id],platooninfo,meas)
    
    t_nstar,t_n,T_nm1,T_n = platooninfo[my_id][0:4]
    lead = np.zeros((T_n+1-t_n,datalen)) #initialize the lead vehicle trajectory 
    for j in leadinfo[0]:
        curleadid = j[0] #current leader ID 
        leadt_nstar = int(sim[curleadid][0,1]) #t_nstar for the current lead, put into int 
        lead[j[1]-t_n:j[2]+1-t_n,:] = sim[curleadid][j[1]-leadt_nstar:j[2]+1-leadt_nstar,:] #get the lead trajectory from simulation 
            
    truelead = np.zeros((T_n+1-t_n,datalen)) #initialize the lead vehicle trajectory 
    for j in leadinfo[0]:
        curleadid = j[0] #current leader ID 
        leadt_nstar = int(sim[curleadid][0,1]) #t_nstar for the current lead, put into int 
        truelead[j[1]-t_n:j[2]+1-t_n,:] = meas[curleadid][j[1]-leadt_nstar:j[2]+1-leadt_nstar,:] #get the lead trajectory from simulation 
    
    
    headway =  lead[:,2]-sim[my_id][t_n-t_nstar:,2]-lead[:,6]
    trueheadway = truelead[:,2]-meas[my_id][t_n-t_nstar:,2]-truelead[:,6]
    
    ans = np.divide(np.absolute(headway-trueheadway),trueheadway)
    ans = np.square(ans)
    ans = np.mean(ans)
    ans = ans**.5
    return ans

def check_realistic(sim, platooninfo, limit = 10, h=.1):
    
    t_nstar, t_n, T_nm1, T_n = platooninfo[0:4]
    offset = t_n - t_nstar
    accel = np.zeros(T_nm1-t_n)#+dim-2 should be 
    count = 0 
    for i in range(len(accel)):
        accel[i] = (sim[i+1+offset,3]-sim[i+offset,3])/h
        if accel[i]>13.3 or accel[i] < -20:
            count += 1
        if count > 9: 
            return False
        if sim[i+offset,3]<0:
            return False
    return True

def lanevehlist2(data,lane,vehs):
    #given two vehicles ids vehs[0] and vehs[-1], find all vehicles inbetween the two in a single lane
    #this is an older version that was written before the sortveh3 was created. Now we can just use that and it's more robust. 
    #both versions do not support the edge case where the starting and ending vehicles are involved in circular dependency. 
    #the vehicles involved in the circular dependency may, or may not, be included in that case. 
	#This is used in makeplatoonlist for speical input when you give it a pair of vehicles 
	
    lanedata = data[data[:,7]==lane]
    
    veh0 = vehs[0]; vehm1 = vehs[-1]
    veh0traj = lanedata[lanedata[:,0]==veh0]
    vehm1traj = lanedata[lanedata[:,0]==vehm1]
    firsttime = veh0traj[0,1]; lasttime = vehm1traj[-1,1]
    lanedata = lanedata[np.all([lanedata[:,1]>=firsttime, lanedata[:,1]<=lasttime],axis=0)]
    vehlist = list(np.unique(lanedata[:,0]))
    
    #get vehicles in front of vhes[0] we don't want
    tn = veh0traj[0,1]; Tn = veh0traj[-1,1]
    lanedata1 = lanedata[np.all([lanedata[:,1]>=tn, lanedata[:,1]<=Tn],axis=0)] #data only in the times that initial vehicle is present
    
#    lanedata1 = np.sort(lanedata1,axis=0) #don't do this 
    
    badvehlist = set()
    for i in range(len(lanedata1)):
        curdt = int(lanedata1[i,1]-tn)
        if lanedata1[i,7] != lane:
            print(i)
        if lanedata1[i,2] > veh0traj[curdt,2]:
#            if lanedata1[i,0] in [581, 621]:
#                pass
            badvehlist.add(lanedata1[i,0])
        
#        try: 
#            if lanedata1[i,2] > veh0traj[curdt,2]:
#                badvehlist.add(lanedata1[i,0])
#        except IndexError:
#            print('why are you doing this to me')
    
    #same as above but for vehm1
    tn = vehm1traj[0,1]; Tn = vehm1traj[-1,1]
    lanedata1 = lanedata[np.all([lanedata[:,1]>=tn, lanedata[:,1]<=Tn],axis=0)] #data only in the times that initial vehicle is present
    
#    lanedata1 = np.sort(lanedata1,axis=0)
    
    for i in range(len(lanedata1)):
        curdt = int(lanedata1[i,1]-tn)
        if lanedata1[i,2] < vehm1traj[curdt,2]:
            badvehlist.add(lanedata1[i,0])
            
    for i in badvehlist: 
#        if i in vehs:
#            continue
        vehlist.remove(i)
        
    
    return vehlist, badvehlist

def re_diff(data,platooninfo, platoons, delay = 0, h=.1,speeds = True, accel = True):
    #re diferentiates speed and/or acceleration for either delay or no delay. 
    #This will work for platoonobjfn or TTobjfn functions, for any delayobjfn or stochobjfn functions you will need to modify this 
    #to redifferentiate over the correct times. 
    for i in platoons: #iterate over vehicles 
        t_nstar, t_n, T_nm1, T_n = platooninfo[i][0:4]
        if delay != 0:
            offset = math.ceil(delay/h)
            offsetend = math.floor(delay/h)
            if T_nm1 + offsetend >= T_n: 
                end = T_n 
            else: 
                end = T_nm1 + offsetend
            start = t_n+offset
            numobs = end-start+1
            if T_n == end: #need T_n to be at least 1 more than T_nm1
                numobs += -1
            
        else: 
            start = t_n
            end = T_nm1 
            numobs = end-start+1
            if T_n==T_nm1: #need T_n to be at least 1 more than T_nm1
                numobs += -1
        
        data[i][start-t_nstar,3] = (data[i][start-t_nstar+1,2] - data[i][start-t_nstar,2])/h
        if accel:
            for j in range(1,numobs):
                time = start+j
                data[i][time-t_nstar,3] = (data[i][time-t_nstar+1,2] - data[i][time-t_nstar,2])/h
                data[i][time-t_nstar-1,8] = (data[i][time-t_nstar,3] - data[i][time-t_nstar-1,3])/h
        else: 
            for j in range(1,numobs):
                time = start+j
                data[i][time-t_nstar,3] = (data[i][time-t_nstar+1,2] - data[i][time-t_nstar,2])/h
#                data[i][time-t_nstar-1,8] = (data[i][time-t_nstar,3] - data[i][time-t_nstar-1,3])/h
            
    return

def fin_dif_wrapper(p,args, *eargs, eps = 1e-8, **kwargs):   
    #returns the gradient for function with call signature obj = objfun(p, *args)
    #note you should pass in 'objfun' as the last entry in the tuple for args
    #so objfun = args[-1]
    #uses first order forward difference with step size eps to compute the gradient 
    out = np.zeros((len(p),))
    objfun = args[-1]
    obj = objfun(p,*args)
    for i in range(len(out)):
        curp = p
        curp[i] += eps
        out[i] = objfun(curp,*args)
    return (out-obj)/eps

def chain_metric(platoon, platooninfo, k = .9, type = 'lead', meas = []):
    #metric that defines how good a platoon is 
    #refer to platoon formation pdf for exact definition 
    res = 0
    for i in platoon:
        T = set(range(platooninfo[i][1], platooninfo[i][2]+1))
        res += c_metric(i, platoon, T, platooninfo, k, type, meas=meas)
    return res


def c_metric(veh, platoon, T, platooninfo, k = .9, type = 'lead', depth=0, meas = []):
    #defines how good a single vehicle in a specific time is. 
    #refer to platoon formation pdf for exact definition 
    
    # leadinfo, folinfo= makeleadinfo(platoon, platooninfo, meas),  makefolinfo(platoon, platooninfo, meas)
    # if veh not in platoon:
    #     return 0
    # targetsList = leadinfo[platoon.index(veh)] if type == 'lead' else folinfo[platoon.index(veh)]
    veh = int(veh)
    if type == 'lead':
        leadinfo = makeleadinfo([veh], platooninfo, meas)
        targetsList = leadinfo[0]
    else:
        folinfo = makefolinfo([veh], platooninfo, meas)
        # targetsList = folinfo[0]
        temp = folinfo[0]
        targetsList = []
        for i in temp:
            Tnstart = platooninfo[i[0]][1]
            Tnm1 = platooninfo[i[0]][2]
            start = max(Tnstart, i[1])
            end = min(Tnm1, i[2])
            if start<end:
                targetsList.append([i[0], start, end])




    def getL(veh, platoon, T):
        L = set([])
        # if veh not in platoon:
        #     return L
        # targetsList = leadinfo[platoon.index(veh)]
        temp = set([])
        for i in targetsList:
            if i[0] not in platoon:
                continue
            temp.update(range(i[1], i[2]+1))
        L = T.intersection(temp)
        if len(L)>0:
            print(veh, len(L), depth)
        return L

    def getLead(veh, platoon, T):
        # if veh not in platoon:
        #     return []
        # targetsList = leadinfo[platoon.index(veh)]
        leads = []
        for i in targetsList:
            if i[0] in platoon and (i[1] in T or i[2] in T):
                leads.append(i[0])
        leads = list(set(leads))
        return leads

    def getTimes(veh, lead, T):
        # targetsList = leadinfo[platoon.index(veh)]
        temp = set([])
        for i in targetsList:
            if i[0] == lead:
                temp.update(range(i[1], i[2]+1))
        temp = T.intersection(temp)
        return temp

    res = len(getL(veh, platoon, T))
    leads = getLead(veh, platoon, T)

    for i in leads:
        res += k*c_metric(i, platoon, getTimes(veh, i, T), platooninfo, k=k, type=type, depth=depth+1, meas=meas)
    return res

def cirdep_metric(platoonlist, platooninfo, k = .9, type = 'veh', meas=[]):
    #platoonlist - list of platoons 
    
    #type = veh checks for circular dependencies. For every vehicle which is 
    #causing a circular dependency it outputs: 
    # tuple of [vehicle, list of lead vehicles, list of lead vehicles platoon indices], vehicle platoon index
    #where vehicle is the vehicle with the circular dependency, which it has becuase of lead vehicles. 
    
    #type = num quantifies how bad a circular dependency is by computing 
    #the change to chain metric when adding the lead vehicle to the platoon 
    #with the circular dependency. Output is a list of floats, same length as platoonlist. 
    if type == 'veh':
        cirList = []
        after = set([])
        veh2platoon = {} #converts vehicle to platoon index
        for i in range(len(platoonlist)):
            for j in platoonlist[i]: 
                veh2platoon[j] = i
        for i in range(len(platoonlist)):
            after.update(platoonlist[i])
        for i in range(len(platoonlist)):
            after -= set(platoonlist[i])
            leadinfo = makeleadinfo(platoonlist[i], platooninfo, meas)
            for j in range(len(platoonlist[i])):
                leaders = [k[0] for k in leadinfo[j]]
                leaders = set(leaders)
                circleadveh = leaders.intersection(after)
                if len(circleadveh)>0:
                    cirList.append(([platoonlist[i][j], list(circleadveh), [veh2platoon[k] for k in circleadveh]], i))
        return cirList
    elif type == 'num':
        res = 0
        cirList = []
        after = set([])
        leader_violate_map = defaultdict(list)
        for i in range(len(platoonlist)): #get set of all vehicles
            after.update(platoonlist[i])
        for i in range(len(platoonlist)): #i is current platoon
            after -= set(platoonlist[i]) #remove vehicles from current platoon 
            leadinfo= makeleadinfo(platoonlist[i], platooninfo, meas)
            temp = []

            for j in range(len(platoonlist[i])):
                leaders = [k[0] for k in leadinfo[j]]
                leaders = set(leaders)
                leaders_after = leaders.intersection(after) #leaders_after are any leaders of i which are not yet calibrated
                if len(leaders_after) > 0:
                    # temp.append((list(leaders_after), i))

                    violated_leaders = [v for v in leaders_after if i not in leader_violate_map[v]]
                    if violated_leaders: # Remove duplicated invoking of chain matric
                        temp.append((violated_leaders, i))
                    for l in leaders_after:
                        leader_violate_map[l].append(i)

                else:
                    temp.append(None)
            cirList.append(temp)
        res = []
        for i in cirList:
            if i == None: 
                res.append(0)
            else:
                temp = 0
                for j in i:
                    if j:
                        for l in j[0]:
                            T = set(range(platooninfo[l][1], platooninfo[l][3]+1))
                            temp += c_metric(l, platoonlist[j[1]], T, platooninfo, k=k, type='follower',meas = meas)
            res.append(temp)
        return res

def plotformat(sim, auxinfo, roadinfo, endtimeind = 3000, density = 2, indlist = [], specialind = 21):
    #get output from simulation module into a format we can plot using plotting functions
    #density = k plots every kth vehicle, indlist = [keys] plots keys only. 
    #specialind doesn't do anything. 
    
    L = roadinfo[0]
    platooninfo = {} #need platooninfo 0 - 4 : observation times
    meas = {} #need columns 1, 2,3, 7
    idcount = 0
    speciallist = []
    
    if indlist == []:
        uselist = list(sim.keys())[::density]
    else:
        uselist = indlist
    for i in uselist:
        cur = sim[i]
        
#        if i == specialind:
#            speciallist.append(idcount)
        
        #initialize output for current vehicle
        curtime = 0
        prevx = -1
        tlist = []
        xlist = []
        vlist = []
        platooninfo[idcount] = [curtime, curtime, None, None]
        for counter, j in enumerate(cur): #for each vehicle
            if j[0] < prevx: #if distance decreases its because we wrapped around - store measurements in new vehicle 
                
                endtime = counter #get end time and update platooninfo
                platooninfo[idcount][2:] = [endtime-1,endtime-1]
                #update meas
                meas[idcount] = np.zeros((endtime-curtime,8))
                meas[idcount][:,1] = tlist
                meas[idcount][:,2] = xlist
                meas[idcount][:,3] = vlist
                #lane just set always to 1
                meas[idcount][:,7] = 1
                
                #reset iteration
                idcount += 1
                curtime = endtime
                prevx = j[0]
                tlist = [curtime]
                xlist = [j[0]]
                vlist = [j[1]]
                platooninfo[idcount] = [curtime, curtime, None, None]
                continue
                
            tlist.append(counter)
            xlist.append(j[0])
            vlist.append(j[1])
            prevx = j[0]
            
            if counter >= endtimeind:
                break
            
        #also need to finish current once for loop ends 
        endtime = counter #get end time and update platooninfo
        platooninfo[idcount][2:] = [endtime,endtime]
        #update meas
        meas[idcount] = np.zeros((endtime-curtime+1,8))
        meas[idcount][:,1] = tlist
        meas[idcount][:,2] = xlist
        meas[idcount][:,3] = vlist
        #lane just set always to 1
        meas[idcount][:,7] = 1
        
        idcount += 1
            

    return meas, platooninfo

def is_pareto_efficient(costs, return_mask = True): #copy pasted from stack exchange. finds pareto front assuming lower is better
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :param return_mask: True to return a mask
    :return: An array of indices of pareto-efficient points.
        If return_mask is True, this will be an (n_points, ) boolean array
        Otherwise it will be a (n_efficient_points, ) integer array of indices.
    """
    is_efficient = np.arange(costs.shape[0])
    n_points = costs.shape[0]
    next_point_index = 0  # Next index in the is_efficient array to search for
    while next_point_index<len(costs):
        nondominated_point_mask = np.any(costs<costs[next_point_index], axis=1)
        nondominated_point_mask[next_point_index] = True
        is_efficient = is_efficient[nondominated_point_mask]  # Remove dominated points
        costs = costs[nondominated_point_mask]
        next_point_index = np.sum(nondominated_point_mask[:next_point_index])+1
    if return_mask:
        is_efficient_mask = np.zeros(n_points, dtype = bool)
        is_efficient_mask[is_efficient] = True
        return is_efficient_mask
    else:
        return is_efficient

#################################################
#old makeleadfolinfo functions - this ravioli code has now been fixed! 
#################################################
#def makeleadfolinfo1(platoons, platooninfo, sim, *args):
##	inputs: 
##    platoons: the platoon we want to calibrate 
##    platooninfo: output from makeplatooninfo
##    meas: measurements; i.e. the data put into dictionary form, also an output from makeplatooninfo
##    
##outputs: 
##    leadinfo: list of lists with the relevant lead info 
##    folinfo: list of lists with the relevant fol info (see below)
##    
##EXAMPLE: 
##    platoons = [[],5,7] means we want to calibrate vehicles 5 and 7 in a platoon
##    
##    leadinfo = [[[1,1,10],[2,11,20]],[[5,10,500]]] Means that vehicle 5 has vehicle 1 as a leader from 1 to 10, 2 as a leader from 11 to 20. 
##    vehicle 7 has 3 as a leader from 10 to 500 (leadinfo[i] is the leader info for platoons[i]. leadinfo[i] is a list of lists, so leadinfo is a list of lists of lists.)
##    
##    folinfo = [[[7,11,20]], [[]]] Means that vehicle 5 has vehicle 7 as a follower in the platoon from 11 to 20, and vehicle 7 has no followers in the platoon
#	
#	
#    #this will get the leader and follower info to use in the objective function and gradient calculation. this will save time 
#    #because of the way scipy works, we cant update the *args we pass in to our custom functions, so if we do this preprocessing here
#    #it will save us from having to do this over and over again every single time we evaluate the objective or gradient.
#    #however, it is still not ideal. in a totally custom implementation our optimization routines wouldn't have to do this at all 
#    #because we would be able to update the *args
#    #additionally, in a totally custom implementation we would make use of the fact that we need to actually evaluate the objective before we can 
#    #evaluate the gradient. in the scipy implementation, everytime we evaluate the gradient we actually evaluate the objective again, which is totally wasted time. 
#    
#    #input/output example:
#    #input : platoons= [[],5]
#    #output : [[[1,1,100]]], [[]] means that vehicle 5 has vehicle 1 as a leader for frame id 1 to frameid 100, and that vehicle 5 has no followers 
#    #which are in platoons
#    
#    leadinfo = [] #initialize output 
#    folinfo = []
#    rinfo = [] #just empty stuff
#    
#    for i in platoons: #iterate over each vehicle in the platoon
#        curleadinfo = [] #for each vehicle, we get these and then these are appeneded at the end so we have a list of the info for each vehicle in the platoon
#        curfolinfo = []
#        currinfo = []
#        t_nstar, t_n, T_nm1, T_n = platooninfo[i][0:4] #get times for current vehicle 
#        leadlist = sim[i][t_n-t_nstar:T_nm1-t_nstar+1,4] #this gets the leaders for each timestep of the current vehicle\
#        curlead = leadlist[0] #initialize current leader
#        curleadinfo.append([curlead, t_n]) #initialization 
#        for j in range(len(leadlist)):
#            if leadlist[j] != curlead: #if there is a new leader
#                curlead = leadlist[j] #update the current leader
#                curleadinfo[-1].append(t_n+j-1) #last time (in frameID) the old leader is observed
#                curleadinfo.append([curlead,t_n+j]) #new leader and the first time (in frameID) it is observed. 
#        curleadinfo[-1].append(t_n+len(leadlist)-1) #termination
#        
#        #do essentially the same things for followers now (we need the follower for adjoint system)
#        #only difference is that we only need to put things in if their follower is in platoons
#        follist = sim[i][t_n-t_nstar:T_n-t_nstar+1,5] #list of followers
#        curfol = follist[0]
#        if curfol in platoons: #if the current follower is in platoons we initialize
#            curfolinfo.append([curfol,t_n])
#        for j in range(len(follist)): #check what we just made to see if we need to put stuff in folinfo
#            if follist[j] != curfol: #if there is a new follower
#                curfol = follist[j]
#                if curfolinfo != []: #if there is anything in curfolinfo
#                    curfolinfo[-1].append(t_n+j-1) #we finish the interval
#                if curfol in platoons: #if new follower is in platoons
#                    curfolinfo.append([curfol,t_n+j]) #start the next interval
#        if curfolinfo != []: #if there is anything to finish
#            curfolinfo[-1].append(t_n+len(follist)-1) #finish it 
#        leadinfo.append(curleadinfo) #append what we just made to the total list 
#        folinfo.append(curfolinfo) 
#        rinfo.append(currinfo) #just a lot of empty stuff in this version
#        
#                
#        
#    return leadinfo, folinfo, rinfo


#def makeleadfolinfo_r(platoons, platooninfo, sim,use_merge_constant=False):
#    #positive r only
#    
#    #this will get the leader and follower info to use in the objective function and gradient calculation. this will save time 
#    #because of the way scipy works, we cant update the *args we pass in to our custom functions, so if we do this preprocessing here
#    #it will save us from having to do this over and over again every single time we evaluate the objective or gradient.
#    #however, it is still not ideal. in a totally custom implementation our optimization routines wouldn't have to do this at all 
#    #because we would be able to update the *args
#    #additionally, in a totally custom implementation we would make use of the fact that we need to actually evaluate the objective before we can 
#    #evaluate the gradient. in the scipy implementation, everytime we evaluate the gradient we actually evaluate the objective again, which is totally wasted time. 
#    
#    #input/output example:
#    #input : platoons= [[],5]
#    #output : [[[1,1,100]]], [[]] means that vehicle 5 has vehicle 1 as a leader for frame id 1 to frameid 100, and that vehicle 5 has no followers 
#    #which are in platoons
#    
#    #note that you can either pass in sim or meas in the position for sim. 
#    
#    leadinfo = [] #initialize output 
#    folinfo = []
#    rinfo = []
#    
#    for i in platoons: #iterate over each vehicle in the platoon
#        curleadinfo = [] #for each vehicle, we get these and then these are appeneded at the end so we have a list of the info for each vehicle in the platoon
#        curfolinfo = []
#        currinfo = []
#        t_nstar, t_n, T_nm1, T_n = platooninfo[i][0:4] #get times for current vehicle 
#        leadlist = sim[i][t_n-t_nstar:T_nm1-t_nstar+1,4] #this gets the leaders for each timestep of the current vehicle\
#        curlead = leadlist[0] #initialize current leader
#        curleadinfo.append([curlead, t_n]) #initialization 
#        for j in range(len(leadlist)):
#            if leadlist[j] != curlead: #if there is a new leader
#                newlead = leadlist[j]
#                oldlead = curlead
#                ##############relaxation constant calculation
#                newt_nstar = platooninfo[newlead][0]
#                oldt_nstar = platooninfo[oldlead][0]
#                olds = sim[oldlead][t_n+j-1-oldt_nstar,2] - sim[oldlead][0,6] - sim[i][t_n+j-1-t_nstar,2] #the time is t_n+j-1; this is the headway
#                news = sim[newlead][t_n+j-newt_nstar,2] - sim[newlead][0,6] - sim[i][t_n+j-t_nstar,2] #the time is t_n+j
#                #below if only adds if headway decreases, otherwise we will always add the relaxation constant, even if it is negative. 
#                if news < olds: #if the headway decreases, then we will add in the relaxation phenomenon
#                    currinfo.append([t_n+j, olds-news]) #we append the time the LC happens (t_n+j), and the "gamma" which is what I'm calling the initial constant we adjust the headway by (olds-news)
##                currinfo.append([t_n+j,olds-news])
#                #################################################
#                curlead = leadlist[j] #update the current leader
#                curleadinfo[-1].append(t_n+j-1) #last time (in frameID) the old leader is observed
#                curleadinfo.append([curlead,t_n+j]) #new leader and the first time (in frameID) it is observed.
#                
#        curleadinfo[-1].append(t_n+len(leadlist)-1) #termination
#        
#        
#        
#        #do essentially the same things for followers now (we need the follower for adjoint system)
#        #only difference is that we only need to put things in if their follower is in platoons
#        follist = sim[i][t_n-t_nstar:T_n-t_nstar+1,5] #list of followers
#        curfol = follist[0]
#        if curfol in platoons: #if the current follower is in platoons we initialize
#            curfolinfo.append([curfol,t_n])
#        for j in range(len(follist)): #check what we just made to see if we need to put stuff in folinfo
#            if follist[j] != curfol: #if there is a new follower
#                curfol = follist[j]
#                if curfolinfo != []: #if there is anything in curfolinfo
#                    curfolinfo[-1].append(t_n+j-1) #we finish the interval
#                if curfol in platoons: #if new follower is in platoons
#                    curfolinfo.append([curfol,t_n+j]) #start the next interval
#        if curfolinfo != []: #if there is anything to finish
#            curfolinfo[-1].append(t_n+len(follist)-1) #finish it 
#        leadinfo.append(curleadinfo) #append what we just made to the total list 
#        folinfo.append(curfolinfo) 
#        rinfo.append(currinfo)
#        
#    if use_merge_constant: 
#        rinfo = merge_rconstant(platoons,platooninfo,sim,leadinfo,rinfo,100)
#                
#        
#    return leadinfo, folinfo, rinfo 
#
#def makeleadfolinfo_r2(platoons, platooninfo, sim,use_merge_constant=False):
#    #negative r
#    
#    #this will get the leader and follower info to use in the objective function and gradient calculation. this will save time 
#    #because of the way scipy works, we cant update the *args we pass in to our custom functions, so if we do this preprocessing here
#    #it will save us from having to do this over and over again every single time we evaluate the objective or gradient.
#    #however, it is still not ideal. in a totally custom implementation our optimization routines wouldn't have to do this at all 
#    #because we would be able to update the *args
#    #additionally, in a totally custom implementation we would make use of the fact that we need to actually evaluate the objective before we can 
#    #evaluate the gradient. in the scipy implementation, everytime we evaluate the gradient we actually evaluate the objective again, which is totally wasted time. 
#    
#    #input/output example:
#    #input : platoons= [[],5]
#    #output : [[[1,1,100]]], [[]] means that vehicle 5 has vehicle 1 as a leader for frame id 1 to frameid 100, and that vehicle 5 has no followers 
#    #which are in platoons
#    
#    #note that you can either pass in sim or meas in the position for sim. 
#    
#    leadinfo = [] #initialize output 
#    folinfo = []
#    rinfo = []
#    
#    for i in platoons: #iterate over each vehicle in the platoon
#        curleadinfo = [] #for each vehicle, we get these and then these are appeneded at the end so we have a list of the info for each vehicle in the platoon
#        curfolinfo = []
#        currinfo = []
#        t_nstar, t_n, T_nm1, T_n = platooninfo[i][0:4] #get times for current vehicle 
#        leadlist = sim[i][t_n-t_nstar:T_nm1-t_nstar+1,4] #this gets the leaders for each timestep of the current vehicle\
#        curlead = leadlist[0] #initialize current leader
#        curleadinfo.append([curlead, t_n]) #initialization 
#        for j in range(len(leadlist)):
#            if leadlist[j] != curlead: #if there is a new leader
#                newlead = leadlist[j]
#                oldlead = curlead
#                ##############relaxation constant calculation
#                newt_nstar = platooninfo[newlead][0]
#                oldt_nstar = platooninfo[oldlead][0]
#                olds = sim[oldlead][t_n+j-1-oldt_nstar,2] - sim[oldlead][0,6] - sim[i][t_n+j-1-t_nstar,2] #the time is t_n+j-1; this is the headway
#                news = sim[newlead][t_n+j-newt_nstar,2] - sim[newlead][0,6] - sim[i][t_n+j-t_nstar,2] #the time is t_n+j
#                #below if only adds if headway decreases, otherwise we will always add the relaxation constant, even if it is negative. 
#                if news > olds: #if the headway increases, then we will add in the relaxation phenomenon
#                    currinfo.append([t_n+j, olds-news]) #we append the time the LC happens (t_n+j), and the "gamma" which is what I'm calling the initial constant we adjust the headway by (olds-news)
##                currinfo.append([t_n+j,olds-news])
#                #################################################
#                curlead = leadlist[j] #update the current leader
#                curleadinfo[-1].append(t_n+j-1) #last time (in frameID) the old leader is observed
#                curleadinfo.append([curlead,t_n+j]) #new leader and the first time (in frameID) it is observed.
#                
#        curleadinfo[-1].append(t_n+len(leadlist)-1) #termination
#        
#        
#        
#        #do essentially the same things for followers now (we need the follower for adjoint system)
#        #only difference is that we only need to put things in if their follower is in platoons
#        follist = sim[i][t_n-t_nstar:T_n-t_nstar+1,5] #list of followers
#        curfol = follist[0]
#        if curfol in platoons: #if the current follower is in platoons we initialize
#            curfolinfo.append([curfol,t_n])
#        for j in range(len(follist)): #check what we just made to see if we need to put stuff in folinfo
#            if follist[j] != curfol: #if there is a new follower
#                curfol = follist[j]
#                if curfolinfo != []: #if there is anything in curfolinfo
#                    curfolinfo[-1].append(t_n+j-1) #we finish the interval
#                if curfol in platoons: #if new follower is in platoons
#                    curfolinfo.append([curfol,t_n+j]) #start the next interval
#        if curfolinfo != []: #if there is anything to finish
#            curfolinfo[-1].append(t_n+len(follist)-1) #finish it 
#        leadinfo.append(curleadinfo) #append what we just made to the total list 
#        folinfo.append(curfolinfo) 
#        rinfo.append(currinfo)
#        
#    if use_merge_constant: 
#        rinfo = merge_rconstant(platoons,platooninfo,sim,leadinfo,rinfo,100)
#                
#        
#    return leadinfo, folinfo, rinfo 
#
#def makeleadfolinfo_r3(platoons, platooninfo, sim,use_merge_constant=False):
#    #positive and negative r. 
#    
#    #this will get the leader and follower info to use in the objective function and gradient calculation. this will save time 
#    #because of the way scipy works, we cant update the *args we pass in to our custom functions, so if we do this preprocessing here
#    #it will save us from having to do this over and over again every single time we evaluate the objective or gradient.
#    #however, it is still not ideal. in a totally custom implementation our optimization routines wouldn't have to do this at all 
#    #because we would be able to update the *args
#    #additionally, in a totally custom implementation we would make use of the fact that we need to actually evaluate the objective before we can 
#    #evaluate the gradient. in the scipy implementation, everytime we evaluate the gradient we actually evaluate the objective again, which is totally wasted time. 
#    
#    #input/output example:
#    #input : platoons= [[],5]
#    #output : [[[1,1,100]]], [[]] means that vehicle 5 has vehicle 1 as a leader for frame id 1 to frameid 100, and that vehicle 5 has no followers 
#    #which are in platoons
#    
#    #note that you can either pass in sim or meas in the position for sim. 
#    
#    leadinfo = [] #initialize output 
#    folinfo = []
#    rinfo = []
#    
#    for i in platoons: #iterate over each vehicle in the platoon
#        curleadinfo = [] #for each vehicle, we get these and then these are appeneded at the end so we have a list of the info for each vehicle in the platoon
#        curfolinfo = []
#        currinfo = []
#        t_nstar, t_n, T_nm1, T_n = platooninfo[i][0:4] #get times for current vehicle 
#        leadlist = sim[i][t_n-t_nstar:T_nm1-t_nstar+1,4] #this gets the leaders for each timestep of the current vehicle\
#        curlead = leadlist[0] #initialize current leader
#        curleadinfo.append([curlead, t_n]) #initialization 
#        for j in range(len(leadlist)):
#            if leadlist[j] != curlead: #if there is a new leader
#                newlead = leadlist[j]
#                oldlead = curlead
#                ##############relaxation constant calculation
#                newt_nstar = platooninfo[newlead][0]
#                oldt_nstar = platooninfo[oldlead][0]
#                olds = sim[oldlead][t_n+j-1-oldt_nstar,2] - sim[oldlead][0,6] - sim[i][t_n+j-1-t_nstar,2] #the time is t_n+j-1; this is the headway
#                news = sim[newlead][t_n+j-newt_nstar,2] - sim[newlead][0,6] - sim[i][t_n+j-t_nstar,2] #the time is t_n+j
#                #below if only adds if headway decreases, otherwise we will always add the relaxation constant, even if it is negative. 
##                if news < olds: #if the headway decreases, then we will add in the relaxation phenomenon
##                    currinfo.append([t_n+j, olds-news]) #we append the time the LC happens (t_n+j), and the "gamma" which is what I'm calling the initial constant we adjust the headway by (olds-news)
#                currinfo.append([t_n+j,olds-news])
#                #################################################
#                curlead = leadlist[j] #update the current leader
#                curleadinfo[-1].append(t_n+j-1) #last time (in frameID) the old leader is observed
#                curleadinfo.append([curlead,t_n+j]) #new leader and the first time (in frameID) it is observed.
#                
#        curleadinfo[-1].append(t_n+len(leadlist)-1) #termination
#        
#        
#        
#        #do essentially the same things for followers now (we need the follower for adjoint system)
#        #only difference is that we only need to put things in if their follower is in platoons
#        follist = sim[i][t_n-t_nstar:T_n-t_nstar+1,5] #list of followers
#        curfol = follist[0]
#        if curfol in platoons: #if the current follower is in platoons we initialize
#            curfolinfo.append([curfol,t_n])
#        for j in range(len(follist)): #check what we just made to see if we need to put stuff in folinfo
#            if follist[j] != curfol: #if there is a new follower
#                curfol = follist[j]
#                if curfolinfo != []: #if there is anything in curfolinfo
#                    curfolinfo[-1].append(t_n+j-1) #we finish the interval
#                if curfol in platoons: #if new follower is in platoons
#                    curfolinfo.append([curfol,t_n+j]) #start the next interval
#        if curfolinfo != []: #if there is anything to finish
#            curfolinfo[-1].append(t_n+len(follist)-1) #finish it 
#        leadinfo.append(curleadinfo) #append what we just made to the total list 
#        folinfo.append(curfolinfo) 
#        rinfo.append(currinfo)
#        
#    if use_merge_constant: 
#        rinfo = merge_rconstant(platoons,platooninfo,sim,leadinfo,rinfo,100)
#                
#        
#    return leadinfo, folinfo, rinfo 
#
#def makeleadfolinfo_r4(platoons, platooninfo, sim,use_merge_constant=True):
#    #positive and negative r. 
#    #merger rule estimates relaxtaion from average headway first, then does last known headway on on-ramp
#    #this one works a lot better than _r5
#    
#    #this will get the leader and follower info to use in the objective function and gradient calculation. this will save time 
#    #because of the way scipy works, we cant update the *args we pass in to our custom functions, so if we do this preprocessing here
#    #it will save us from having to do this over and over again every single time we evaluate the objective or gradient.
#    #however, it is still not ideal. in a totally custom implementation our optimization routines wouldn't have to do this at all 
#    #because we would be able to update the *args
#    #additionally, in a totally custom implementation we would make use of the fact that we need to actually evaluate the objective before we can 
#    #evaluate the gradient. in the scipy implementation, everytime we evaluate the gradient we actually evaluate the objective again, which is totally wasted time. 
#    
#    #input/output example:
#    #input : platoons= [[],5]
#    #output : [[[1,1,100]]], [[]] means that vehicle 5 has vehicle 1 as a leader for frame id 1 to frameid 100, and that vehicle 5 has no followers 
#    #which are in platoons
#    
#    #note that you can either pass in sim or meas in the position for sim. 
#    
#    leadinfo = [] #initialize output 
#    folinfo = []
#    rinfo = []
#    
#    for i in platoons: #iterate over each vehicle in the platoon
#        curleadinfo = [] #for each vehicle, we get these and then these are appeneded at the end so we have a list of the info for each vehicle in the platoon
#        curfolinfo = []
#        currinfo = []
#        t_nstar, t_n, T_nm1, T_n = platooninfo[i][0:4] #get times for current vehicle 
#        leadlist = sim[i][t_n-t_nstar:T_nm1-t_nstar+1,4] #this gets the leaders for each timestep of the current vehicle\
#        curlead = leadlist[0] #initialize current leader
#        curleadinfo.append([curlead, t_n]) #initialization 
#        for j in range(len(leadlist)):
#            if leadlist[j] != curlead: #if there is a new leader
#                newlead = leadlist[j]
#                oldlead = curlead
#                ##############relaxation constant calculation
#                newt_nstar = platooninfo[newlead][0]
#                oldt_nstar = platooninfo[oldlead][0]
#                olds = sim[oldlead][t_n+j-1-oldt_nstar,2] - sim[oldlead][0,6] - sim[i][t_n+j-1-t_nstar,2] #the time is t_n+j-1; this is the headway
#                news = sim[newlead][t_n+j-newt_nstar,2] - sim[newlead][0,6] - sim[i][t_n+j-t_nstar,2] #the time is t_n+j
#                #below if only adds if headway decreases, otherwise we will always add the relaxation constant, even if it is negative. 
##                if news < olds: #if the headway decreases, then we will add in the relaxation phenomenon
##                    currinfo.append([t_n+j, olds-news]) #we append the time the LC happens (t_n+j), and the "gamma" which is what I'm calling the initial constant we adjust the headway by (olds-news)
#                currinfo.append([t_n+j,olds-news])
#                #################################################
#                curlead = leadlist[j] #update the current leader
#                curleadinfo[-1].append(t_n+j-1) #last time (in frameID) the old leader is observed
#                curleadinfo.append([curlead,t_n+j]) #new leader and the first time (in frameID) it is observed.
#                
#        curleadinfo[-1].append(t_n+len(leadlist)-1) #termination
#        
#        
#        
#        #do essentially the same things for followers now (we need the follower for adjoint system)
#        #only difference is that we only need to put things in if their follower is in platoons
#        follist = sim[i][t_n-t_nstar:T_n-t_nstar+1,5] #list of followers
#        curfol = follist[0]
#        if curfol in platoons: #if the current follower is in platoons we initialize
#            curfolinfo.append([curfol,t_n])
#        for j in range(len(follist)): #check what we just made to see if we need to put stuff in folinfo
#            if follist[j] != curfol: #if there is a new follower
#                curfol = follist[j]
#                if curfolinfo != []: #if there is anything in curfolinfo
#                    curfolinfo[-1].append(t_n+j-1) #we finish the interval
#                if curfol in platoons: #if new follower is in platoons
#                    curfolinfo.append([curfol,t_n+j]) #start the next interval
#        if curfolinfo != []: #if there is anything to finish
#            curfolinfo[-1].append(t_n+len(follist)-1) #finish it 
#        leadinfo.append(curleadinfo) #append what we just made to the total list 
#        folinfo.append(curfolinfo) 
#        rinfo.append(currinfo)
#        
#    if use_merge_constant: 
#        rinfo = merge_rconstant(platoons,platooninfo,sim,leadinfo,rinfo,200)
#            
#        
#    return leadinfo, folinfo, rinfo 
#
#def makeleadfolinfo_r5(platoons, platooninfo, sim,use_merge_constant=True):
#    #positive and negative r. 
#    #for merger, gets headway from last known headway on merge ramp first, then uses average headway if possible. 
#    
#    #this will get the leader and follower info to use in the objective function and gradient calculation. this will save time 
#    #because of the way scipy works, we cant update the *args we pass in to our custom functions, so if we do this preprocessing here
#    #it will save us from having to do this over and over again every single time we evaluate the objective or gradient.
#    #however, it is still not ideal. in a totally custom implementation our optimization routines wouldn't have to do this at all 
#    #because we would be able to update the *args
#    #additionally, in a totally custom implementation we would make use of the fact that we need to actually evaluate the objective before we can 
#    #evaluate the gradient. in the scipy implementation, everytime we evaluate the gradient we actually evaluate the objective again, which is totally wasted time. 
#    
#    #input/output example:
#    #input : platoons= [[],5]
#    #output : [[[1,1,100]]], [[]] means that vehicle 5 has vehicle 1 as a leader for frame id 1 to frameid 100, and that vehicle 5 has no followers 
#    #which are in platoons
#    
#    #note that you can either pass in sim or meas in the position for sim. 
#    
#    leadinfo = [] #initialize output 
#    folinfo = []
#    rinfo = []
#    
#    for i in platoons: #iterate over each vehicle in the platoon
#        curleadinfo = [] #for each vehicle, we get these and then these are appeneded at the end so we have a list of the info for each vehicle in the platoon
#        curfolinfo = []
#        currinfo = []
#        t_nstar, t_n, T_nm1, T_n = platooninfo[i][0:4] #get times for current vehicle 
#        leadlist = sim[i][t_n-t_nstar:T_nm1-t_nstar+1,4] #this gets the leaders for each timestep of the current vehicle\
#        curlead = leadlist[0] #initialize current leader
#        curleadinfo.append([curlead, t_n]) #initialization 
#        for j in range(len(leadlist)):
#            if leadlist[j] != curlead: #if there is a new leader
#                newlead = leadlist[j]
#                oldlead = curlead
#                ##############relaxation constant calculation
#                newt_nstar = platooninfo[newlead][0]
#                oldt_nstar = platooninfo[oldlead][0]
#                olds = sim[oldlead][t_n+j-1-oldt_nstar,2] - sim[oldlead][0,6] - sim[i][t_n+j-1-t_nstar,2] #the time is t_n+j-1; this is the headway
#                news = sim[newlead][t_n+j-newt_nstar,2] - sim[newlead][0,6] - sim[i][t_n+j-t_nstar,2] #the time is t_n+j
#                #below if only adds if headway decreases, otherwise we will always add the relaxation constant, even if it is negative. 
##                if news < olds: #if the headway decreases, then we will add in the relaxation phenomenon
##                    currinfo.append([t_n+j, olds-news]) #we append the time the LC happens (t_n+j), and the "gamma" which is what I'm calling the initial constant we adjust the headway by (olds-news)
#                currinfo.append([t_n+j,olds-news])
#                #################################################
#                curlead = leadlist[j] #update the current leader
#                curleadinfo[-1].append(t_n+j-1) #last time (in frameID) the old leader is observed
#                curleadinfo.append([curlead,t_n+j]) #new leader and the first time (in frameID) it is observed.
#                
#        curleadinfo[-1].append(t_n+len(leadlist)-1) #termination
#        
#        
#        
#        #do essentially the same things for followers now (we need the follower for adjoint system)
#        #only difference is that we only need to put things in if their follower is in platoons
#        follist = sim[i][t_n-t_nstar:T_n-t_nstar+1,5] #list of followers
#        curfol = follist[0]
#        if curfol in platoons: #if the current follower is in platoons we initialize
#            curfolinfo.append([curfol,t_n])
#        for j in range(len(follist)): #check what we just made to see if we need to put stuff in folinfo
#            if follist[j] != curfol: #if there is a new follower
#                curfol = follist[j]
#                if curfolinfo != []: #if there is anything in curfolinfo
#                    curfolinfo[-1].append(t_n+j-1) #we finish the interval
#                if curfol in platoons: #if new follower is in platoons
#                    curfolinfo.append([curfol,t_n+j]) #start the next interval
#        if curfolinfo != []: #if there is anything to finish
#            curfolinfo[-1].append(t_n+len(follist)-1) #finish it 
#        leadinfo.append(curleadinfo) #append what we just made to the total list 
#        folinfo.append(curfolinfo) 
#        rinfo.append(currinfo)
#        
#    if use_merge_constant: 
#        rinfo = merge_rconstant2(platoons,platooninfo,sim,leadinfo,rinfo,200)
#                
#        
#    return leadinfo, folinfo, rinfo 
#
#def makeleadfolinfo_r6(platoons, platooninfo, sim,use_merge_constant=False,merge_from_lane=7,merge_lane=6):
#    #positive and negative r. 
#    #if the lane change is a merger; we will get rid of it. 
#    
#    #this will get the leader and follower info to use in the objective function and gradient calculation. this will save time 
#    #because of the way scipy works, we cant update the *args we pass in to our custom functions, so if we do this preprocessing here
#    #it will save us from having to do this over and over again every single time we evaluate the objective or gradient.
#    #however, it is still not ideal. in a totally custom implementation our optimization routines wouldn't have to do this at all 
#    #because we would be able to update the *args
#    #additionally, in a totally custom implementation we would make use of the fact that we need to actually evaluate the objective before we can 
#    #evaluate the gradient. in the scipy implementation, everytime we evaluate the gradient we actually evaluate the objective again, which is totally wasted time. 
#    
#    #input/output example:
#    #input : platoons= [[],5]
#    #output : [[[1,1,100]]], [[]] means that vehicle 5 has vehicle 1 as a leader for frame id 1 to frameid 100, and that vehicle 5 has no followers 
#    #which are in platoons
#    
#    #note that you can either pass in sim or meas in the position for sim. 
#    
#    leadinfo = [] #initialize output 
#    folinfo = []
#    rinfo = []
#    
#    for i in platoons: #iterate over each vehicle in the platoon
#        curleadinfo = [] #for each vehicle, we get these and then these are appeneded at the end so we have a list of the info for each vehicle in the platoon
#        curfolinfo = []
#        currinfo = []
#        t_nstar, t_n, T_nm1, T_n = platooninfo[i][0:4] #get times for current vehicle 
#        leadlist = sim[i][t_n-t_nstar:T_nm1-t_nstar+1,4] #this gets the leaders for each timestep of the current vehicle\
#        curlead = leadlist[0] #initialize current leader
#        curleadinfo.append([curlead, t_n]) #initialization 
#        for j in range(len(leadlist)):
#            if leadlist[j] != curlead: #if there is a new leader
#                newlead = leadlist[j]
#                oldlead = curlead
#                ##############relaxation constant calculation
#                newt_nstar = platooninfo[newlead][0]
#                oldt_nstar = platooninfo[oldlead][0]
#                olds = sim[oldlead][t_n+j-1-oldt_nstar,2] - sim[oldlead][0,6] - sim[i][t_n+j-1-t_nstar,2] #the time is t_n+j-1; this is the headway
#                news = sim[newlead][t_n+j-newt_nstar,2] - sim[newlead][0,6] - sim[i][t_n+j-t_nstar,2] #the time is t_n+j
#                #below if only adds if headway decreases, otherwise we will always add the relaxation constant, even if it is negative. 
##                if news < olds: #if the headway decreases, then we will add in the relaxation phenomenon
##                    currinfo.append([t_n+j, olds-news]) #we append the time the LC happens (t_n+j), and the "gamma" which is what I'm calling the initial constant we adjust the headway by (olds-news)
#                currinfo.append([t_n+j,olds-news])
#                
#                if sim[i][t_n+j-t_nstar,7]==merge_lane and sim[i][t_n+j-1-t_nstar,7]==merge_from_lane: #if the lane change is a merger
#                    currinfo.pop(-1) #remove that entry
#                
#                #################################################
#                curlead = leadlist[j] #update the current leader
#                curleadinfo[-1].append(t_n+j-1) #last time (in frameID) the old leader is observed
#                curleadinfo.append([curlead,t_n+j]) #new leader and the first time (in frameID) it is observed.
#                
#        curleadinfo[-1].append(t_n+len(leadlist)-1) #termination
#        
#        
#        
#        #do essentially the same things for followers now (we need the follower for adjoint system)
#        #only difference is that we only need to put things in if their follower is in platoons
#        follist = sim[i][t_n-t_nstar:T_n-t_nstar+1,5] #list of followers
#        curfol = follist[0]
#        if curfol in platoons: #if the current follower is in platoons we initialize
#            curfolinfo.append([curfol,t_n])
#        for j in range(len(follist)): #check what we just made to see if we need to put stuff in folinfo
#            if follist[j] != curfol: #if there is a new follower
#                curfol = follist[j]
#                if curfolinfo != []: #if there is anything in curfolinfo
#                    curfolinfo[-1].append(t_n+j-1) #we finish the interval
#                if curfol in platoons: #if new follower is in platoons
#                    curfolinfo.append([curfol,t_n+j]) #start the next interval
#        if curfolinfo != []: #if there is anything to finish
#            curfolinfo[-1].append(t_n+len(follist)-1) #finish it 
#        leadinfo.append(curleadinfo) #append what we just made to the total list 
#        folinfo.append(curfolinfo) 
#        rinfo.append(currinfo)
#        
#    if use_merge_constant: 
#        rinfo = merge_rconstant(platoons,platooninfo,sim,leadinfo,rinfo,200)
#                
#        
#    return leadinfo, folinfo, rinfo 




