
"""
@author: rlk268@cornell.edu
debugs recent fixes - 
first, that the new platoon format isn't messing anything up
second, that the bug fix with multiple followers is working

also gets the the platoons to use for the last part of the adjoint paper

"""
from havsim.calibration.calibration import calibrate_tnc2, calibrate_GA

#pguess = [20*3.3,.086/3.3/2, 1.545, .5, .175, 60 ] #this seems like a very good second guess
pguess = [10*3.3,.086/3.3, 1.545, 2, .175, 5 ] #original guess
#pguess = [10*3.3,.086/3.3/2, .5, .5, .175, 60 ]

mybounds = [(20,120),(.001,.1),(.1,2),(.1,5),(0,3), (.1,75)]

plist = [[10*3.3,.086/3.3, 1.545, 2, .175, 5 ], [20*3.3,.086/3.3/2, 1.545, .5, .175, 60 ], [10*3.3,.086/3.3/2, .5, .5, .175, 60 ]]
bounds = [(20,120),(.001,.1),(.1,2),(.1,5),(0,3), (.1,75)]
args = (True, 6)

lists = [[381.0, 391.0], [335.0, 326.0, 334.0]]
#lists = [[381], [391], [335],[326],[334]]

out = calibrate_tnc2(plist,bounds,meas,platooninfo,lists,makeleadfolinfo,platoonobjfn_objder,None,OVM,OVMadjsys,OVMadj,True,6,cutoff=0,cutoff2=4.5,order=1,budget = 3)

#%% just want to check this test platoon see what's going on 
#335, 326, 334 shows the type of complex behavior that you have a hard time describing without a complciated LC model
from havsim.plotting import * 
animatetraj(meas,platooninfo, [381,391,335,326,334])
#platoonplot(meas, None, platooninfo, [381,391,335,326,334], lane=None, colorcode = True)

#%% get test platoons to use 
from havsim.calibration.algs import sortveh3
vehIDs = np.unique(data[data[:,7]==2,0])
sortedvehID2 = sortveh3(vehIDs,2,meas,platooninfo)
sortedvehID = sortedvehID2[20:]
platlist = [[i] for i in sortedvehID]
#%% #get vehicles to use run above cell then this one 
from havsim.calibration.helper import makeleadinfo, makefolinfo
def cirdep_metric(platoonlist, platooninfo, k = .9, type = 'veh'):
    if type == 'veh':
        cirList = []
        after = set([])
        for i in range(len(platoonlist)):
            after.update(platoonlist[i])
        for i in range(len(platoonlist)):
            after -= set(platoonlist[i])
            leadinfo, folinfo, = makeleadinfo(platoonlist[i], platooninfo, meas), makefolinfo(platoonlist[i],platooninfo,meas)
            for j in range(len(platoonlist[i])):
                leaders = [k[0] for k in leadinfo[j]]
                leaders = set(leaders)
                if len(leaders.intersection(after))>0:
                    cirList.append((platoonlist[i][j], i))
        return cirList
    elif type == 'num':
        res = 0
        cirList = []
        after = set([])
        for i in range(len(platoonlist)):
            after.update(platoonlist[i])
        for i in range(len(platoonlist)):
            after -= set(platoonlist[i])
            leadinfo, folinfo, unused = makeleadfolinfo(platoonlist[i], platooninfo, meas)
            for j in range(len(platoonlist[i])):
                leaders = [k[0] for k in leadinfo[j]]
                leaders = set(leaders)
                if len(leaders.intersection(after)) > 0:
                    cirList.append((platoonlist[i][j], i))
        for i in cirList:
            T = set(range(platooninfo[i[0]][1], platooninfo[i[0]][2]))
            res += c_metric(i[0], platoonlist[i[1]], T, platooninfo, k=k, type='follower')
        return res
    
res = cirdep_metric(platlist, platooninfo)
for i in res: 
    platlist.remove([i[0]])
usevehlist = [i[0] for i in platlist]
usevehlist = usevehlist[:100]

#%% redo adjoint platoon size experiment
from havsim.calibration.calibration import calibrate_tnc2, calibrate_GA
from havsim.calibration.helper import makeleadfolinfo
from havsim.calibration.models import OVM, OVMadjsys, OVMadj
from havsim.calibration.opt import platoonobjfn_obj, platoonobjfn_objder
import pickle
import math 
def platoontest(vehlist, meas, platooninfo):
    def pltn_helper(vehlist,size):
        out = []
        length =len(vehlist)
        num_full = math.floor(length/size)
        leftover = length-num_full*size
        for i in range(num_full): #add full platoons
    #        curplatoon = [[],vehlist[i:i+size]]
            curplatoon = []
            usei = i*size
            for j in range(size): 
                curplatoon.append(vehlist[usei+j])
            out.append(curplatoon)
        i = num_full*size
        if leftover >0:
            temp = []
            for j in range(leftover):
                temp.append(vehlist[i+j])
            out.append(temp)
        return out
    #pltn = [[[],i] for i in followerchain.keys()]
    maxsize = 10
    lists = [] #lists we will test 
    for i in range(maxsize):
        lists.append(pltn_helper(vehlist,i+1))
        
    plist = [[10*3.3,.086/3.3, 1.545, 2, .175, 5 ], [20*3.3,.086/3.3/2, 1.545, .5, .175, 60 ], [10*3.3,.086/3.3/2, .5, .5, .175, 60 ]]
    bounds = [(20,120),(.001,.1),(.1,2),(.1,5),(0,3), (.1,75)]
    
    output = []
    output2 = []
    
    for i in range(3): #change this part here depending on whether you want to show that these scale awfully or not 
        out2 = calibrate_GA(bounds,meas,platooninfo,lists[i],makeleadfolinfo,platoonobjfn_obj,None,OVM,OVMadjsys,OVMadj,True,6,order=1)
        output2.append(out2)
    
    for i in range(maxsize):
        out = calibrate_tnc2(plist,bounds,meas,platooninfo,lists[i],makeleadfolinfo,platoonobjfn_objder,None,OVM,OVMadjsys,OVMadj,True,6,cutoff=0,cutoff2=5.5,order=1,budget = 3)
        output.append(out)
    return output, output2, lists

out, out2, lists = platoontest(usevehlist,meas,platooninfo)

with open('/home/rlk268/data/pickle/plattest.pkl', 'wb') as f:
    pickle.dump([out, out2, lists],f)
