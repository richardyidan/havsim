
"""
@author: rlk268@cornell.edu
"""


from havsim.calibration.calibration import calibrate_tnc2, calibrate_GA
from havsim.calibration.helper import makeleadfolinfo, obj_helper
from havsim.calibration.models import OVM, OVMadjsys, OVMadj, IDM_b3, IDMadjsys_b3, IDMadj_b3
from havsim.calibration.opt import platoonobjfn_obj, platoonobjfn_objder
#simulation 
from havsim.simulation.simulation import eq_circular, simulate_cir, update2nd_cir, update_cir
from havsim.simulation.models import IDM_b3, IDM_b3_eql
#plotting 
from havsim.plotting import  platoonplot, plotflows, plotvhd, animatevhd_list, animatetraj, meanspeedplot, optplot, selectoscillation, plotformat, selectvehID
#data processing
from havsim.calibration.algs import makeplatoonlist



def OVMbenchmark(meas, platooninfo, platoonlist, budget = 1):
    plist = [[10*3.3,.086/3.3, 1.545, 2, .175, 5 ],[20*3.3,.086/3.3/2, 1.545, .5, .175, 60 ], [10*3.3,.086/3.3/2, .5, .5, .175, 60 ]]
    bounds = [(20,120),(.001,.1),(.1,2),(.1,5),(0,3), (.1,75)]
    out, unused, rmse = calibrate_tnc2(plist, bounds, meas, platooninfo, platoonlist, makeleadfolinfo, platoonobjfn_objder, None, OVM, OVMadjsys, OVMadj, True, 6, order = 1)
    
    
    return out, rmse


def IDMbenchmark(meas, platooninfo, platoonlist, budget = 1):
    plist = [[10*3.3,.086/3.3, 1.545, 2, .175, 5 ],[20*3.3,.086/3.3/2, 1.545, .5, .175, 60 ], [10*3.3,.086/3.3/2, .5, .5, .175, 60 ]]
    bounds = [(20,120),(.001,.1),(.1,2),(.1,5),(0,3), (.1,75)]
    out, unused, rmse = calibrate_tnc2(plist, bounds, meas, platooninfo, platoonlist, makeleadfolinfo, platoonobjfn_objder, None, IDM_b3, IDMadjsys_b3, IDMadj_b3, True, 6, order = 1)
    return out, rmse

testplatoon = [[108,112]]
out, rmse = OVMbenchmark(meas, platooninfo, testplatoon)
sim = copy.deepcopy(meas)
sim = obj_helper(out,OVM, OVMadjsys, OVMadj, meas, sim, platooninfo, testplatoon, makeleadfolinfo, platoonobjfn_obj,(True,6))
optplot((out, 0, rmse), meas, sim, platooninfo, testplatoon, OVM, OVMadjsys, OVMadj, makeleadfolinfo, platoonobjfn_obj, (True,6))