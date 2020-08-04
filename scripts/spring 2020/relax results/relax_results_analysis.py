
# Also need to run first cell of relax_results_cf_calibration file
import havsim.simulation.calibration as hc
import havsim.simulation.calibration_models as hm
import math
import havsim.calibration.helper as helper
import numpy as np
import pickle

def analyze_res(res_list, veh_list, meas, platooninfo, dt, kwargs, mergeind = math.inf, times = 100, realistic = 2):
    out = {'overall mse':None, 'mse near LC':None, 'mse for merges':None, 'mse for many LC':None, 'realistic acc':None}
    temp = [ga['fun']/(3.28084**2) for ga in res_list]
    out['overall mse'] = (np.mean(temp), np.median(temp), np.std(temp))

    nearlc = []
    merge = []
    manylc = []
    real = []
    out2 = []
    for count, veh in enumerate(veh_list):
        res = res_list[count]
        if count >= mergeind:
            merge.append(res['fun']/(3.28084**2))  # merge
        cal = hc.make_calibration([veh], meas, platooninfo, dt, **kwargs)
        cal.simulate(res['x'])

        leadinfo = helper.makeleadinfo([veh], platooninfo, meas)
        lctimes = [cur[1] for cur in leadinfo[0]]
        t_nstar, t_n, T_nm1 = platooninfo[veh][:3]
        if lctimes[0] == t_nstar:
            lctimes = lctimes[1:]

        if len(leadinfo[0]) > 3:
            manylc.append(res['fun']/(3.28084**2))  # many lc
        curmeas = meas[veh][t_n-t_nstar:T_nm1+1-t_nstar,2]
        cursim = np.array(cal.all_vehicles[0].posmem)
        for curtime in lctimes:
            curmse = np.square(cursim[curtime-t_n:curtime-t_n+times] - curmeas[curtime-t_n:curtime-t_n+times])
            curmse = np.sum(curmse)/len(curmse)/(3.28084**2)
            nearlc.append(curmse)  # near lc

        simacc = [(cursim[i+2] - 2*cursim[i+1] + cursim[i])/(dt**2) for i in range(len(cursim)-2)]
        measacc = [(curmeas[i+2] - 2*curmeas[i+1] + curmeas[i])/(dt**2) for i in range(len(curmeas)-2)]
        temp1 = max(max(simacc), abs(min(simacc)))
        temp2 = max(max(measacc), abs(min(measacc)))
        if temp1 > 1.25*temp2:
            real.append(0)
        else:
            real.append(1)
        # out2.append(temp1/temp2)
    out['mse near LC'] = (np.mean(nearlc), np.median(nearlc), np.std(nearlc))
    out['mse for merges'] = (np.mean(merge), np.median(merge), np.std(merge))
    out['mse for many LC'] = (np.mean(manylc), np.median(manylc), np.std(manylc))
    out['realistic acc'] = (np.mean(real), np.median(real), np.std(real))
    print(out)
    return out

lclist = lc_list.copy()
lclist.extend(merge_list)
mergeind = len(lc_list)
#%%
with open('IDMrelax.pkl', 'rb') as f:
    res1, res2 = pickle.load(f)
res1.extend(res2)
kwargs = {}
idm1 = analyze_res(res1, lclist, meas, platooninfo, .1, kwargs, mergeind = mergeind)

with open('IDMnorelax.pkl', 'rb') as f:
    res1, res2, res3 = pickle.load(f)
res1.extend(res2)
kwargs = {'vehicle_class':NoRelaxIDM}

idm2 = analyze_res(res1, lclist, meas, platooninfo, .1, kwargs, mergeind = mergeind)
idm3 = analyze_res(res3, nolc_list, meas, platooninfo, .1, kwargs)

#%%
with open('OVMrelax.pkl', 'rb') as f:
    res1, res2 = pickle.load(f)
res1.extend(res2)
kwargs = {'vehicle_class':hm.OVMCalibrationVehicle}
ovm1 = analyze_res(res1, lclist, meas, platooninfo, .1, kwargs, mergeind = mergeind)

with open('OVMnorelax.pkl', 'rb') as f:
    res1, res2, res3 = pickle.load(f)
res1.extend(res2)
kwargs = {'vehicle_class':NoRelaxOVM}

ovm2 = analyze_res(res1, lclist, meas, platooninfo, .1, kwargs, mergeind = mergeind)
ovm3 = analyze_res(res3, nolc_list, meas, platooninfo, .1, kwargs)

#%%
with open('Newellrelax.pkl', 'rb') as f:
    res1, res2 = pickle.load(f)
res1.extend(res2)
kwargs = {'vehicle_class':hm.NewellCalibrationVehicle}
n1 = analyze_res(res1, lclist, meas, platooninfo, .1, kwargs, mergeind = mergeind)

with open('Newellnorelax.pkl', 'rb') as f:
    res1, res2, res3 = pickle.load(f)
res1.extend(res2)
kwargs = {'vehicle_class':NoRelaxNewell}

n2 = analyze_res(res1, lclist, meas, platooninfo, .1, kwargs, mergeind = mergeind)
n3 = analyze_res(res3, nolc_list, meas, platooninfo, .1, kwargs)

#%%
with open('2pIDM.pkl', 'rb') as f:
    res1, res2 = pickle.load(f)
res1.extend(res2)
kwargs = {'vehicle_class':hm.Relax2IDM}
idm4 = analyze_res(res1, lclist, meas, platooninfo, .1, kwargs, mergeind = mergeind)

with open('2psrelax.pkl', 'rb') as f:
    res1, res2 = pickle.load(f)
res1.extend(res2)
kwargs = {'vehicle_class':hm.RelaxShapeIDM}
idm5 = analyze_res(res1, lclist, meas, platooninfo, .1, kwargs, mergeind = mergeind)

with open('ExpIDM.pkl', 'rb') as f:
    res1, res2 = pickle.load(f)
res1.extend(res2)
kwargs = {'vehicle_class':hm.RelaxExpIDM}
idm6 = analyze_res(res1, lclist, meas, platooninfo, .1, kwargs, mergeind = mergeind)

with open('NewellLL.pkl', 'rb') as f:
    res1, res2 = pickle.load(f)
res1.extend(res2)
kwargs = {'vehicle_class': hm.NewellLL, 'event_maker':hm.make_ll_lc_event, 'lc_event_fun':hm.ll_lc_event}
n4 = analyze_res(res1, lclist, meas, platooninfo, .1, kwargs, mergeind = mergeind)

with open('SKArelax.pkl', 'rb') as f:
    res1, res2 = pickle.load(f)
res1.extend(res2)
kwargs = {'vehicle_class': hm.SKA_IDM}
idm7 = analyze_res(res1, lclist, meas, platooninfo, .1, kwargs, mergeind = mergeind)
#%%
with open('2pvhdIDM.pkl', 'rb') as f:
    res1, res2 = pickle.load(f)
res1.extend(res2)
kwargs = {'vehicle_class': hm.Relax2vhdIDM}
idm8 = analyze_res(res1, lclist, meas, platooninfo, .1, kwargs, mergeind = mergeind)
