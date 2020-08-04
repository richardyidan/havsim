
# Also need to run first cell of relax_results_cf_calibration file
import havsim.simulation.calibration as hc
import havsim.simulation.calibration_models as hm
import math
import havsim.calibration.helper as helper
import numpy as np

def analyze_res(res_list, veh_list, meas, platooninfo, dt, kwargs, mergeind = math.inf, times = 100, realistic = 2):
    out = {'overall mse':None, 'mse near LC':None, 'mse for merges':None, 'mse for many LC':None, 'realistic acc':None}
    temp = [ga['fun']/(3.28084**2) for ga in res_list]
    out['overall mse'] = (np.mean(temp), np.median(temp), np.std(temp))

    nearlc = []
    merge = []
    manylc = []
    real = []
    for count, veh in enumerate(veh_list):
        res = res_list[count]
        if count >= mergeind:
            merge.append(res['fun']/(3.28084**2))  # merge
        cal = hc.make_calibration([veh], meas, platooninfo, dt, **kwargs)
        cal.simulate(res['x'])

        leadinfo = helper.makeleadinfo([veh], platooninfo, meas)
        lctimes = [cur[1] for cur in leadinfo]
        t_nstar, t_n, T_nm1 = platooninfo[veh][:3]

        if len(leadinfo) > 3:
            manylc.append(res['fun']/(3.28084**2))  # many lc
        curmeas = meas[veh][t_n-t_nstar:T_nm1+1-t_nstar,2]
        cursim = np.array(cal.all_vehicles[0].posmem)
        for curtime in lctimes:
            curmse = np.square(cursim[curtime:curtime+times] - curmeas[curtime:curtime+times])
            curmse = np.sum(curmse)/len(curmse)/(3.28084**2)
            nearlc.append(curmse)  # near lc

        simacc = [(cursim[i+2] - 2*cursim[i+1] + cursim[i])/(dt**2) for i in range(len(cursim)-2)]
        measacc = [(curmeas[i+2] - 2*curmeas[i+1] + curmeas[i])/(dt**2) for i in range(len(curmeas)-2)]
        if max(simacc) > 2*max(measacc):
            real.append(0)
        else:
            real.append(1)
    out['mse near LC'] = (np.mean(nearlc), np.median(nearlc), np.std(nearlc))
    out['mse for merges'] = (np.mean(merge), np.median(merge), np.std(merge))
    out['mse for many LC'] = (np.mean(manylc), np.median(manylc), np.std(manylc))
    out['realistic acc'] = (np.mean(real), np.median(real), np.std(real))
    print(out)
    return out

#%% Run above




