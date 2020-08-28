
"""
@author: rlk268@cornell.edu
"""
#%%
# training, testing, maxhd, maxv, mina, maxa = make_dataset(meas, platooninfo)
# model = RNNCFModel(maxhd, maxv, mina, maxa)
# loss = masked_MSE_loss
# opt = tf.keras.optimizers.Adam(learning_rate = .001)

# training_loop(model, loss, opt, training, nbatches = 1000, nveh = 32, nt = 100, lstm_units = 20)
# training_loop(model, loss, opt, training, nbatches = 1000, nveh = 32, nt = 200, lstm_units = 20)
# training_loop(model, loss, opt, training, nbatches = 1000, nveh = 32, nt = 300, lstm_units = 20)
# training_loop(model, loss, opt, training, nbatches = 1000, nveh = 32, nt = 500, lstm_units = 20)

# model.save_weights('trained LSTM')
# #%%
# out = generate_trajectories(model, list(testing.keys()), testing, loss = loss)
# out2 = generate_trajectories(model, list(training.keys()), training, loss = loss)

#%%

res_list = make_into_analyze_res_format(out, list(testing.keys()), testing)
test_out = analyze_res_NN(res_list, meas, platooninfo, .1)


#%%
import havsim.calibration.helper as helper
import math
import numpy as np

def make_into_analyze_res_format(out, vehlist, ds, dt = .1):
    # want dictionary with values as a dict with keys 'posmem' 'speedmem'
    res_list = {}
    for count, veh in enumerate(vehlist):
        traj_len = ds[veh]['times'][1] - ds[veh]['times'][0]+1
        res_list[veh] = {'posmem':None, 'speedmem':None}
        temp = list(out[0][count,:traj_len])
        res_list[veh]['posmem'] = temp

        try:
            res_list[veh]['speedmem'] = [(temp[i+1] - temp[i])/(dt) for i in range(traj_len-1)]
        except:
            print('hello')
        res_list[veh]['speedmem'].append(float(out[1][count]))

    return res_list


def analyze_res_NN(res_list, meas, platooninfo, dt, mergeind = math.inf, times = 100, realistic = 1.1):
    out = {'overall mse':None, 'mse near LC':None, 'mse for merges':None, 'mse for many LC':None, 'realistic acc':None, 'short lc':None}
    temp = [ga['fun']/(3.28084**2) for ga in res_list]
    out['overall mse'] = (np.mean(temp), np.median(temp), np.std(temp))

    mse = []
    nearlc = []
    merge = []
    manylc = []
    real = []
    out2 = []
    shortlc = []
    for count, item in enumerate(res_list.items()):
        veh, res = item

        leadinfo = helper.makeleadinfo([veh], platooninfo, meas)
        lctimes = [cur[1] for cur in leadinfo[0]]
        rinfo = helper.makerinfo([veh], platooninfo, meas, leadinfo)
        if len(lctimes) > len(rinfo[0]):  # this handles an edge case
            rinfo[0].insert(0, [None, -1])
        t_nstar, t_n, T_nm1 = platooninfo[veh][:3]
        if lctimes[0] == t_nstar:
            lctimes = lctimes[1:]

        curmeas = meas[veh][t_n-t_nstar:T_nm1+1-t_nstar,2]
        cursim = np.array(res['posmem'])
        curmse = np.mean(np.square(curmeas-cursim))/(3.28084**2)

        mse.append(curmse)
        if len(leadinfo[0]) > 3:
            manylc.append(curmse)  # many lc
        if count >= mergeind:
            merge.append(curmse)
        for count, curtime in enumerate(lctimes):
            curmse = np.square(cursim[curtime-t_n:curtime-t_n+times] - curmeas[curtime-t_n:curtime-t_n+times])
            curmse = np.sum(curmse)/len(curmse)/(3.28084**2)
            nearlc.append(curmse)  # near lc
            # try:
            curg = rinfo[0][count][1]
            if curg > 0:
                shortlc.append(curmse)
            # except:
            #     print('hello')

        simacc = [(cursim[i+2] - 2*cursim[i+1] + cursim[i])/(dt**2) for i in range(len(cursim)-2)]
        measacc = [(curmeas[i+2] - 2*curmeas[i+1] + curmeas[i])/(dt**2) for i in range(len(curmeas)-2)]
        upper = max(realistic*max(measacc), 4*3.28)
        lower = min(-6*3.28, realistic*min(measacc))
        temp1, temp2 = max(simacc), min(simacc)
        if temp1 < upper and temp2 > lower:
            real.append(1)
        else:
            real.append(0)
            out2.append(((upper, lower), (temp1, temp2)))

    out['mse'] = (np.mean(mse), np.median(mse), np.std(mse))
    out['mse near LC'] = (np.mean(nearlc), np.median(nearlc), np.std(nearlc))
    out['mse for merges'] = (np.mean(merge), np.median(merge), np.std(merge))
    out['mse for many LC'] = (np.mean(manylc), np.median(manylc), np.std(manylc))
    out['realistic acc'] = (np.mean(real))
    out['short lc'] = (np.mean(shortlc), np.median(shortlc), np.std(shortlc))
    print(out)
    return out, out2