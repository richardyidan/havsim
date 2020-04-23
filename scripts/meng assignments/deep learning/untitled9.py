
"""
@author: rlk268@cornell.edu
"""

badvehlist = []
for i in sim_info.keys(): 
    if sim_info[i][0] > 30:
        badvehlist.append(i)

def helpmeplot(veh):
    plt.figure()
    t_nstar, t_n, T_nm1, = platooninfo[veh][:3]
    plt.plot(sim[veh][:,0])
    plt.plot(out[1][veh][t_n-t_nstar:T_nm1-t_nstar,2])
    plt.plot(meas[veh][t_n-t_nstar:T_nm1-t_nstar,2])
    plt.legend(['NN','benchmark','measurements'])
    
    
    plt.figure()
#    t_nstar, t_n, T_nm1, = platooninfo[veh][:3]
    plt.plot(sim[veh][:,1])
    plt.plot(out[1][veh][t_n-t_nstar:T_nm1-t_nstar,3])
    plt.plot(meas[veh][t_n-t_nstar:T_nm1-t_nstar,3])
    plt.legend(['NN','benchmark','measurements'])
    
for i in [31, 46, 75, 88, 99]:
    out = benchmark(meas,platooninfo, [[badvehlist[i]]], usemodel = 'OVM')
    helpmeplot(badvehlist[i])