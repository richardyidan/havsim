
"""
@author: rlk268@cornell.edu
"""

# categorize vehicles
veh_list = meas.keys()
merge_list = []
lc_list = []
nolc_list = []
for veh in veh_list:
    t_nstar, t_n = platooninfo[veh][0:2]
    if t_n > t_nstar and meas[veh][t_n-t_nstar-1,7]==7 and meas[veh][t_n-t_nstar,7]==6:
        merge_list.append(veh)
    elif len(platooninfo[veh][4]) > 1:
        lc_list.append(veh)
    else:
        nolc_list.append(veh)
