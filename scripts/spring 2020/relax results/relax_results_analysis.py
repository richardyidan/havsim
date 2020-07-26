#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 07:06:44 2020

@author: rlk268
"""

relaxacc, relaxdec, norelaxacc, norelaxdec = 0, 0, 0, 0
relaxjerk, norelaxjerk = [], []
for count, vehid in enumerate(lc_list[30:35]): 
    cal = hc.make_calibration([vehid], meas, platooninfo, .1, hc.CalibrationVehicle)
    cal2 = hc.make_calibration([vehid], meas, platooninfo, .1, hc.CalibrationVehicle)
    parameters = relax_lc_res[count][0]
    cal.simulate(parameters)
    relaxveh = cal.all_vehicles[0]
    
    
    parameters = list(norelax_lc_res[count][0])
    parameters.append(.1)
    cal2.simulate(parameters)
    norelaxveh = cal2.all_vehicles[0]
    
    acc, acc2 = [], []
    for i in range(len(relaxveh.speedmem)-1):
        acc.append((relaxveh.speedmem[i+1]-relaxveh.speedmem[i])/.1)
        acc2.append((norelaxveh.speedmem[i+1]-norelaxveh.speedmem[i])/.1)
    if max(acc)>10:
        relaxacc += 1
    if min(acc)<-7*3.3:
        relaxdec += 1
    if max(acc2) > 10:
        norelaxacc += 1
    if min(acc2) < -7*3.3:
        norelaxdec += 1
    plt.figure()
    plt.plot(acc)
    plt.plot(acc2)
    jerk, jerk2 = [], []
    for i in range(len(acc)-1):
        jerk.append((acc[i+1]-acc[i])/.1)
        jerk2.append((acc2[i+1]-acc2[i])/.1)
    relaxjerk.append(max(max(jerk), -min(jerk)))
    norelaxjerk.append(max(max(jerk2), -min(jerk2)))
    plt.figure()
    plt.plot(jerk)
    plt.plot(jerk2)
    
    
    