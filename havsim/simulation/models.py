
"""
@author: rlk268@cornell.edu

houses all the different models for simulation

for models, we want to create some sort of way to modularly build models. For example,
circular/straight roads have different headway calculations, or optionally add or exclude things like 
bounds on velocity/acceleration, or extra regimes, or have something like 
random noise added or not. 

Something like a wrapper function which can accept different parts and combine them into one single thing
"""

import numpy as np 
import scipy.optimize as sc 

def IDM_b3(veh, lead, p,leadlen, *args,dt=.1):
    #IDM with bounded velocity for circular road
    
    s = lead[0]-leadlen-veh[0]
    
#    if s < 0: #wrap around in circular can cause negative headway values; in this case we add an extra L to headway
#        s = s + args[0]
    
    #check if need to modify the headway 
    if args[1]:
        s = s + args[0]

    outdx = veh[1]
    outddx = p[3]*(1-(veh[1]/p[0])**4-((p[2]+veh[1]*p[1]+(veh[1]*(veh[1]-lead[1]))/(2*(p[3]*p[4])**(1/2)))/(s))**2)
    
    if veh[1]+dt*outddx < 0:
        outddx = -veh[1]/dt
    
    return [outdx, outddx]