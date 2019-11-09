
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
    #IDM with bounded velocity 

    outdx = vehdx
    outddx = p[3]*(1-(vehdx/p[0])**4-((p[2]+vehdx*p[1]+(vehdx*(vehdx-leaddx))/(2*(p[3]*p[4])**(1/2)))/(leadx-leadlen-vehx))**2)
    
#    

    if vehdx+dt*outddx < 0:
        outddx = -vehdx/dt
    
    return outdx, outddx