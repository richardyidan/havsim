
"""
@author: rlk268@cornell.edu

houses all the different models for simulation

for models, we want to create some sort of way to modularly build models. For example,
circular/straight roads have different headway calculations, or optionally add or exclude things like 
bounds on velocity/acceleration, or extra regimes, or have something like 
random noise added or not. 

Something like a wrapper function which can accept different parts and combine them into one single thing

models should have the following call signature: 
    veh - list of state of vehicle model is being applied to
    lead - list of state for vehicle's leader
    p - parameters
    leadlen - length of lead vehicle
    *args - additional inputs should be stored in modelinfo dict, and passed through *args
    dt = .1 - timestep 
    
    they return the derivative of the state i.e. how to update in the next timestep 
"""

import numpy as np 
import scipy.optimize as sc 

def IDM_b3(p, veh, lead, *args,dt=.1):
    #state is defined as [x,v,s] triples
    #IDM with bounded velocity for circular road
    
    #old code 
#    s = lead[0]-leadlen-veh[0]
#    if s < 0: #wrap around in circular can cause negative headway values; in this case we add an extra L to headway
#        s = s + args[0]
    #check if need to modify the headway 
#    if args[1]:
#        s = s + args[0]

    outdx = veh[1]
    outddx = p[3]*(1-(veh[1]/p[0])**4-((p[2]+veh[1]*p[1]+(veh[1]*(veh[1]-lead[1]))/(2*(p[3]*p[4])**(1/2)))/(veh[2]))**2)
    
    if veh[1]+dt*outddx < 0:
        outddx = -veh[1]/dt
    
    return [outdx, outddx]

def IDM_b3_eql(p, s, v, find = 's', maxs = 1e4):
    #finds equilibrium solution for s or v, given the other
    
    #find = s - finds equilibrium headway (s) given speed v, 
    #find = v - finds equilibrium velocity (v) given s 
    
    if find == 's':
        s = ((p[2]+p[1]*v)**2/(1- (v/p[0])**4))**.5
        return s 
    if find == 'v':
        eqlfun = lambda x: ((p[2]+p[1]*x)**2/(1- (x/p[0])**4))**.5 - s
        v = sc.bisect(eqlfun, 0, maxs)
        return v
    
    
def sv_obj(sim, auxinfo):
    #maximize squared velocity = sv 
    obj = 0 
    for i in sim.keys(): 
        for j in sim[i]: #squared velocity 
            obj = obj - j[1]**2
            if j[2] < .2:
                obj = obj + 2**(-5*(j[2]-.2)) - 1
    return obj 

#def sv_obj(sim, auxinfo, cons = 1e-4):
#    #maximize squared velocity = sv 
#    obj = 0 
#    for i in sim.keys(): 
#        for j in sim[i]: #squared velocity 
#            obj = obj - j[1]**2
#    obj = obj * cons
#    for i in sim.keys():
#        for j in range(len(sim[i])): #penality for collisions
#            lead = auxinfo[i][1]
#            leadx = sim[lead][j][0]
#            leadlen = auxinfo[lead][0]
#            s = leadx - leadlen - sim[i][j][0]
#            if s < .2:
#                obj = obj + 2**(-5*(s-.2)) - 1
#    return obj 

"""
in general, I think it is better to just manually solve for the equilibrium solution when possible
instead of using root finding on the model naively. 
I also think eql might be better if it uses bisection instead of newton since bisection 
is more robust, and assuming we are just looking for headway or velocity
we can give bracketing bounds based on intuition
"""
def eql(model, v, p, length, tol=1e-4): 
    #finds equilibrium headway for a given speed and parameters value for second order model 
    #there should only be a single root 
    def wrapperfun(x):
        dx, ddx = model(0,v,x,v,p,length)
        return ddx
    
    guess = 10
    headway = 0
    try:
#        headway = sc.newton(wrapperfun, x0=guess,maxiter = 50) #note might want to switch newton to bisection with arbitrarily large headway 
        headway = sc.bisect(wrapperfun, 0, 1e4)
    except RuntimeError: 
        pass
    counter = 0
    
    while abs(wrapperfun(headway)) > tol and counter < 20: 
        guess = guess + 10
        try:
            headway = sc.newton(wrapperfun, x0=guess,maxiter = 50)
        except RuntimeError: 
            pass
        counter = counter + 1
        
        testout = model(0,v,headway,v,p,length)
        if testout[1] ==0:
            return headway 
        
    return headway

def noloss(*args):
    pass

