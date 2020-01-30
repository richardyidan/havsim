
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

    outdx = veh[1]
    outddx = p[3]*(1-(veh[1]/p[0])**4-((p[2]+veh[1]*p[1]+(veh[1]*(veh[1]-lead[1]))/(2*(p[3]*p[4])**(1/2)))/(veh[2]))**2)
    
    if veh[1]+dt*outddx < 0:
        outddx = -veh[1]/dt
    
    return [outdx, outddx]

def IDM_b3_b(p, veh, lead, *args,dt=.1):
    #state is defined as [x,v,s] triples
    #IDM with bounded velocity and acceleration

    outdx = veh[1]
    outddx = p[3]*(1-(veh[1]/p[0])**4-((p[2]+veh[1]*p[1]+(veh[1]*(veh[1]-lead[1]))/(2*(p[3]*p[4])**(1/2)))/(veh[2]))**2)
    
    #bounded acceleration in m/s/s units
    if outddx > 3:
        outddx = 3
    elif outddx < -7: 
        outddx = -7
    
    if veh[1]+dt*outddx < 0:
        outddx = -veh[1]/dt
    
    
    return [outdx, outddx]

def IDM_b3_sh(p, veh, lead, *args,dt=.1):
    #state is defined as [x,v,s] triples
    #IDM with bounded velocity for circular road
    #we add a speed harmonization thing that says you can't go above w/e
    
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
    
    if veh[1] + dt*outddx > p[5]:
        outddx = (p[5] - veh[1])/dt
        
    
    return [outdx, outddx]

def mobil_dis(plc, p, pfold, pfnew, placeholder): #discretionary lane changing following mobil model 
    pass

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
    
def FS(p, veh, lead, *args, dt = .1):
    #follower stopper control model 
    #7 parameters, we have U (desired velocity) as a parameter
    dv = (lead[1] - veh[1])
    if dv > 0:
        dv = 0
    dv = dv**2
    usev = min(lead[1],p[6])
    #determine model regime
    dx1 = p[3] + .5/p[0]*dv
    if veh[2] < dx1:
        vcmd = 0
    else:
        dx2 = p[4] + .5/p[1]*dv
        if veh[2] < dx2:
            vcmd = usev*(veh[2]-dx1)/(dx2-dx1)
        else: 
            dx3 = p[5] + .5/p[2]*dv
            if veh[2] < dx3:
                vcmd = usev + (p[6] - usev)*(veh[2]-dx2)/(dx3-dx2)
            else:
                vcmd = p[6]
    
    outdx = veh[1]
    outddx = p[7]*(vcmd - veh[1])
    
    if outddx > 2:
        outddx = 2
    elif outddx < -5:
        outddx = -5
    
    return [outdx, outddx]

def linearCAV(p,veh,lead,*args, dt = .1):
    #p[0] = jam distance
    #p[1] = headway slope
    #p[2] = max speed
    #p[3] = sensitivity parameter for headway feedback
    #p[4] - initial beta value - sensitivity for velocity feedback
    #p[5] - short headway 
    #p[6] -large headway
    
    Vs = p[1]*(veh[2]-p[0]) #velocity based on triangular FD
    #determine model regime
    if Vs > p[2]:
        Vs = p[2]
    if lead[1] > p[2]:
        lv = p[2]
    else:
        lv = lead[1]
    if veh[2] > p[6]: 
        A = 1
        B = 0
    else:
        A = p[3]
        if veh[2] > p[5]:
            B = (1 - (veh[2]-p[5])/(p[6]-p[5]))*p[4]
        else:
            B = p[4]
    
    outdx = veh[1]
    outddx = A*(Vs - veh[1]) + B*(lv - veh[1])
    
    return [outdx, outddx]

def PIS(p, veh, lead, *args, dt = .1):
    #partial integrator with saturation control model.
    pass


    
def sv_obj(sim, auxinfo):
    #maximize squared velocity = sv 
    obj = 0 
    for i in sim.keys(): 
        for j in sim[i]: #squared velocity 
            obj = obj - j[1]**2
            if j[2] < .2: #penalty for having very small or negative headway. 
                obj = obj + 2**(-5*(j[2]-.2)) - 1 #-1 guarentees piecewise smooth which is condition for continuous gradient
    return obj 

def l2v_obj(sim,auxinfo):
    #minimize the l2 norm of the velocity
    #there is also a term that encourages the average velocity to be large. 
    obj = 0
    for i in sim.keys():
        vlist = []
        for j in sim[i]:
            vlist.append(j[1])
        avgv = np.mean(vlist)
        vlist = np.asarray(vlist)
        curobj = vlist - avgv
        curobj = np.square(curobj)
        curobj = sum(curobj)
        obj  = obj + curobj
        obj = obj - len(vlist)*avgv
    return obj
            
def avgv_obj(sim,auxinfo):
    obj = 0
    for i in sim.keys():
        for j in sim[i]:
            obj = obj - j[1] #maximize average velocity
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

def dboundary(speed, veh, dt):
    acc = (speed - veh[1])/dt
    if acc > 2:
        acc = 2
    elif acc < -5:
        acc = -5
    return [veh[1],acc]