
"""
@author: rlk268@cornell.edu
houses the main code for running simulations 
    
    
"""

from havsim.simulation.models import dboundary
import numpy as np 
import math 

###############code for single lane circular road#################

def simulate_step(curstate, auxinfo, roadinfo, updatefun, dt): 
    """
    does a step of the simulation on a single lane circular road 
    
    inputs - 
    curstate - current state of the simulation, dictionary where each value is the current state of each vehicle 
    states are a length n list where n is the order of the model. 
    So for 2nd order model the state of a vehicle is a list of position and speed
    
    
    auxinfo - dictionary where keys are IDs, the values are 
    0 - current model regime 
    1 - current leader
    2 - current lane
    3 - current road 
    4 - length
    5 - parameters
    6 - model 
    7 - modelupdate
    8 - init entry time
    9 - past model reg info
    10 - past leader info
    11 - past lane info 
    12 - past road info
    this is information that we will always have
    
    roadinfo - dictionary
    
    updatefun - overall function that updates the states based on actions, there is also a specific updatefun for each vehicle (modelupdate)
    
    dt - timestep
    
    outputs - 
    nextstate - current state in the next timestep
    
    auxinfo - auxinfo may be changed during the timestep
    
    """
    nextstate = {}
    a = {}
    
    #get actions
    for i in curstate.keys():
        a[i] = auxinfo[i][6](auxinfo[i][5], curstate[i], curstate[auxinfo[i][1]], dt = dt)
        
    #update current state 
    nextstate = updatefun(curstate, a, auxinfo, roadinfo, dt)
    
    return nextstate, auxinfo


def update_cir(state, action, auxinfo, roadinfo, dt):
    #given states and actions returns next state 
    #meant for circular road to be used with simulate_step
    
    nextstate = {}
    
    #update states
#    for i in state.keys():
#        nextstate[i] = [state[i][0] + dt*action[i][0], state[i][1] + dt*action[i][1], None ]
#        if nextstate[i][0] > roadinfo[0]: #wrap around 
#            nextstate[i][0] = nextstate[i][0] - roadinfo[0]
    
    for i in state.keys():
        nextstate[i] = auxinfo[i][7](state[i],action[i],dt,roadinfo) #update is specific based on vehicle 
        
    #update headway, which is part of state      
    for i in state.keys(): 
        #calculate headway
        leadid = auxinfo[i][1]
        nextstate[i][2] = nextstate[leadid][0] - nextstate[i][0] - auxinfo[leadid][4]
        
        #check for wraparound and if we need to update any special states for circular 
        if nextstate[i][2] < -roadinfo[1]: 
            nextstate[i][2] = nextstate[i][2] + roadinfo[0]
        
    return nextstate

def update2nd_cir(state, action, dt, roadinfo):
    #standard update function for a second order model
    #meant to be used with update_cir
    nextstate = [state[0] + dt*action[0], state[1] + dt*action[1], None ]
    if nextstate[0] > roadinfo[0]: #wrap around 
        nextstate[0] = nextstate[0] - roadinfo[0]
    
    return nextstate



def simulate_cir(curstate, auxinfo, roadinfo, updatefun = update_cir, timesteps=1000, dt=.25 ):
    """
    simulates vehicles on a circular test track
    
    inputs -
    curstate - dict, gives initial state
    auxinfo - dict, initialized auxinfo, see simulate step
    roadinfo - dict
    L - float, length of road
    timesteps - int, number of timesteps to simulate
    dt - float, length of timestep
    
    outputs - 
    sim - all simulated states 
    auxinfo - updated auxinfo
    """
    #initialize
    sim = {i:[curstate[i]] for i in curstate.keys()}
    
    for j in range(timesteps): 
        #update states
        nextstate, auxinfo= simulate_step(curstate,auxinfo,roadinfo, updatefun, dt)
        
        #update iteration
        curstate = nextstate
        for i in curstate.keys(): 
            sim[i].append(curstate[i])
            
    return sim,curstate, auxinfo


def eq_circular(p, model, modelupdate, eqlfun, n, length = 2, L = None, v = None, perturb = 1e-2):
    #given circular road with length L with n vehicles which follow model model with parameters p, 
    #solves for the equilibrium solution and initializes vehicles in this circular road in the eql solution, 
    #with the perturbation perturb applied to one of the vehicles. 
    #you can eithe initialize L, in which case it will solve for v, or you can 
    #initialize v, and it will solve for the L. 
    #inputs - 
    #p- model parameters (scalar)
    #length - length of vehicles (scalar)
    #model - function for the model 
    #n - number of vehicles
    #l = None - length of circular test track
    #v = None - eq'l speed used 
    #perturb = 1e-2 - how much to perturb from the eql solution
    
    #outputs - 
    #curstate - state of the eql solution with perturbation
    #auxinfo - initialized auxiliary info for simulation
    #roadinfo - initialized road info for simulation 
    
    #first we need to solve for the equilibrium solution which forms the basis for the IC. 
    if L == None and v == None: 
        print('you need to specify either L or v to create the equilibrium solution')
        return
    elif L == None: 
        s = eqlfun(p,None,v,find='s')
        L = (s+length)*n
    elif v == None: 
        s = L / n - length
        v = eqlfun(p,s,None,find='v')
        
    #initialize based on equilibrium
    initstate = {n-i-1: [(s+length)*i,v, s] for i in range(n)}
    initstate[n-1][0] = initstate[n-1][0] + perturb #apply perturbation
    initstate[n-1][1] = initstate[n-1][1] + perturb
    initstate[n-1][2] = initstate[n-1][2] - perturb
    
    #create auxinfo
    auxinfo = {i:[0, i-1, 1, 1, length, p, model, modelupdate, 0, [],[],[],[]] for i in range(n)}
    auxinfo[0][1] = n-1 #first vehicle needs to follow last vehicle
        
    #create roadinfo
    roadinfo = [L, 1/6*L]
    
    return initstate, auxinfo, roadinfo

def simcir_obj(p, initstate, auxinfo, roadinfo, idlist, model, modelupdate, lossfn, updatefun = update_cir,  timesteps = 1000,  dt = .25, objonly = True):
    #p - parameters for AV 
    #idlist - vehicle IDs which will be controlled 
    #model - parametrization for AV 
    #simple simulation on circular road mainly based on simulate_step
    for i in idlist: 
        auxinfo[i][5] = p
        auxinfo[i][6] = model
        auxinfo[i][7] = modelupdate
        
    sim, curstate, auxinfo = simulate_cir(initstate, auxinfo, roadinfo, updatefun = updatefun, timesteps=timesteps, dt = dt)
    obj = lossfn(sim, auxinfo)
    
    if objonly:
        return obj
    else: 
        return obj, sim, curstate, auxinfo, roadinfo
    
########################end code for single lane circular road#################
        
##################code for simple network with discretionary changes only, no merges/diverges, no routes###########

def simulate_step2(curstate, auxinfo, roadinfo, modelinfo, updatefun, timeind, dt): 
    """
    does a step of the simulation for the full simulation which includes boundary conditions and LC
    
    inputs - 
    curstate - current state of the simulation, dictionary where each value is the current state of each vehicle 
    states are a length n list where n is the order of the model. 
    So for 2nd order model the state of a vehicle is a list of position and speed
    
    
    auxinfo - dictionary where keys are IDs, the values are 
    0 - current model regime 
    1 - current leader
    2 - current lane
    3 - current road 
    4 - length
    5 - parameters
    6 - model 
    7 - model helper 
    8 - LC parameters
    9 - LC model 
    10 - update function
    11 - followers  (left, current, right)
    13 - init entry time
    14 - past model reg info
    15 - past LC regime info 
    16 - past leader info
    17 - past road info
    18 - LC regime

    
    roadinfo - dictionary, encodes the road network and also stores boundary conditions 
    
    modelinfo - dictionary, stores any additional information which is not part of the state,
    does not explicitly state the regime of the model, 
    but is needed for the simulation/gradient calculation (e.g. relax amounts, LC model regimes, action point amounts, realization of noise)
    dictionary of dict
    
    
    updatefun - overall function that updates the states based on actions, there is also a specific updatefun for each vehicle (modelupdate
    
    dt - timestep
    
    outputs - 
    nextstate - current state in the next timestep
    
    auxinfo - auxinfo may be changed during the timestep
    
    """
    nextstate = {}
    a = {}
    
    #get actions in latitudinal movement 
    for i in curstate.keys():
#        if auxinfo[i][1] ==None: 
#            dbc = roadinfo[auxinfo[i][3]][4][timeind] #speed at downstream 
#            a[i] = dboundary(dbc, curstate[i],dt)
#        else:
#            #standard call signature for CF model 
#            a[i] = auxinfo[i][6](auxinfo[i][5],curstate[i],curstate[auxinfo[i][1]], dt = dt)
        
        a[i] = auxinfo[i][7](i, curstate, auxinfo, roadinfo, modelinfo,timeind, dt, auxinfo[i][0][1]) #wrapper function for model call 
        
    #get actions in latitudinal movement (from LC model)
    lca = LCmodel(a, curstate, auxinfo, roadinfo, modelinfo, timeind, dt)
    
    #update current state 
    nextstate = updatefun(curstate, a, auxinfo, roadinfo, dt)
    
    return nextstate, auxinfo


def std_CF(veh, curstate, auxinfo, roadinfo, modelinfo,timeind, dt, relax): 
    #supposed to be model helper for standard CF model
    vehaux = auxinfo[veh]
    if relax:
        curstate[veh][2] += modelinfo[veh][0] #add relaxation if needed
    
    if vehaux[1] == None: #no leader
        dbc = roadinfo[vehaux[3]][4][vehaux[2]][timeind]
        out = dboundary(dbc, curstate[veh], dt)
    else: 
        out = vehaux[7](vehaux[5], curstate[veh], curstate[vehaux[1]], dt) #standard CF call 
        
    if relax: #remove relaxation 
        curstate[veh][2] += -modelinfo[veh][0]
    return out 
    
def get_headway(curstate, auxinfo, roadinfo, fol, lead):
    hd = curstate[lead][0] - curstate[fol][0] - auxinfo[lead][4]
    if auxinfo[fol][3] != auxinfo[lead][3]: #only works for vehicles 1 road apart
        
        hd += headway_helper(roadinfo,auxinfo[fol][3],auxinfo[fol][2], auxinfo[lead][3])
        
    return hd

def headway_helper(roadinfo, folroad, follane, leadroad):
    nextroad, nextlane = roadinfo[folroad][1][follane]
    out = roadinfo[folroad][2]
    while nextroad != leadroad: 
        out += roadinfo[folroad][2]
        folroad, follane = nextroad, nextlane
        nextroad, nextlane = roadinfo[folroad][1][follane]
        
    return out

def LCmodel(a, curstate, auxinfo, roadinfo, modelinfo, timeind, dt, userelax = False): 
    #Based on MOBIL strategy
    #elements which won't be included 
    #   - cooperation for discretionary lane changes
    #   - aggressive state of target vehicle to force lane changes 
    
    
    #LC parameters (by index)
    #0 - safety criterion 
    #1 - incentive criteria
    #2 - politeness
    #3 - probability to check discretionary
    #4 - bias on left side 
    #5 - bias on right side
    lca = {}
    
    for i in curstate.keys(): 
        curaux = auxinfo[i]
        p = curaux[8]
        
        if np.random.rand()>curaux[8][3]: #check discretionary with this probability
            continue
        
        lfol = curaux[11][0]
        rfol = curaux[11][2]
        if lfol == '' and rfol == '': 
            continue
        else:  #calculate change for follower, calculate current vehicle acc
            
            fol = curaux[11][1]
            curhd = curstate[i][2]
            if fol == None:
                fola = 0
                newfola = 0
            else:
                folaux = auxinfo[fol]
                folhd = curstate[fol][2] #current follower headway 
                #get current follower acceleration 
                if folaux[0][1] and not userelax: 
                    fola = folaux[7](fol, curstate, auxinfo, roadinfo, modelinfo, timeind, dt, False)
                else: 
                    fola = a[fol]
                #get new follower acceleration
                lead = curaux[1]
                if lead == None: 
                    folaux[1] = None
                    newfola = folaux[7](fol,curstate,auxinfo,roadinfo,modelinfo,timeind,dt,False)
                else:
                    newfolhd = get_headway(curstate, auxinfo, roadinfo, fol, lead)
                    curstate[fol][2] = newfolhd
                    folaux[1] = lead
                    newfola = folaux[7](fol,curstate,auxinfo,roadinfo,modelinfo,timeind,dt,False)
                
                #get vehicle acceleration if needed
                
                if curaux[0][1] and not userelax and curaux[1] is not None: 
                    cura = curaux[7](i, curstate, auxinfo, roadinfo, modelinfo,timeind, dt, False)
                else: 
                    cura = a[i]
                
        if lfol != '': #new to calculate new vehicle acceleration, new left follower acceleration 
            
            #this code is wrapped in mobil_change now 
#            if lfol == None: 
#                lfola = 0
#                newlfola = 0
#            else: 
#                lfolaux = auxinfo[lfol]
#                #left follower current acceleration 
#                if lfolaux[0][1] and not userelax:
#                    lfola = lfolaux[7](lfol, curstate, auxinfo, roadinfo, modelinfo,timeind, dt, False)
#                else: 
#                    lfola = a[lfol]
#                #left side leader
#                llead = lfolaux[1]
#                
#                #get new follower acceleration and vehicle acceleration
#                lfolaux[1] = i
#                newlfolhd = get_headway(curstate,auxinfo,roadinfo,lfol,i)
#                curstate[lfol][2] = newlfolhd
#                
#                if lfolaux[0][1] and not userelax:
#                    newlfola = lfolaux[7](lfol, curstate, auxinfo, roadinfo, modelinfo,timeind, dt, False)
#                else: 
#                    newlfola = lfolaux[7](lfol, curstate, auxinfo, roadinfo, modelinfo,timeind, dt, True)
#                if llead == None: 
#                    curaux[1] = None
#                    curaux[2] = lfolaux[2]
#                    newla = curaux[7](i, curstate, auxinfo, roadinfo, modelinfo, timeind, dt, False) #lead is none means we don't check relax
#                
#                else: 
#                    curaux[1] = llead
#                    newlhd = get_headway(curstate, auxinfo, roadinfo, i, llead)
#                    curstate[i][2] = newlhd
#                    if curaux[0][1] and not userelax: 
#                        newla = curaux[7](i,curstate,auxinfo,roadinfo,modelinfo,timeind,dt,False)
#                    else: 
#                        newla = curaux[7](i,curstate,auxinfo,roadinfo,modelinfo,timeind,dt,True)
#                    
#            lincentive = newla - cura + p[2]*(newlfola - lfola + newfola - fola) #no bias term 
            
            lincentive, newla, lfola, newlfola = mobil_change(i,lfol, curstate, auxinfo, roadinfo, 
                                                                            modelinfo, timeind, dt, userelax, a, cura, newfola, fola, p)

        else: 
            lincentive = -math.inf
        
        if rfol != '': 
            rincentive, newra, rfola, newrfola = mobil_change(i, rfol, curstate, auxinfo, roadinfo, modelinfo,
                                                                             timeind, dt, userelax, a, cura, newfola, fola, p)
        else: 
            rincentive = -math.inf
        
        
        if rincentive > lincentive: 
            side = 'r'
            incentive = rincentive
            selfsafe = newra
            folsafe = newrfola
        else:
            side = 'l'
            incentive = lincentive
            selfsafe = newla
            folsafe = newlfola
        
        if incentive > p[1]: #incentive criteria
            if selfsafe > p[0] and folsafe > p[0]:
                lca[i] = side
            else: 
                #do tactical/cooperation step if desired
                pass
                
        #reset changes to curstate
        curstate[i][2] = curhd
        curstate[fol][2] = folhd
        curaux[1] = lead
        folaux[1] = i
                
    return lca
            
                
            
                
def mobil_change(i,lfol, curstate, auxinfo, roadinfo, modelinfo, timeind, dt, userelax, a, cura, newfola, fola, p):
    curaux = auxinfo[i]
    if lfol == None: 
        lfola = 0
        newlfola = 0
    else: 
        lfolaux = auxinfo[lfol]
        #left follower current acceleration 
        if lfolaux[0][1] and not userelax:
            lfola = lfolaux[7](lfol, curstate, auxinfo, roadinfo, modelinfo,timeind, dt, False)
        else: 
            lfola = a[lfol]
        #left side leader
        llead = lfolaux[1]
        
        #get new follower acceleration and vehicle acceleration
        lfolaux[1] = i
        lfolhd = curstate[lfol][2]
        newlfolhd = get_headway(curstate,auxinfo,roadinfo,lfol,i)
        curstate[lfol][2] = newlfolhd
        
        if lfolaux[0][1] and not userelax:
            newlfola = lfolaux[7](lfol, curstate, auxinfo, roadinfo, modelinfo,timeind, dt, False)
        else: 
            newlfola = lfolaux[7](lfol, curstate, auxinfo, roadinfo, modelinfo,timeind, dt, True)
        if llead == None: 
            curaux[1] = None
            curaux[2] = lfolaux[2]
            newla = curaux[7](i, curstate, auxinfo, roadinfo, modelinfo, timeind, dt, False) #lead is none means we don't check relax
        
        else: 
            curaux[1] = llead
            newlhd = get_headway(curstate, auxinfo, roadinfo, i, llead)
            curstate[i][2] = newlhd
            if curaux[0][1] and not userelax: 
                newla = curaux[7](i,curstate,auxinfo,roadinfo,modelinfo,timeind,dt,False)
            else: 
                newla = curaux[7](i,curstate,auxinfo,roadinfo,modelinfo,timeind,dt,True)
        
        curstate[lfol][2] = lfolhd
        lfolaux[1] = llead
            
    lincentive = newla - cura + p[2]*(newlfola - lfola + newfola - fola) #no bias term 
    
            
    return lincentive, newla, lfola, newlfola


def update_sn(a, lca, curstate, auxinfo, roadinfo, modelinfo, timeind, dt):
    #update lanes, leaders, followers for all lane change actions 
    #vehicles may change at same time into same gap because we don't check this case
    for i in lca.keys(): 
        #define change side, opposite side
        curaux = auxinfo[i]
        road = curaux[3]
        lane = curaux[2]
        if lca[i] == 'l': 
            lcside = 0
            opside = 2
            lcsidelane = lane-1
            opsidelane = lane+1
        else: 
            lcside = 2
            opside = 0
            lcsidelane = lane+1
            opsidelane = lane-1
        
        #update opposite side leader
        opfol = curaux[11][opside]
        if opfol == '':
            pass
        else:
            if opfol == None:
                oplead = roadinfo[road][6][opsidelane]
            else: 
                oplead = auxinfo[opfol][1]
            if oplead is not None: 
                auxinfo[oplead][11][lcside] = curaux[11][1] #opposite side LC side follower is current follower
        
        #update current leader
        if curaux[1] == None: 
            pass
        else: 
            auxinfo[curaux[1]][11][1] = curaux[11][1]
            if curaux[11][lcside] == auxinfo[curaux[1]][11][lcside]:
                auxinfo[curaux[1]][11][lcside] = i
        
        #update LC side leader
        lcfol = curaux[11][lcside]
        if lcfol == None: 
            lclead = roadinfo[road][6][lcsidelane]
        else: 
            lclead = auxinfo[lcfol][1]
        if lclead is not None: 
            auxinfo[lclead][11][opside] = curaux[11][1]
            auxinfo[lclead][11][1] = i
            
        #update vehicle
        curaux[1] = lclead
        curaux[11][opside] = curaux[11][1]
        curaux[11][1] = lcfol
        
        #update new LC side 
        #check if new LC side even exists
        if lca[i]=='l' and lcsidelane ==0:
            pass
        elif opsidelane == roadinfo[road][0]:
            pass
        else: 
            newlcside = lcsidelane -1 if lca[i] == 'l' else lcsidelane + 1
            if lclead == None: 
                if lcfol == None: 
                    newlcveh = roadinfo[road][6][newlcside]
                newlcveh = lcfol
            else: 
                newlcveh = lclead
            
        
        
                
        
        
    
    pass
    
    

def simulate_sn():
    """
    simulate on a simple network (sn = simple network)
    """
    pass


########################end code for simple network#################