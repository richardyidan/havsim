
"""
@author: rlk268@cornell.edu
houses the main code for running simulations 
    
    
"""

from havsim.simulation.models import dboundary
import numpy as np 

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
    lca = LCmodel()
    
    #update current state 
    nextstate = updatefun(curstate, a, auxinfo, roadinfo, dt)
    
    return nextstate, auxinfo

def update_net(state, action, LCaction, auxinfo, roadinfo, dt):
    return 

def std_CF(veh, curstate, auxinfo, roadinfo, modelinfo,timeind, dt, relax): 
    #supposed to be model helper for standard CF model
    vehaux = auxinfo[veh]
    if relax:
        curstate[veh][2] += modelinfo[veh][0] #add relaxation if needed
    
    if vehaux[1] == None: #no leader
        dbc = roadinfo[vehaux[3]][4][timeind]
        out = dboundary(dbc, curstate[veh], dt)
    else: 
        out = vehaux[7](vehaux[5], curstate[veh], curstate[vehaux[1]], dt) #standard CF call 
        
    if relax: #remove relaxation 
        curstate[veh][2] += -modelinfo[veh][0]
    return out 
    
def get_headway(curstate, auxinfo, roadinfo, fol, lead):
    if fol == None or lead == None:
        return None
    hd = curstate[lead][0] - curstate[fol][0] - auxinfo[lead][4]
    if auxinfo[fol][3] != auxinfo[lead][3]: #only works for vehicles 1 road apart
        hd += roadinfo[auxinfo[fol][3]]
        
    return hd

def LCmodel(curstate, auxinfo, roadinfo, modelinfo, timeind, dt, userelax = False): 
    #Based on MOBIL strategy
    #elements which won't be included 
    #   - cooperation for discretionary lane changes
    #   - aggressive state of target vehicle to force lane changes 
    
    
    #LC parameters (by index)
    #0 - safety criterion 
    #1 - incentive criteria
    #2 - politeness
    #3 - probability to check discretionary
    lca = {}
    
    for i in curstate.keys(): 
        curaux = auxinfo[i]
        if np.random.rand()>curaux[8][3]: #check discretionary with this probability
            continue
        #check left side 
        lfol = curaux[11][0]
        if lfol !='': #'' = can not change 
            #if left follower/vehicle are in relax state, need to compute right acceleration to use
            if curaux[0][1] and not userelax: 
                cura = auxinfo[i][7](i, curstate, auxinfo, roadinfo, modelinfo,timeind, dt, False)
            if auxinfo[lfol][0][1] and not userelax:
                fola = auxinfo[lfol][7](lfol, curstate, auxinfo, roadinfo, modelinfo,timeind, dt, False)
            llead = auxinfo[lfol][1]
            #current headways
            curhd = curstate[i][2]
            lfolhd = curstate[lfol][2]
            #get new headways
            newhd = get_headway(curstate, auxinfo, roadinfo, i, llead) 
            newfolhd = get_headway(curstate, auxinfo, roadinfo, lfol, i)
            #get new acceleration for i
            curstate[i][2] = newhd
            newa = auxinfo[i][7](i, curstate, auxinfo, roadinfo, modelinfo,timeind, dt, False)
            #get new acceleration for lfol
            curstate[lfol][2] = newfolhd
            newlfola = auxinfo[lfol][7](lfol, curstate, auxinfo, roadinfo, modelinfo,timeind, dt, False)
        
        rfol = curaux[11][2]
        if rfol != '': 
            rlead = auxinfo[rfol][1]
            
    
    
    
    
    
    ######### first attempt 
    for i in curstate.keys(): 
        if modelinfo[i][0] == 0: #discretionary only 
            plc = auxinfo[i][8]
            
            newfolveh = vehorder[i][0] #do this as well for the vehorder[i][2] 
            #wrap all of this in a new function 
            if newfolveh != None: #check left side
                #compute new headway 
                newleadveh = auxinfo[newfolveh][1]
                newvehstate = curstate[i][:2]
                newvehstate.append(curstate[newleadveh][0] -newvehstate[0] - auxinfo[newleadveh][4])
                #check for case when leader is in a different road
                #check for when leader is None and boundary condition is used 
                #check for when follower is none 
                #use relax as well; should have relax in the state space
                
                #compute new acceleration for current vehicle 
                newacc = auxinfo[i][6](auxinfo[i][5], newvehstate, curstate[newleadveh], dt = dt)
                
                #compute new headway/accel for potential new follower
                newfolstate = curstate[newfolveh][:2]
                newfolstate.append(newvehstate[0] - newfolstate[0] - auxinfo[i][4])
                
                newfolacc = auxinfo[newfolveh][6](auxinfo[newfolveh][5], newfolstate, curstate[i])
                
                #compute new headway/accel for current follower
                nl = auxinfo[i][1]
                nf = vehorder[i][1]
                folstate = curstate[nf][:2]
                folstate.append(curstate[nl][0] - folstate[0] - auxinfo[nl][1])
                
                folacc = auxinfo[nf][6](auxinfo[nf][5], folstate, curstate[nl])
                
                if newacc > plc[0] and newfolacc > plc[0] and folacc > plc[0]: #safety requirement #use nested if statements instead 
                    incentive = newacc - a[i] + plc[2]*(newfolacc + folacc - a[nf] - a[newfolveh]) - plc[1] #there is no bias term 
                    
            else: 
                incentive = -math.inf
                
            
def checksafety():
    
    pass



def simulate_sn():
    """
    simulate on a simple network (sn = simple network)
    """
    pass


########################end code for simple network#################