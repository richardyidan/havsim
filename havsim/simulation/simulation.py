
"""
@author: rlk268@cornell.edu
houses the main code for running simulations 

TO DO /
    implementing boundary conditions 
    getting networks working (need to handle changing roads, some rule for merging) (relaxation, mobil)
    adding simple lane changing and multi lane 
    how to do the adjoint calculation
    reinforcement learning/rewriting calibration code
    
    want some loss function that won't end up converging to a lower speed like the l2v will - 
    something that balances between stability and high speed. 
    paper on AV control
    
    add some more models
    
    at some point will probably (?) need to refactor this code again so it's in a polished state-
    need to think about exactly what we need out of states, actions, how to handle things like different models,
    different update functions, handling derivatives, handling different loss functions, how lane changing will work 
    etc. 
    
    
    
"""

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

def simulate_step2(curstate, auxinfo, roadinfo, modelinfo, updatefun, dt): 
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
    11 - followers in adjacent lanes 
    12 - init entry time
    13 - past model reg info
    14 - past leader info
    15 - past lane info 
    16 - past road info

    
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
    lca = {}
    
    #note to self: good design pattern looks like- function which handles arguments, function which does actual call
    #eg for longitudinal, 1 function gets leader, any other extra parts, then it is passed to actual model
    #get actions in longitudinal movement (from CF model)
    #key is vehicle, value is acceleration in longitudinal movement 
    for i in curstate.keys():
        a[i] = auxinfo[i][6](auxinfo[i][5],curstate[i],curstate[auxinfo[i][1]], dt = dt)
        
    #get actions in latitudinal movement (from LC model)
    #key is vehicle, value is 
    lca = LCmodel()
    
    #update current state 
    nextstate = updatefun(curstate, a, auxinfo, roadinfo, dt)
    
    return nextstate, auxinfo

def update_net(state, action, auxinfo, roadinfo, dt):
    #given states and actions returns next state 
    
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

def update2nd(state, action, dt, roadinfo):
    #standard update function for a second order model
    nextstate = [state[0] + dt*action[0], state[1] + dt*action[1], None ]
    if nextstate[0] > roadinfo[0]: #wrap around 
        nextstate[0] = nextstate[0] - roadinfo[0]
    
    return nextstate

def LCmodel(): 
    #Based on MOBIL strategy
    #elements which won't be included 
    #   - cooperation for discretionary lane changes
    #   - aggressive state of target vehicle to force lane changes 
    
    
    #LC parameters (by index)
    #0 - safety criterion 
    #1 - incentive criteria
    #2 - politeness
    lca = {}
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



def simulate_net():
    """
    simulate on a network
    """
    pass