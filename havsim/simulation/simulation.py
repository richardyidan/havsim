
"""
@author: rlk268@cornell.edu
houses the main code for running simulations 

TO DO /

    add some control models (CACC like in JM, follower stopper and PI with saturation)
    get some more loss functions
     some quick modifications to get plotting api to work on sim format so we can properly debug/make pretty pictures
    implementing boundary conditions (for now do the simplest possible)
    getting networks working (need to handle changing roads, some rule for merging)
     and write extended abstract
     
    adding lane changing and multi lane 
    
    how to do the adjoint calculation
    
    paper on AV control
    
    reinforcement learning/rewriting calibration code

    
"""
#some tool for creating custom models out of modular parts, which should be part of models subpackage. 
#need to think about how to use modelinfo to implement all the parts of models that we need, and how
#to automatically get the gradient as easily as possible.
#Also some things seem difficult to differentiate. Like the relaxation amount for example,
#potentially complicated to differentiate if you are including the time varying thing
#I think the way you can do this is to have something where you keep track of the regime
#of the model, and then you can also augment the state space. 
#Like for example for the relaxation example you would have a 
#special state when you experience a change, and record the value as a seperate entry in the state. 
#then when you are doing the gradient calculation everything just sort of falls into place automatically. 
#no messing around with all the funny business yourself. 

def simulate_step(curstate, auxinfo, roadinfo, updatefun, dt): 
    """
    does a step of the simulation
    
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
    7 - init entry time
    8 - past model reg info
    9 - past leader info
    10 - past lane info 
    11 - past road info
    this is information that we will always have
    
    roadinfo - dictionary
    
    updatefun - 
    
    dt - timestep
    
    outputs - 
    nextstate - current state in the next timestep
    
    auxinfo - auxinfo may be changed during the timestep
    
    loss - loss calculated at the current step 
    
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
    for i in state.keys():
        nextstate[i] = [state[i][0] + dt*action[i][0], state[i][1] + dt*action[i][1], None ]
        if nextstate[i][0] > roadinfo[0]: #wrap around 
            nextstate[i][0] = nextstate[i][0] - roadinfo[0]
        
    #update headway, which is part of state      
    for i in state.keys(): 
        #calculate headway
        leadid = auxinfo[i][1]
        nextstate[i][2] = nextstate[leadid][0] - nextstate[i][0] - auxinfo[leadid][4]
        
        #check for wraparound and if we need to update any special states for circular 
        if nextstate[i][2] < -roadinfo[1]: 
            nextstate[i][2] = nextstate[i][2] + roadinfo[0]
        
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

def simulate_net():
    """
    simulate on a network
    """
    pass

        #old code 
#    for i in curstate.keys(): 
#        #get necessary info for update
#        leadid = auxinfo[i][1]
#        lead = curstate[leadid]
#        leadlen = auxinfo[leadid][0]
#        
#        #call model for vehicle
#        out = auxinfo[i][4](curstate[i],lead,auxinfo[i][3],leadlen, *modelinfo[i], dt=dt)
#        
#        #update position in nextstate for vehicle
#        nextstate[i] = [curstate[i][0] + dt* out[0], curstate[i][1] + dt*out[1]]

    #this is one way to update the headway where we explicitly track if vehicles have a leader
    #which has wrapped around. But you know you could just make this so much simpler if you just
    #assume that if the headway is extremely negative, it's because you wrapped around. 
#    temp = [] #keep track of vehicles which wrap-around in next timestep
#    for i in nextstate.keys():
#        if auxinfo[i][1]: #if in activated 
#            if nextstate[i][0] > L: #follower wraps around so can reset 
#                temp.append(i)
#                auxinfo[i][1] = 0
#        else: 
#            if nextstate[auxinfo[i][1]][0] > L: #if leader wraps around
#                if nextstate[i][0] <= L:
#                    auxinfo[i][1] = 1 #active wrap around state for i 
#                else:  #edge case where both leader and vehicle wrap around in same time step
#                    temp.append(i)
#        for i in temp: 
#            nextstate[i][0] = nextstate[i][0] - L

def eq_circular(p, model, eqlfun, n, length = 2, L = None, v = None, perturb = 1e-2):
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
    auxinfo = {i:[0, i-1, 1, 1, length, p, model, 0, [],[],[],[]] for i in range(n)}
    auxinfo[0][1] = n-1 #first vehicle needs to follow last vehicle
        
    #create roadinfo
    roadinfo = [L, 3/4*L]
    
    return initstate, auxinfo, roadinfo

def simcir_obj(p, initstate, auxinfo, roadinfo, idlist, model, lossfn, updatefun = update_cir,  timesteps = 1000,  dt = .25, objonly = True):
    #p - parameters for AV 
    #idlist - vehicle IDs which will be controlled 
    #model - parametrization for AV 
    for i in idlist: 
        auxinfo[i][5] = p
        auxinfo[i][6] = model
        
    sim, curstate, auxinfo = simulate_cir(initstate, auxinfo, roadinfo, updatefun = updatefun, timesteps=timesteps, dt = dt)
    obj = lossfn(sim, auxinfo)
    
    if objonly:
        return obj
    else: 
        return obj, sim, curstate, auxinfo, roadinfo

