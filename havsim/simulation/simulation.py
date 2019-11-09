
"""
@author: rlk268@cornell.edu
houses the main code for running simulations 

TO DO //
    update model to use for testing purposes
    test and debug simulate_cir and simulate_step. 

    implementing boundary conditions
    
    getting networks working
    
    how to do the adjoint calculation
    
    some tool for creating custom models out of modular parts, which should be part of models subpackage. 
    
    need to think about how to use modelinfo to implement all the parts of models that we need
    
"""

def simulate_cir(curstate, auxinfo, modelinfo, L, timesteps, dt) :
    """
    simulates vehicles on a circular test track
    
    inputs -
    curstate - dict, gives initial state
    auxinfo - dict, initialized auxinfo, see simulate step
    modelinfo - dict, any things which may apply, see simulate step
    L - float, length of road
    timesteps - int, number of timesteps to simulate
    dt - float, length of timestep
    
    outputs - 
    sim - all simulated states 
    auxinfo - updated auxinfo
    modelinfo - updated modelinfo
    """
    #initialize
    sim = {i:[curstate[i]] for i in curstate.keys()}
    
    
    for j in timesteps: 
        #update states
        nextstate, auxinfo, modelinfo = simulate_step(curstate,auxinfo,modelinfo,dt)
        
        #check for wraparound 
        for i in curstate.keys():
            if curstate[i][0] > L:
                curstate[i][0] = curstate[i][0] - L
        
        #update iteration
        curstate = nextstate
        for i in curstate.keys(): 
            sim[i].append(curstate[i])
            
    return sim,auxinfo,modelinfo

def simulate_net():
    """
    simulate on a network
    """
    pass

def simulate_step(curstate, auxinfo, modelinfo, dt): 
    """
    does a step of the simulation
    
    inputs - 
    curstate - current state of the simulation, dictionary where each value is the current state of each vehicle 
    states are a length n list where n is the order of the model. 
    So for 2nd order model the state of a vehicle is a list of position and speed
    
    
    auxinfo - dictionary where keys are IDs, the values are length (0), curleader (1), init entry time (2), 
    parameters (3), model (4), lane (5), road (6), past lead info (7); this is information that we will always have
    
    modelinfo - dictionary where keys are IDs, the values are special extra parts which 
    are only needed for certain models. The values in here are specific to the model we are using 
    
    dt - timestep
    
    outputs - 
    nextstate - current state in the next timestep
    
    auxinfo - auxinfo and modelinfo may be changed during the timestep
    modelinfo - 
    
    """
    nextstate = {}
    for i in curstate.keys(): 
        #get necessary info for update
        leadid = auxinfo[i][1]
        lead = curstate[leadid]
        leadlen = auxinfo[leadid][0]
        
        #call model for vehicle
        out = auxinfo[i][4](curstate[i],lead,auxinfo[i][3],leadlen, dt=dt)
        
        #update position in nextstate for vehicle
        nextstate[i] = [curstate[i][0] + dt* out[0], curstate[i][1] + dt*out[1]]
        
        
    return nextstate, auxinfo, modelinfo