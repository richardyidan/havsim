
"""
@author: rlk268@cornell.edu
houses the main code for running simulations 
    
    
"""

from havsim.simulation.models import dboundary, IDM_b3, IDM_b3_b
import numpy as np 
import math 
import scipy.optimize as sc 

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
    -discretionary only changing, no routes 
    -on/off ramps cannot be simulated, only diverges/merges
    -no mandatory changing means bottlenecks are not going to lead to correct behavior
    -no relaxation
    -no tactical or cooperative behavior
    
    inputs - 
    curstate - current state of the simulation, dictionary where each value is the current state of each vehicle 
    states are a length n list where n is the order of the model. 
    So for 2nd order model the state of a vehicle is a list of position and speed
    
    
    auxinfo - dictionary where keys are IDs (floats), the values are a list with these indices
    0 - current model regime 
    1 - current leader (key)
    2 - current lane (int)
    3 - current road  (key)
    4 - length (float)
    5 - parameters (list of floats)
    6 - model (function)
    7 - model helper (function)
    8 - LC parameters (list of floats)
    9 - LC model (function)
    10 - update function (function)
    11 - followers  (left, current, right) (list of keys)
    13 - init entry time (int)
    14 - past model reg info - past info are tuples of (value, first time, last time) (sparse format)
    15 - past LC regime info 
    16 - past leader info
    17 - past road info
    18 - past lane info
    19 - LC regime 
    20 - left/right leaders - list of set of keys which have the vehicle as a right follower (in 0 index) and 
    left follower (in 2 index). 1 index is empty to be consistent with followers (11 index)
    auxinfo also has some special keys - 
    if key is a string, it is corresponding to an anchor point where no vehicle exists. These special anchors 
    are at the beginning of tracks (a track would be interpreted as a list of (key, lane) tuples that represent
    the roads you follow if you go in a straight line along the road network). 

    
    roadinfo - dictionary, encodes the road network and also stores boundary conditions 
    0 - number of lanes (int)
    1 - what lanes connect to (array of (key, lane) tuples), correspond to lanes
    2 - length of road (float)
    3 - upstream boundary - list of lists, inner lists are sequences of speeds (or none), outer lists correspond to lanes
    4 - downstream boundary - same format as upstream boundary, but for downstream 
    5 - inflow buffer - list of floats, represents how close we are to adding next vehicle
    6 - anchor vehicles - list of keys corresponding to anchor for each lane (string key)
    7 - what lanes are connected from (array of (key, lane) tuples) corresponding to lanes 
    8 - first vehicles - list of keys or None corresponding to first vehicle in lane. Only required for certain lanes
    roadinfo also contains special keys which are tuples of (road1, road2) which correspond to the distance between 
    these roads - these are for properly computing the headway when vehicles are on different 
    roads. 
    Each lane in a road is interpreted as being the same length. 
    
    
    
    modelinfo - dictionary, stores any additional information which is not part of the state,
    does not explicitly state the regime of the model, 
    but is needed for the simulation/gradient calculation (e.g. relax amounts, LC model regimes, action point amounts, realization of noise)
    dictionary of dict
    
    
    updatefun - overall function that updates the states based on actions, there is also a specific updatefun for each vehicle (modelupdate
    
    dt - timestep
    
    outputs - 
    updates all the data structures in place 
    curstate - current state in the next timestep
    auxinfo - auxinfo may be changed during the timestep
    roadinfo
    modelinfo 
    """
#    nextstate = {}
    a = {}
    lca = {}
    
    
    #get actions in latitudinal movement 
    for i in curstate.keys():
        a[i] = auxinfo[i][7](i, curstate, auxinfo, roadinfo, modelinfo,timeind, dt, auxinfo[i][0][1]) #wrapper function for model call 
        
        
    #get actions in latitudinal movement (from LC model)
    for i in curstate.keys(): 
        std_LC(i, lca, a, curstate, auxinfo, roadinfo, modelinfo, timeind, dt)
        
    
    #update current state 
    updatefun(a, lca, curstate, auxinfo, roadinfo, modelinfo, timeind, dt) #updates curstate, auxinfo, roadinfo in place 
    
    
    #update inflow
    increment_inflow2(curstate, auxinfo, roadinfo, timeind, dt) #adds vehicle with default parameters 
    
    
    return curstate, auxinfo, roadinfo


def std_CF(veh, curstate, auxinfo, roadinfo, modelinfo,timeind, dt, relax): 
    #supposed to be model helper for standard CF model
    vehaux = auxinfo[veh]
    if relax:
        curstate[veh][2] += modelinfo[veh][0] #add relaxation if needed
        
        #actual call looks like this 
        if vehaux[1] == None: #no leader -> action chosen based on boundary conditions 
            dbc = roadinfo[vehaux[3]][4][vehaux[2]][timeind]
            out = dboundary(dbc, curstate[veh], dt)
        else:  #standard CF call 
            out = vehaux[6](vehaux[5], curstate[veh], curstate[vehaux[1]], dt) 
            
        curstate[veh][2] += -modelinfo[veh][0] #undo relaxation 
        
    else: 
        if vehaux[1] == None: #no leader -> action chosen based on boundary conditions 
            dbc = roadinfo[vehaux[3]][4][vehaux[2]][timeind]
            out = dboundary(dbc, curstate[veh], dt)
        else:  #standard CF call 
            out = vehaux[6](vehaux[5], curstate[veh], curstate[vehaux[1]], dt) 
            
    return out 
    
def get_headway(curstate, auxinfo, roadinfo, fol, lead):
    hd = curstate[lead][0] - curstate[fol][0] - auxinfo[lead][4]
    if auxinfo[fol][3] != auxinfo[lead][3]:
#        hd += headway_helper(roadinfo,auxinfo[fol][3],auxinfo[fol][2], auxinfo[lead][3]) #old solution 
        hd += roadinfo[(auxinfo[fol][3], auxinfo[lead][3])] #better to just store in roadinfo 
    return hd

def get_dist(curstate, auxinfo, roadinfo, fol, lead):
    dist = curstate[lead][0] - curstate[fol][0]
    if auxinfo[fol][3] != auxinfo[lead][3]:
        dist += roadinfo[(auxinfo[fol][3], auxinfo[lead][3])]
    return dist
        
def get_dist2(curstate, auxinfo, roadinfo, fol, lead):
    #can handle case where fol might be special string (i.e. anchor vehicle)
    if type(fol) == str: 
        dist = curstate[lead][0]
        if auxinfo[fol][3] != auxinfo[lead][3]:
            dist += roadinfo[(auxinfo[fol][3], auxinfo[lead][3])]
        return dist
    else: 
        return get_dist(curstate, auxinfo, roadinfo, fol, lead)
#        dist = curstate[lead][0] - curstate[fol][0]
#        if auxinfo[fol][3] != auxinfo[lead][3]:
#            dist += roadinfo[(auxinfo[fol][3], auxinfo[lead][3])]
#    return dist

def headway_helper(roadinfo, folroad, follane, leadroad):
    #deprecated######
    #this will have problems if follower is actually ahead of leader
    nextroad, nextlane = roadinfo[folroad][1][follane]
    out = roadinfo[folroad][2]
    while nextroad != leadroad: 
        out += roadinfo[folroad][2]
        folroad, follane = nextroad, nextlane
        nextroad, nextlane = roadinfo[folroad][1][follane]
        
    return out

def dist_to_end(roadinfo, road, lane):
    #calculates distance to the end of network starting from road 
    L = roadinfo[road][2] 
    nextroad, nextlane = roadinfo[road][1][lane][:]
    while nextroad is not None: 
        road = nextroad
        lane = nextlane
        L += roadinfo[road][2]
        nextroad, nextlane = roadinfo[road][1][lane][:]
    return L 
    
    
def std_LC(i, lca, a, curstate, auxinfo, roadinfo, modelinfo, timeind, dt, userelax_cur = True, userelax_new = False, get_fol = True): 
    #more generalized/updated version of LCmodel
    curaux = auxinfo[i]
    lfol, rfol = curaux[11][0], curaux[11][2]
    
    if lfol == '' and rfol == '': #nothing to check 
        return 
    
    if np.random.rand() > curaux[8][3]: #consider change only with this probability 
        return 
    
#    p = curaux[8]
#    curhd = curstate[i][2]
    #for both left and right sides, get new headway, new leader, cur/new headway for new follower
    #initialize
    llead=rlead=newlhd=newrhd=lfolhd=rfolhd=newlfolhd=newrfolhd = 0
    if lfol != '': 
        llead = auxinfo[lfol][1]
        if llead == None: 
            newlhd = dist_to_end(roadinfo, curaux[3], curaux[2]) - curstate[i][0] #assuming you stay on same road, works for simple network only 
        else:
            newlhd = get_headway(curstate, auxinfo, roadinfo, i, llead)
        if type(lfol) == str:
            pass
        else:
            lfolhd = curstate[lfol][2]
            newlfolhd = get_headway(curstate, auxinfo, roadinfo, lfol, i)
            
    if rfol != '': #same code as for lfol
        rlead = auxinfo[rfol][1]
        if rlead == None: 
            newrhd = dist_to_end(roadinfo, curaux[3], curaux[2]) - curstate[i][0]
        else:
            newrhd = get_headway(curstate, auxinfo, roadinfo, i, rlead)
        if type(rfol) == str:
            pass
        else:
            rfolhd = curstate[rfol][2]
            newrfolhd = get_headway(curstate, auxinfo, roadinfo, rfol, i)
            
    
    if get_fol: #model call uses current follower headway 
        fol = curaux[11][1]
        lead = curaux[1]
        if type(fol) == str or lead == None:
            newfolhd = 0
        else: 
            newfolhd = get_headway(curstate, auxinfo, roadinfo, fol, lead)
        
    #current standard call signature
    mobil(i, a, lca, curstate, auxinfo, roadinfo, modelinfo, lfol, rfol, llead, rlead, newlhd, newrhd, lfolhd, rfolhd, 
                newlfolhd, newrfolhd, fol, lead, newfolhd, timeind, dt, userelax_cur, userelax_new)
        
    return
        
def mobil(i, a, lca, curstate, auxinfo, roadinfo, modelinfo, lfol, rfol, llead, rlead, newlhd, newrhd, lfolhd, rfolhd, 
          newlfolhd, newrfolhd, fol, lead, newfolhd, timeind, dt, userelax_cur, userelax_new):
    #returns action according to mobil strategy - refactored of LCmodel and mobil_change
    #options to use/not use relaxation when computing
    #userelax_cur = True will recompute the action without relaxation if relaxation is activated (for all vehicles involved)
    #userelax_new = True whether or not to use relaxation when computing the new actions 
    
    #LC parameters (by index) e.g. parameters = [2, .1, .2, .2, .2, .2]
    #0 - safety criterion 
    #1 - incentive criteria
    #2 - politeness
    #3 - probability to check discretionary
    #4 - bias on left side 
    #5 - bias on right side
    
    lincentive = rincentive = -math.inf
    curaux = auxinfo[i]
#    folaux = auxinfo[fol]
#    lfolaux = auxinfo[lfol]
#    rfolaux = auxinfo[rfol]
    
    p = curaux[8]
    curhd = curstate[i][2]
    
    #vehicle's current action
    if not userelax_cur and curaux[0][1] and curaux[1] is not None: 
        cura = curaux[7](i, curstate, auxinfo, roadinfo, modelinfo, timeind, dt, False)[1]
    else: 
        cura = a[i][1]
    
    #get follower's action and new action 
    fola, newfola = mobil_helper(fol, lead, i, newfolhd, a, curstate, auxinfo, roadinfo, modelinfo, timeind, dt, userelax_cur, userelax_new)
    
#    if type(fol) == str: 
#        fola = 0
#        newfola = 0
#    else: 
#        #current follower acceleration 
#        if not userelax_cur and folaux[0][1]: #don't use relaxation 
#            fola = folaux[7](fol, curstate, auxinfo, roadinfo, modelinfo, timeind, dt, False)
#        else: 
#            fola = a[fol]
#        #new follower acceleration 
#        curfolhd = curstate[fol][2]
#        folaux[1] = lead
#        curstate[fol][2] = newfolhd
#        mybool = userelax_new and folaux[0][1]
#        newfola = folaux[7](fol,curstate, auxinfo, roadinfo, modelinfo, timeind, dt, mybool)
#        #reset follower state
#        curstate[fol][2] = curfolhd
#        folaux[1] = i
    
    if lfol != '':
        #get left follower's action and new action 
        lfola, newlfola = mobil_helper(lfol, i, llead, newlfolhd, a, curstate, auxinfo, 
                                       roadinfo, modelinfo, timeind, dt, userelax_cur, userelax_new)
        #vehicle's new action 
        curaux[1] = llead
        curstate[i][2] = newlhd
        mybool = userelax_new and curaux[0][1]
        newla = curaux[7](i, curstate, auxinfo, roadinfo, modelinfo, timeind, dt, mybool)[1]
        #reset vehicle's state 
        curstate[i][2] = curhd
        curaux[1] = lead
        #calculate incentive 
        lincentive = newla - cura + p[2]*(newlfola-lfola + newfola - fola) + p[4]
        
    if rfol != '': 
        rfola, newrfola = mobil_helper(rfol, i, rlead, newrfolhd, a, curstate, auxinfo, 
                                       roadinfo, modelinfo, timeind, dt, userelax_cur, userelax_new)
        #vehicle's new action 
        curaux[1] = rlead
        curstate[i][2] = newrhd
        mybool = userelax_new and curaux[0][1]
        newra = curaux[7](i, curstate, auxinfo, roadinfo, modelinfo, timeind, dt, mybool)[1]
        #reset vehicle's state 
        curstate[i][2] = curhd
        curaux[1] = lead
        #calculate incentive
        rincentive = newra - cura + p[2]*(newrfola - rfola + newfola - fola) + p[5]
    
    
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
    return
        
        
        
def mobil_helper(fol, lead, i, newfolhd, a, curstate, auxinfo, roadinfo, modelinfo, timeind, dt, userelax_cur, userelax_new):
    #calculates the current, new acceleration (action) for vehicle fol
    #fol -vehicle to calculate current, new acceleration for 
    #lead - new lead vehicle 
    #i - current lead vehicle 
    #newfolhd - newheadway corresponding to new lead vehicle 
    
    folaux = auxinfo[fol]
    if type(fol) == str: 
        fola = 0
        newfola = 0
    else: 
        #current follower acceleration 
        if not userelax_cur and folaux[0][1]: #don't use relaxation 
            fola = folaux[7](fol, curstate, auxinfo, roadinfo, modelinfo, timeind, dt, False)[1]
        else: 
            fola = a[fol][1]
        #new follower acceleration 
        curfolhd = curstate[fol][2]
        folaux[1] = lead
        curstate[fol][2] = newfolhd
        mybool = userelax_new and folaux[0][1]
        newfola = folaux[7](fol,curstate, auxinfo, roadinfo, modelinfo, timeind, dt, mybool)[1]
        #reset follower state
        curstate[fol][2] = curfolhd
        folaux[1] = i
    
    return fola, newfola
    

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
    for i in lca.keys():  #believe this is the only part of code which cannot be parralelized - updating lane changes 
        #prepare for updates
        curaux = auxinfo[i]
        road = curaux[3]
        lane = curaux[2]
        fol = curaux[11][1]
        folaux = auxinfo[fol]
        #define change side, opposite side
        if lca[i] == 'l': 
            lcside = 0
            opside = 2
            lcsidelane = lane-1
#            opsidelane = lane+1
        else: 
            lcside = 2
            opside = 0
            lcsidelane = lane+1
#            opsidelane = lane-1
        #################
        #update current leader 
        lead = curaux[1]
        if lead == None: 
            pass
        else: 
            auxinfo[lead][11][1] = fol
#            if curaux[11][lcside] == auxinfo[lead][11][lcside]: 
#                auxinfo[lead][11][lcside] = i 
        
        #update opposite side leaders 
        for j in curaux[20][opside]:
            auxinfo[j][11][lcside] = fol
        #update lc side leaders 
        for j in curaux[20][lcside]: 
            auxinfo[j][11][opside] = fol
            
        #update follower
        folaux[20][lcside].update(curaux[20][lcside])
        folaux[20][opside].update(curaux[20][opside])
        folaux[1] = lead
        folaux[16][-1].append(timeind) #update even if follower is special (i.e. str)
        folaux[16].append([lead, timeind + 1])
        
        #update vehicle 
        #update opposite side for veh
        opsidefol = curaux[11][opside] 
        if opsidefol != '':
            auxinfo[opsidefol][20][lcside].remove(i) #old opposite side follower 
        curaux[11][opside] = fol 
        folaux[20][lcside].add(i)
        #update cur lc side follower for veh 
        lcfol = curaux[11][lcside]
        lcfolaux = auxinfo[lcfol]
        lclead = lcfolaux[1]
        lcfolaux[1] = i 
        lcfolaux[16][-1].append(timeind)
        lcfolaux[16].append([i, timeind + 1])
        lcfolaux[20][opside].remove(i)
        curaux[11][1] = lcfol
        #update lc side leader
        curaux[1] = lclead 
        curaux[16][-1].append(timeind)
        curaux[16].append([lclead, timeind + 1])
        curaux[18][-1].append(timeind)
        curaux[18].append([lcsidelane, timeind+1])
        curaux[2] = lcsidelane
        if lclead is not None: 
            auxinfo[lclead][11][1] = i
        #update for new left/right leaders
        newset = set()
        for j in lcfolaux[20][opside].copy():
            curdist = get_dist(curstate, auxinfo, roadinfo, j, i)
            if curdist < 0: 
                auxinfo[j][11][lcside] = i
                newset.add(j)
                lcfolaux[20][opside].remove(j)
        curaux[20][opside] = newset
        
        newset = set()
        maxdist = -math.inf
        minveh = None
        for j in lcfolaux[20][lcside].copy(): 
            curdist = get_dist(curstate, auxinfo, roadinfo, j, i)
            if curdist < 0: 
                auxinfo[j][11][opside] = i
                newset.add(j)
                lcfolaux[20][lcside].remove(j)
                if curdist > maxdist: 
                    maxdist = curdist
                    minveh = j #minveh is the closest new lc side follower 
        curaux[20][lcside] = newset
        #update new lcside 
        if lcsidelane == 0 or lcsidelane == roadinfo[road][0]-1: 
            curaux[11][lcside] = ''
        else: 
            if minveh is not None: 
                curaux[11][lcside] = auxinfo[minveh][11][1]
                auxinfo[curaux[11][lcside]][20][opside].add(i)
            else: 
                #use guess and leadfol_find to get new lcside follower 
                guess = lcfolaux[11][lcside]
                unused, newlcsidefol = leadfol_find(curstate, auxinfo, roadinfo, i, guess)
                curaux[11][lcside] = newlcsidefol
                auxinfo[newlcsidefol][20][opside].add(i)
        
        #check for vehicles moving into same gap (check left/right leaders of the lcfol)
        
        #also at this point you would also want to reset cooperative and tactical states 
        
        #relaxation would also be calculated at this step
        
#        ###############
#        #update opposite side leader
#        opfol = curaux[11][opside]
#        if opfol == '':
#            pass
#        else:
#            if opfol == None:
#                oplead = roadinfo[road][6][opsidelane]
#            else: 
#                oplead = auxinfo[opfol][1]
#            if oplead is not None: 
#                auxinfo[oplead][11][lcside] = curaux[11][1] #opposite side LC side follower is current follower
#        
#        #update current leader
#        if curaux[1] == None: 
#            pass
#        else: 
#            auxinfo[curaux[1]][11][1] = curaux[11][1]
#            if curaux[11][lcside] == auxinfo[curaux[1]][11][lcside]:
#                auxinfo[curaux[1]][11][lcside] = i
#        
#        #update LC side leader
#        lcfol = curaux[11][lcside]
#        if lcfol == None: 
#            lclead = roadinfo[road][6][lcsidelane]
##            ####last in road updates #no these are wrong 
##            roadinfo[road][6][lcsidelane] = i #update last vehicle for road if necessary
##            roadinfo[road][6][lane] = curaux[1]
#        else: 
#            lclead = auxinfo[lcfol][1]
#            auxinfo[lcfol][1] = i  #update leader for lcfol 
#        if lclead is not None: 
#            auxinfo[lclead][11][opside] = curaux[11][1]
#            auxinfo[lclead][11][1] = i
#            
#        #update vehicle and its follower
#        fol = curaux[11][1]
#        if fol is not None: 
#            auxinfo[fol][1] = curaux[1]
#        curaux[1] = lclead
#        curaux[11][opside] = fol
#        curaux[11][1] = lcfol
#        
#        #update memory for current vehicle
#        curaux[16][-1].append(timeind)
#        curaux[16].append([lclead, timeind+1])
#        curaux[18][-1].append(timeind)
#        curaux[18].append([lcsidelane, timeind+1])
#        
#        #update memory for followers 
#        if fol is not None: 
#            auxinfo[fol][16][-1].append(timeind)
#            auxinfo[fol][16].append([auxinfo[fol][1], timeind+1])
#        if lcfol is not None: 
#            auxinfo[lcfol][16][-1].append(timeind)
#            auxinfo[lcfol][16].append([i, timeind+1])
#        
#        
#        #update new LC side 
#        #check if new LC side even exists - update the followers accordingly 
#        if lcsidelane ==0:
#            curaux[11][0] = ''
#        elif lcsidelane == roadinfo[road][0]:
#            curaux[11][2] = ''
#        else: 
#            newlcside = lcsidelane -1 if lca[i] == 'l' else lcsidelane + 1 #lane index for new side 
#            
#            #basically need to figure out the leader/follower on the new lc side because the leader
#            #needs to have its opposite side follower updated the follower is the update for the vehicle lc side
#            #need to get a guess for a vehicle we think it could be
#            if lclead == None: 
#                if lcfol == None: 
#                    newlcveh = roadinfo[road][6][newlcside]
#                newlcveh = auxinfo[lcfol][11][lcside]
#            else: 
#                newlcveh = auxinfo[lclead][11][lcside]
#            if newlcveh == None: 
#                newlcveh = roadinfo[road][6][newlcside]
#            
#            #find new lcside follower and leader, if any 
#            newlclead, newlcfol = leadfol_find(curstate, auxinfo, roadinfo, i, newlcveh)
#            
#            if newlclead != None: 
#                auxinfo[newlclead][11][opside] = i
#            curaux[11][lcside]= newlcfol
            
        
#        #update first in roads  
#        #in general this code has problems when vehicle may be first for several roads 
#        if roadinfo[road][6][lane] == i: 
#            roadinfo[road][6][lane] = curaux[1] #need to set before updating curaux[1]
#        if roadinfo[road][6][lcsidelane] == lclead: #this is not true if i is not on same road 
#            roadinfo[road][6][lcsidelane] == i
            
            
        
        
    #update all vehicles states 
    for i in curstate.keys(): 
        update2nd(i, curstate, auxinfo, roadinfo, a[i], dt)
        
    
    
    #update all vehicles left and right followers    
    for i in curstate.keys():
        curaux = auxinfo[i]
        lfol, rfol = curaux[11][0], curaux[11][2]
        if lfol == '' or type(lfol) == str:
            pass
        #you could calculate headway here, but this is faster and works except in very weird edge case 
        #may just want to use getdist 
        elif curstate[i][0] < curstate[lfol][0] and curaux[3] == auxinfo[lfol][3]: 
            #update vehicle
            curaux[11][0] = auxinfo[lfol][11][1]
            auxinfo[curaux[11][0]][20][2].add(i)
            auxinfo[lfol][20][2].remove(i)
            
            auxinfo[auxinfo[lfol][11][2]][20][0].remove(lfol)
            auxinfo[lfol][11][2] = i
            curaux[20][0].add(lfol)
            

        if rfol == '' or type(rfol) == str:
            pass
        elif curstate[i][0] < curstate[rfol][0] and curaux[3] == auxinfo[rfol][3]: 
            curaux[11][2] = auxinfo[rfol][11][1]
            auxinfo[curaux[11][2]][20][0].add(i)
            auxinfo[rfol][20][0].remove(i)
            
            auxinfo[auxinfo[rfol][11][0]][20][2].remove(rfol)
            auxinfo[rfol][11][0] = i
            curaux[20][2].add(rfol)
            

            
    #check if roads change
    dellist = []
    for i in curstate.keys():
        if curstate[i][0] > roadinfo[auxinfo[i][3]][2]: #roads change 
            curaux = auxinfo[i]
            newroad, newlane = roadinfo[curaux[3]][1][curaux[2]]
            if newroad == None: #vehicle reaches end - remove from simulation
                #update follower's lead
                lfol, fol, rfol = auxinfo[i][11][:] #getting key error? some things not being removed right? 
                if lfol is not '': 
                    auxinfo[lfol][20][2].remove(i)
                if rfol is not '':
                    auxinfo[rfol][20][0].remove(i)
                auxinfo[fol][1] = None
                #update memory
                auxinfo[fol][16][-1].append(timeind)
                auxinfo[fol][16].append([None,timeind+1])
                curaux[17][-1].append(timeind)
                curaux[18][-1].append(timeind)
                dellist.append(i)
                continue
#            newroad, newlane = newroad[0], newroad[1]
            #update memory 
            curaux[17][-1].append(timeind)
            curaux[17].append([newroad, timeind+1])
            curaux[18][-1].append(timeind)
            curaux[18].append([newlane, timeind+1])
            #update states
            curstate[i][0] += -roadinfo[curaux[3]][2]
            curaux[2], curaux[3] = newlane, newroad

            
#            #update road's first vehicle 
#            curfirst = roadinfo[newroad][6][newlane]
#            if auxinfo[curfirst][3] != curaux[3] or curstate[curfirst][0] > curstate[i][0]: 
#                roadinfo[newroad][6][newlane] = i 
                
            #update followers for vehicle 
            if newlane == 0: #new left side is null
                if curaux[11][0] != '': 
                    auxinfo[curaux[11][0]][20][2].remove(i)
                    curaux[11][0] = ''
            elif curaux[11][0] == '': #new change on left side
                newfolguess = roadinfo[newroad][newlane-1][8]
                newlead, newfol = leadfol_find(curstate, auxinfo, roadinfo, i, newfolguess)
                curaux[11][0] = newfol
                auxinfo[newfol][20][2].add(i)
            if newlane == roadinfo[road][0]: #same thing for other side
                if curaux[11][2] != '': 
                    auxinfo[curaux[11][2]][20][0].remove(i)
                    curaux[11][2] = ''
            elif curaux[11][2] == '': #new change on left side
                newfolguess = roadinfo[newroad][newlane+1][8]
                newlead, newfol = leadfol_find(curstate, auxinfo, roadinfo, i, newfolguess)
                curaux[11][2] = newfol
                auxinfo[newfol][20][0].add(i)
                
                #old code
#                if newlead is not None: 
#                    newleaddist = get_dist(curstate, auxinfo, roadinfo, i, newlead)
#                    if newleaddist > 0: 
#                        curaux[11][0] = auxinfo[newlead][11][1]
#                    else: 
#                        curaux[11][0] = newlead
#                else: 
#                    #need code to find the vehicle 
#                    pass
#            if newlane == roadinfo[road][0]: #same thing for other side 
#                curaux[11][2] = ''
#            elif curaux[11][2] is '': #new change on right side
#                pass
                
    for i in dellist: 
        del curstate[i]
    
    #keep special vehicles updated
    for i in roadinfo.keys():
        for count, j in enumerate(roadinfo[i][8]):
            if j == None: #some lanes don't have this quantity
                continue
            elif type(j) == str: #string type -> using anchor vehicle 
                if auxinfo[j][1] is not None: 
                    roadinfo[i][8][count] = auxinfo[j][1]
            elif j in lca: #vehicle changed lanes -> default to follower
                if lca[j] == 'l': #
                    newguess = auxinfo[j][11][2]
                else:
                    newguess = auxinfo[j][11][0]
                if newguess == '': #special case where we can't find a vehicle to use as update
                    #defaults to anchor vehicle; in general could use better heuristic in rare corner case
                    #(e.g. check left/right of follower)
                    roadinfo[i][8][count] = roadinfo[i][6][count] 
                else:
                    roadinfo[i][8][count] = newguess
            else: #regular update 
                #check if vehicle passes threshold or moved onto new road 
                if auxinfo[j][3] == i: 
                    roadinfo[i][8][count] = auxinfo[j][11][1]
            
    

def update2nd(i, curstate,  auxinfo, roadinfo, a, dt):
    curstate[i][0] += dt*a[0]
    curstate[i][1] += dt*a[1]
    
    #no you can't compute headway here - this is bug 
    lead = auxinfo[i][1]
    if lead is not None: 
        curstate[i][2] = get_headway(curstate, auxinfo, roadinfo, i, lead)
    
    return curstate
    
def leadfol_find(curstate, auxinfo, roadinfo, veh, guess):
    #guess is a vehicle which might either be the new lcside leader or follower of veh. 
    #we assume that you don't guess None, and if you do then it means there are no  leader/follower
    #returns lcside leader, follower, in that order. 
    
    #this is bugged see lane.leadfol_find method
    if guess == None: 
        return None, None
    else: 
        hd = get_dist2(curstate, auxinfo, roadinfo, guess, veh)
        if hd < 0: 
            nextguess = auxinfo[guess][1]
            if nextguess == None: 
                return nextguess, guess
            nexthd = get_dist2(curstate, auxinfo, roadinfo, nextguess, veh)
            while nexthd < 0: 
                guess = nextguess
                nextguess = auxinfo[guess][1]
                if nextguess == None: 
                    return nextguess, guess
                nexthd = get_dist2(curstate, auxinfo,roadinfo,nextguess,veh)
            return nextguess, guess
        else:
            nextguess = auxinfo[guess][11][1]
            if nextguess == None: 
                return guess, nextguess
            nexthd = get_dist2(curstate, auxinfo, roadinfo, nextguess, veh)
            while nexthd > 0: 
                guess = nextguess
                nextguess = auxinfo[guess][11][1]
                if nextguess == None: 
                    return guess, nextguess
                nexthd = get_dist2(curstate, auxinfo, roadinfo, nextguess, veh)
        
            return guess, nextguess
        
def increment_inflow2(curstate, auxinfo, roadinfo, timeind, dt, defaultspeed = 5, chkhd = 15):
    #hacky solution for now - refer to notes BC3- 1. 
    #defaultspeed and chkhd magic numbers 
    for i in roadinfo.keys(): 
        #increment flow buffer
        for count, j in enumerate(roadinfo[i][3]):
            if j == None: 
                continue
            else: 
                roadinfo[i][5][count] += dt*j[timeind]
                #check if we need to add vehicle
                add = False
                if roadinfo[i][5][count]>= 1: 
                    anchor = roadinfo[i][6][count]
                    lead = auxinfo[anchor][1]
                    if lead is None: 
                        add = True
                        hd = None
                    else: 
                        hd = get_dist2(curstate, auxinfo, roadinfo, anchor, lead ) - auxinfo[lead][4]
                        if hd > chkhd: 
                            add = True
                if add: 
                    newind = list(curstate.keys())
                    newind = newind[-1]+1 if len(newind)>0 else 0
                    curp = [23, 1.2, 2, 1.1, 1.5]
                    curLC = [-2, .1, .2, .2, .2, 0]
                    curp[0] += np.random.rand()*20-10
                    curstate[newind] = [0, defaultspeed, hd]
                    auxinfo[newind] = [[None,False], lead, count, i, 3, curp, IDM_b3_b, std_CF, curLC, None, None, [None, None, None], 
                            [], timeind+1, [], [], [], [], [], [], [set(), None, set()]]
                    
                    #update the followers of leader for [20]
                    if lead is not None: 
                        auxinfo[lead][11][1] = newind
                    for k in auxinfo[anchor][20][0]: 
                        auxinfo[k][11][2] = newind
                    auxinfo[newind][20][0] = auxinfo[anchor][20][0]
                    auxinfo[anchor][20][0] = set()
                    
                    for k in auxinfo[anchor][20][2]: 
                        auxinfo[k][11][0] = newind
                    auxinfo[newind][20][2] = auxinfo[anchor][20][2]
                    auxinfo[anchor][20][2] = set()
                    
                    auxinfo[anchor][1] = newind
                    auxinfo[anchor][16][-1].append(timeind)
                    auxinfo[anchor][16].append([newind, timeind+1])
                    
                    #update followers in [11]
                    if count == 0: 
                        auxinfo[newind][11][0] = ''
                    else: 
                        leftanchor = roadinfo[i][6][count-1]
                        auxinfo[newind][11][0] = leftanchor
                        auxinfo[leftanchor][20][2].add(newind)
                    auxinfo[newind][11][1] = anchor
                    if count == roadinfo[i][0]-1: 
                        auxinfo[newind][11][2] = ''
                    else: 
                        rightanchor = roadinfo[i][6][count+1]
                        auxinfo[newind][11][2] = rightanchor
                        auxinfo[rightanchor][20][0].add(newind)
                    
                    #initialize memory 
                    auxinfo[newind][16].append([lead, timeind+1])
                    auxinfo[newind][17].append([i, timeind+1])
                    auxinfo[newind][18].append([count, timeind+1])
        
        
def simulate_sn(curstate, auxinfo, roadinfo, modelinfo, timesteps = 1000, dt = .25, starttime = 0):
    """
    simulate on a simple network (sn = simple network)
    """
    sim = {i: curstate[i] for i in curstate.keys()}
    
    for j in range(timesteps): 
        simulate_step2(curstate, auxinfo, roadinfo, modelinfo, update_sn, starttime + j, dt)
        
        for i in curstate.keys(): 
            if i in sim.keys(): 
                sim[i].append(curstate[i].copy())
            else: 
                sim[i] = [curstate[i].copy()]
    
    return sim, curstate, auxinfo, roadinfo, starttime + j+1


########################end code for simple network#################
    
def update_net(vehicles, lc_actions, inflow_lanes, merge_lanes, vehid, timeind, dt): 
    #update followers/leaders for all lane changes 
    for veh in lc_actions.keys(): 
        update_change(veh, timeind) #this cannot be done in parralel
        
        #apply relaxation 
        new_relaxation(veh, timeind, dt)
    
    #update all states, memory and headway 
    for veh in vehicles: 
        veh.update(timeind, dt)
    for veh in vehicles:
        if veh.lead is not None: 
            veh.hd = veh.lane.get_headway(veh, veh.lead)
        
    #update left and right followers
    for veh in vehicles:
        update_lrfol(veh)
        
    #inflow goes here
    for lane in inflow_lanes: 
        vehid = lane.increment_inflow(vehicles, vehid, timeind, dt)
    
    #update merge_anchors
    for lane in merge_lanes:
        for i in range(len(lane.merge_anchors)):
            veh, pos = lane.merge_anchors[i]
            if pos == None: #some merge anchors may not need to be updated 
                continue
            if veh.cf_parameters == None:  
                lead = veh.lead
                if lead is not None and lead.lane is lane and lead.pos < pos:
                    lane.merge_anchors[i][0] = lead
                    
            elif veh in lc_actions: 
                if lc_actions[veh] == 'l': 
                    lane.merge_anchors[i][0] = veh.rfol
                else: 
                    lane.merge_anchors[i][0] = veh.lfol
            
            else:
                if veh.pos > pos: 
                    lane.merge_anchors[i][0] = veh.fol
                    
    #update roads and routes last
                    
    return 

        
def update_lrfol(veh):
    lfol, rfol = veh.lfol, veh.rfol
    if lfol == '':
        pass
    elif veh.lane.get_dist(veh,lfol) > 0: 
        veh.lfol = lfol.fol
        veh.lfol.rlead.add(veh)
        lfol.rlead.remove(veh)
        
        lfol.rfol.llead.remove(lfol)
        lfol.rfol = veh
        veh.llead.add(lfol)
        
    if rfol == '':
        pass
    elif veh.lane.get_dist(veh,lfol) > 0: 
        veh.lfol = lfol.fol
        veh.lfol.rlead.add(veh)
        lfol.rlead.remove(veh)
        
        lfol.rfol.llead.remove(lfol)
        lfol.rfol = veh
        veh.llead.add(lfol)
        
        
        
    

def new_relaxation(veh,timeind, dt):
    rp = veh.relaxp
    if veh.lead == None or rp == None: 
        return
    olds = veh.hd
    news = veh.lane.get_headway(veh, veh.lead)
    if olds is None: 
        olds = veh.get_eql(veh.speed)
    
    relaxamount = olds-news
    relaxlen = math.ceil(rp/dt) - 1
    curr =  relaxamount*np.linspace(1 - dt/rp, 1 - dt/rp*relaxlen,relaxlen)
    
    if veh.in_relax: #add to existing relax
        curlen = len(veh.relax)
        newend = timeind + relaxlen #time index when relax ends 
        newrelax = np.zeros((newend - veh.relax_start+1))
        newrelax[0:curlen] = veh.relax
        newrelax[timeind-veh.relax_start+1:] += curr
        veh.relax = newrelax
    else: #create new relax
        veh.in_relax = True
        veh.relax_start = timeind + 1
        veh.relax = curr
    
    return

def update_change(veh, timeind): 
    #logic for updating - logic is complicated because we avoid doing any sorts - faster this way 
    #no check for vehicles moving into same gap
    #no cooperative tactical components 
    
    #initialization 
    lane = veh.lane
    if veh.lc == 'l':
        lcsidefol, opsidefol, lcsidelead, opsidelead = 'lfol', 'rfol', 'llead', 'rlead'
        lcsidelane = lane.connect_left
        newlcsidelane = lcsidelane.connect_left
        
    else: 
        lcsidefol, opsidefol, lcsidelead, opsidelead = 'rfol', 'lfol', 'rlead', 'llead'
        lcsidelane = lane.connect_right
        newlcsidelane = lcsidelane.connect_right
    
    
    #update current leader
    lead = veh.lead 
    fol = veh.fol
    if lead == None: 
        pass
    else: 
        lead.fol = fol
    
    #update opposite/lc side leaders
    for j in getattr(veh, opsidelead):
        setattr(j, lcsidefol, fol)
    for j in getattr(veh, lcsidelead):
        setattr(j, opsidefol, fol)
    
    #update follower
    getattr(fol,lcsidelead).update(getattr(veh, lcsidelead))
    getattr(fol,opsidelead).update(getattr(veh, opsidelead))
    fol.lead = lead
    fol.leadmem.append((lead, timeind+1))
    
    #update opposite side for vehicle 
    vehopsidefol = getattr(veh, opsidefol)
    if vehopsidefol != '': 
        getattr(vehopsidefol, lcsidelead).remove(veh)
    setattr(veh, opsidefol, fol)
    getattr(fol, lcsidelead).add(veh)
    #update cur lc side follower for vehicle 
    lcfol = getattr(veh, lcsidefol)
    lcfol.lead = veh
    lcfol.leadmem.append((veh, timeind+1))
    getattr(lcfol, opsidelead).remove(veh)
    veh.fol = lcfol
    #update lc side leader
    lclead = lcfol.lead
    veh.lead = lclead
    veh.leadmem.append((lclead, timeind+1))
    veh.lanemem.append((lcsidelane, timeind+1))
    veh.lane = lcsidelane
    if lclead is not None: 
        lclead.fol = veh
    #update for new left/right leaders - opside first 
    newleads = set()
    oldleads = getattr(lcfol, opsidelead)
    for j in oldleads.copy(): 
        curdist = lane.get_dist(j, veh)
        if curdist < 0: 
            setattr(j, lcsidefol, veh)
            newleads.add(j)
            oldleads.remove(j)
    setattr(veh, opsidelead, newleads)
    #lcside 
    newleads = set()
    oldleads = getattr(lcfol, lcsidelead)
    maxdist = -math.inf
    minveh = None
    for j in oldleads.copy():
        curdist = lane.get_dist(j, veh)
        if curdist < 0: 
            setattr(j, opsidefol, veh)
            newleads.add(j)
            oldleads.remove(j)
            if curdist > maxdist: 
                maxdist = curdist 
                minveh = j #minveh is the leader of new lc side follower 
    setattr(veh, lcsidelead, newleads)
    #update new lcside 
    if newlcsidelane: #new lcside is None
        setattr(veh, lcsidefol, '')
    else: 
        if minveh is not None: 
            setattr(veh, lcsidefol, minveh.fol)
            getattr(minveh.fol,opsidelead).add(veh)
        else: 
            guess = get_guess(lcfol, lclead, veh, lcsidefol, newlcsidelane)
            unused, newlcsidefol = lcsidelane.leadfol_find(veh, guess)
            setattr(veh, lcsidefol, newlcsidefol)
            getattr(newlcsidefol, opsidelead).add(veh)
            
    veh.lc = None 
            
    return 
        
def get_guess(lcfol, lclead, veh, lcsidefol, newlcsidelane):
    #need to find new lcside follower for veh
    guess = getattr(lcfol, lcsidefol)
    anchor = newlcsidelane.anchor
    if guess == '' or guess.lane.anchor is not anchor: 
        guess = getattr(lclead, lcsidefol)
        if guess == '' or guess.lane.anchor is not anchor: 
            guess = anchor
    return guess 

class simulation: 
    def __init__(): 
        pass
    
    def step(self):
        lc_actions = {}
        
        for veh in self.vehicles: 
            veh.action = veh.call_cf(veh.lead, veh.lane, timeind, dt, veh.in_relax)
            
        for veh in self.vehicles: 
            veh.call_lc(lc_actions, veh.check_lc, timeind, dt)
            
        #update function goes here 
        
def eql_wrapper(eqlfun, eql_type = 'v', bounds = (1e-10, 120), tol = .5):
    #eqlfun -> fun to wrap, needs call signature like (parameters, input, *args)
    #eql_type = 's' - if 'v', eqlfun takes in velocity and outputs headway. if 's', it takes in headway and outputs velocity
    #if eql_type = 'both', then eqlfun takes in an additional argument (parameters, input, input_type) and will return the other quantity
    #bounds = (1e-10, 120) - bounds used when eql_type != find. Should define an interval that the soln is in 
    #tol = .5 - if need to numerically invert the function, this is the tolerance used 
    if eql_type != 'both':
        def get_eql(self, x, input_type = 'v'):
            if input_type == eql_type: 
                return eqlfun(self.cf_parameters, x)
            elif input_type != eql_type: 
                def inveql(y):
                    return x - eqlfun(self.cf_parameters, y)
                ans = sc.root_scalar(inveql, bracket = bounds, xtol = tol, method = 'brentq')
                if ans['converged']: 
                    return ans['root']
                else: 
                    raise RuntimeError('could not invert provided equilibrium function')
    else:
        def get_eql(self, x, input_type = 'v'):
            return eqlfun(self.cf_parameters, x, input_type)
    
    return get_eql

def FD_wrapper(eqlfun):
    #returns flow based on provided equilibrium function 
    def get_flow(self, x, leadlen = None, input_type = 'v'):
        if leadlen == None: 
            lead = self.lead 
            if lead != None: 
                leadlen = lead.len
            else:
                leadlen = self.len
        if input_type == 'v':
            s = self.get_eql(x, input_type = input_type)
            return x / (s + leadlen)
        elif input_type == 's': 
            v = self.get_eql(x, input_type = input_type)
            return v / (s + leadlen)
    return get_flow
                
        
def invFD_wrapper(eqlfun, eql_type = 'v', bounds = (1e-10, 120), tol = .01):
    if eql_type != 'both':
        def inv_flow(self, x, leadlen = None, output_type = 'v', congested = True):
            if leadlen == None: 
                lead = self.lead 
            if lead != None: 
                leadlen = lead.len
            else:
                leadlen = self.len
                
            def maxfun(y):
                return -self.get_flow(y, leadlen = leadlen, input_type = eql_type)
            
            res = sc.minimize_scalar(maxfun, bracket = bounds, tol = tol)
            if res['converged']:
                if congested: 
                    invbounds = (bounds[0], res['x'])
                else:
                    invbounds = (res['x'], bounds[1])
            else:
                raise RuntimeError('could not find maximum flow')
                
                
            if eql_type == 'v':
                def invfun(y): 
                    return x - y/(self.get_eql(y, input_type = eql_type) + leadlen)
            elif eql_type == 's':
                def invfun(y):
                    return x - self.get_eql(y, input_type = eql_type)/(y+leadlen)
            
            ans = sc.root_scalar(invfun, bracket = invbounds, xtol = tol, method = 'brentq')
            
            if ans['converged']:
                if output_type == eql_type: 
                    return ans['root']
                elif output_type == 's':
                    return ans['root']/x - leadlen
                elif output_type == 'v':
                    return (ans['root']+leadlen)*x
            else: 
                raise RuntimeError('could not invert provided equilibrium function')
    else: 
        raise RuntimeError('not supported')
    
    return inv_flow
        
        
    


def CF_wrapper(cfmodel, acc_bounds = [-7,3]): 
    #acc_bounds controls [lower, upper] bounds on acceleration 
    #assumes a second order model which has inputs of (p, state), where state
    #is a list of all values needed, p is a list of parameters, and
    #output is a float giving the acceleration 
    
    #may want to change the call signature to simplify it but the general 
    #structure of call_cf makes sense 
    def call_cf(self, lead, lane, timeind, dt, userelax): 
        if lead is None: 
            acc = lane.call_downstream(self, timeind, dt)
            
        else:
            if userelax:
                currelax = self.relax[timeind - self.relax_start]
                self.hd += currelax #can add check to see if relaxed headway is too small
                acc = cfmodel(self.cf_parameters, [self.hd, self.speed, lead.speed])
                self.hd += -currelax
            else: 
                acc = cfmodel(self.cf_parameters, [self.hd, self.speed, lead.speed])
            
        if acc > acc_bounds[1]: 
            acc = 3
        elif acc < acc_bounds[0]: 
            acc = -7
        
        if self.speed + dt*acc < 0: 
            acc = -self.speed/dt
            
        return acc
    
    return call_cf

def call_lc_helper(lfol, veh, lcsidelane):
    #does headway calculation for new potential follower lfol (works for either side)
    #bug with lane used - needs to use correct lane 
    llead, llane = lfol.lead, lfol.lane
    if llead == None: 
        newlhd = lcsidelane.dist_to_end(veh)
        #note in this case lfol will not have its headway updated - 
        #for mobil this is OK but in general may need an extra headway calculation here 
        #e.g. lfol.hd = llane.dist_to_end(lfol)
    else: 
        newlhd = lcsidelane.get_headway(veh, llead)
    if lfol.cf_parameters == None:
        newlfolhd = 0
    else: 
        newlfolhd = llane.get_headway(lfol, veh)
    
    return llead, newlfolhd, newlhd
        

def LC_wrapper(lcmodel, get_fol = True, **kwargs): #userelax_cur = True, userelax_new = False
    #lcmodel - model to wrap. Assume it takes as input the vehicle, 
        #new left follower headway, new left headway, new right follower headway, new right headway, 
        #new follower headway (if get_fol is True), timeind, dt, *args, **kwargs
    #get_fol - lc model uses the current follower if True 
    #kwargs - keyword arguments which are passed to lcmodel 
    
    def call_lc(self, chk_lc, timeind, dt):
        lfol, rfol, lane = self.lane = self.lfol, self.rfol, self.lane
        if lfol== '' and rfol == '':
            return 
        
        if np.random.rand() < chk_lc: 
            return 
        
        if lfol != '': 
            llead, newlfolhd, newlhd = call_lc_helper(lfol, self, lane.connect_left)
        else:
            llead = newlfolhd = newlhd = None
        
        if rfol != '': 
            rlead, newrfolhd, newrhd = call_lc_helper(rfol, self, lane.connect_right)
        else:
            rlead = newrfolhd = newrhd = None
            
        if get_fol: 
            fol, lead = self.fol, self.lead
            if fol.cf_parameters == None: 
                newfolhd = 0 
            elif self.lead == None: 
                newfolhd = fol.lane.dist_to_end(fol)
            else: 
                newfolhd = fol.lane.get_headway(fol, lead)
                
            #do model call now 
            lcmodel(self, newlfolhd, newlhd, newrfolhd, newrhd, newfolhd, timeind, dt, 
                    lfol, llead, rfol, rlead, fol, lead, lane, **kwargs)
        else: 
            lcmodel(self, newlfolhd, newlhd, newrfolhd, newrhd, timeind, dt, 
                    lfol, llead, rfol, rlead, lane, **kwargs)
            
    return call_lc
        
class vehicle: 
    #2 options for implementing your own LC model and CF model - 
    #option 1 - we have decorators for book keeping work for the LC/CF functions, user specifies the model 
    #and what the vehicle object calls is the decorated model which will handle formatting/bookkeeping issues
    #the default class will use this design so users can write their own model in a standard format 
    #and directly feed that in (easier but more restrictive)
    
    #option 2 - you can inherit this class and write your own functions (more work but less restrictive)
    def __init__(self, vehid,lane, p, lcp, length = 2, relaxp = None,
                 cfmodel = None, free_cf = None, lcmodel = None, eqlfun = None, check_lc = .25): 
        self.vehid = vehid
        self.lane = lane
        self.road = lane.road

        #model parameters
        self.cf_parameters = p
        self.relaxp = relaxp
        self.length = length
        self.lc_parameters = lcp
        
        #leader/follower relationships
        self.lead = None
        self.fol = None
        self.lfol = None
        self.rfol = None
        self.llead = None
        self.rlead = None
        
        #state
        self.pos = None
        self.speed = None
        self.hd = None
        self.action = None
        
        #memory
        self.inittime= None
        self.endtime = None
        self.leadmem = []
        self.lanemem = []
        self.posmem = []
        self.speedmem = []
        self.relaxmem = []
        #will want lfol and rfol memory if you want gradient wrt LC parameters, probably need memory for lc output as well 
        #won't store headway to save a bit of memory
        
        self.in_relax = False
        self.relax = None
        self.relax_start = None
        
        self.check_lc = check_lc
        
        if cfmodel is not None: 
            self.call_cf = CF_wrapper(cfmodel)
        
        if free_cf is not None: 
            self.free_cf = staticmethod(free_cf)
            
        if eqlfun is not None: 
            self.get_eql = eql_wrapper(eqlfun)
            
        if lcmodel is not None: 
            self.call_lc = LC_wrapper(lcmodel)
            
    def __hash__(self):
        return self.vehid
        
    def __eq__(self, other):
        return self.vehid == other.vehid
    
    def __ne__(self, other):
        return not(self is other)
            
    def update(self, timeind, dt): 
        #update state
        self.pos += self.speed*dt
        self.speed += self.action*dt
        
        #update memory and relax
        self.posmem.append(self.pos)
        self.speedmem.append(self.speed)
        if self.in_relax:
            if timeind == self.relax_start + len(self.relax) - 1:
                self.in_relax = False
                self.relaxmem.append((self.relax_start, timeind, self.relax))
                
    
            

def downstream_wrapper(timeseries = None, starttimeind = 0, method = 'speed', congested = True):
    #downstream function -> method of lane, takes in (veh, timeind, dt)
    #and returns action (acceleration) for the vehicle 
    
    if method == 'speed':
        @staticmethod
        def call_downstream(veh, timeind, dt):
            speed = timeseries[timeind-starttimeind]
            return (speed - veh.speed)/dt
        return call_downstream
    
    elif method == 'free':
        @staticmethod
        def free_downstream(veh, *args):
            return veh.free_cf(veh.cf_parameters, veh.speed)
        return free_downstream
    
    elif method == 'flow':
        @staticmethod
        def call_downstream(veh, timeind, dt):
            flow = timeseries[timeind - starttimeind]
            speed = veh.inv_flow(flow, output_type = 'v', congested = congested)
            return (speed - veh.speed)/dt
        return call_downstream
    
#def free_downstream_wrapper(free_cf_model):
#    #this only works if all vehicles have same model - needs to call something vehicle specific
#    @staticmethod
#    def call_downstream(veh, *args):
#        return free_cf_model(veh.cf_parameters, veh.speed)

        
class anchor_vehicle:
    #anchor vehicles have cf_parameters as None 
    def __init__(self, lane, time, lfol = None, rfol = None, lead = None, rlead = set(), llead = set()):
        self.cf_parameters = None 
        self.lane = lane
        self.road = lane.road
        
        self.lfol = lfol #I think anchor vehicles just need the lead/llead/rlead attributes and none of the fol attributes
        self.rfol = rfol
        self.lead = lead
        self.rlead = rlead
        self.llead = llead
        
        self.pos = 0
        self.hd = 0 
        self.length = 0
        
        self.leadmem = [[lead,time]]
        
def get_inflow_wrapper(speed_fun, inflow_type = 'flow'):
    #to use the inflow functions provided, you need the following methods in the lane 
    # - get_inflow = 'flow' - accepts either timeseries of flow ('flow'), or timeseries of speed ('speed'). 
    #If giving speeds, the vehicle to be added needs a get_eql method
    # - generate_parameters (accepts no arguments, returns cf/lc_parameters, and all keyword arguments, 
    # for a new vehicle)
    
    #returns get_inflow function, which accepts timeind and returns the flow at that time
    
    if inflow_type == 'flow':
        def get_inflow(self, timeind):
            return speed_fun(timeind), None
    elif inflow_type == 'speed':
        def get_inflow(self, timeind):
            spd = speed_fun(timeind)
            lead = self.anchor.lead
            if lead is not None:
                leadlen = lead.len
            else:
                leadlen = self.newveh.len
            s = self.newveh.get_eql(spd, find = 's')
            return spd / (s + leadlen), spd
    elif inflow_type == 'congested':
        def get_inflow(self, timeind):
            lead = self.anchor.lead
            if lead is not None:
                leadlen = lead.len
                spd = lead.speed
            else:
                leadlen = self.newveh.len
                spd = speed_fun(timeind)
            s = self.newveh.get_eql(spd, find = 's')
            return spd / (s + leadlen), spd

    return get_inflow
        
def timeseries_wrapper(timeseries, starttimeind = 0):
    def out(timeind):
        return timeseries[timeind-starttimeind]
    return out
    
def eql_inflow_congested(lane, inflow, c = .8, check_gap = True):
    #suggested by treiber for congested conditions, requires to invert the inflow to obtain 
    #the steady state headway. the actual headway on the road must be at least c * the steady state headway 
    #for the vehicle to be added. 
    #if check_gap is False, we don't have to invert the flow, we will always just add at the equilibrium speed
    #the vehicle is added with a speed obtained from the equilibrium speed with the current headway 
    
    lead = lane.anchor.lead
    hd = lane.get_headway(lane.anchor, lead)
    if check_gap == True:
        se = lane.newveh.inv_flow(inflow, leadlen = lead.len, output_type = 's') #headway corresponding to current flow
    else:
        se = -math.inf
    if hd > c*se: #condition met
        spd = lane.veh.get_eql(hd, input_type = 's')
        return 0, spd, hd
    else:
        return None
    
def eql_inflow_free(lane, inflow):
    #suggested by treiber for free conditions, requires to invert the inflow to obtain 
    #the velocity 
    lead = lane.anchor.lead
    hd = lane.get_headway(lane.anchor, lead)
    spd = lane.newveh.inv_flow(inflow, leadlen = lead.len, output_type = 'v', congested = False) #speed corresponding to current flow
    return 0, spd, hd

def shifted_speed_inflow(lane, dt, shift = 1, accel_bound = -2):
    #gives the first speed based on the shifted speed of the lead vehicle (similar to newell model)
    #shift = 1 - shift in time, measured in real time 
    #accel_bound = -2 - if not None, the acceleration of the vehicle 
    #must be greater than the accel_bound. Otherwise, no such bound is enforced
    lead = lane.anchor.lead
    hd = lane.get_headway(lane.anchor, lead)
    if len(lead.speedmem)*dt < shift: 
        spd = lead.speedmem[0]
    else:
        ind = math.ceil(shift/dt)
        spd = lead.speedmem[-ind]
        
    if accel_bound is not None: 
        newveh = lane.newveh
        newveh.pos = 0
        newveh.spd = spd
        newveh.hd = hd
        acc = newveh.call_cf(lead, lane, None, dt, False)
        if acc > accel_bound: 
            return 0, spd, hd
        else: 
            return None
    
    return 0, spd, hd

def speed_inflow(lane, speed_fun, timeind, dt, accel_bound = -2):
    #gives the first speed based on the shifted speed of the lead vehicle (similar to newell model)
    #shift = 1 - shift in time, measured in real time 
    #accel_bound = -2 - if not None, the acceleration of the vehicle 
    #must be greater than the accel_bound. Otherwise, no such bound is enforced
    lead = lane.anchor.lead
    hd = lane.get_headway(lane.anchor, lead)
    spd = speed_fun(timeind)
        
    if accel_bound is not None: 
        newveh = lane.newveh
        newveh.pos = 0
        newveh.spd = spd
        newveh.hd = hd
        acc = newveh.call_cf(lead, lane, None, dt, False)
        if acc > accel_bound: 
            return 0, spd, hd
        else: 
            return None
    
    return 0, spd, hd 
    


def increment_inflow_wrapper(speed_fun = None, method = 'ceql', accel_bound = -2, check_gap = True, shift = 1, c = .8):
    #method = 'eql' vehicles have 0 acceleration when being added. The speed is defined by the vehicles
    #equilibrium function. Thus to use this method, the vehicle to be added must have a get_eql method 
    #(and use a model which predicts acceleration)
    #if check_gap = True, the headway must be at least as big as the equilibrium headway corresponding to 
    #the current inflow. 
    #method = 'shifted' - uses shifted speed 
    def increment_inflow(self, vehicles, vehid, timeind, dt):
        inflow, spd = self.get_inflow(timeind)
        self.inflow_buffer += inflow * dt
        
        if self.inflow_buffer >= 1: 
            if self.anchor.lead is None: 
                if spd is None: 
                    spd = speed_fun(timeind)
                out = (0, spd, None)
            elif method == 'ceql': 
                out = eql_inflow_congested(self, inflow, c = c, check_gap = check_gap)
            elif method == 'feql':
                out = eql_inflow_free(self, inflow)
            elif method == 'shifted':
                out = shifted_speed_inflow(self, dt, shift = shift, accel_bound = accel_bound)
            elif method == 'speed':
                out = speed_inflow(self, speed_fun, timeind, dt, accel_bound = accel_bound)
            
            if out == None:  
                return
            #add vehicle with the given initial conditions
            pos, speed, hd = out[0], out[1], out[2]
            newveh = lane.newveh
            lead = self.anchor.lead
            anchor = lane.anchor
            #initialize state
            self.pos = pos
            self.speed = speed
            self.hd= hd
            
            #initalize memory
            self.inittime = timeind+1
            self.leadmem.append((lead, timeind+1))
            self.lanemem.append((lane,timeind+1))
            self.posmem.append(pos)
            self.speedmem.append(speed)
            
            #update leader/follower relationships
            #leader relationships
            lead.fol = newveh
            newveh.lead = lead
            for rlead in anchor.rlead: 
                rlead.lfol = newveh
            newveh.rlead = anchor.rlead
            anchor.rlead = set()
            for llead in anchor.llead:
                llead.rfol = newveh
            newveh.llead = anchor.llead
            anchor.llead = set()
            
            #update anchor and follower relationships
            anchor.leadmem.append((newveh, timeind+1))
            anchor.lead = newveh
            if lane.connect_left == None:
                newveh.lfol = ''
            else:
                leftanchor = lane.connect_left.anchor
                newveh.lfol = leftanchor
                leftanchor.rlead.add(newveh)
            newveh.fol = anchor
            if lane.connect_right == None:
                newveh.rfol = ''
            else:
                rightanchor = lane.connect_right.anchor
                newveh.rfol = rightanchor
                rightanchor.llead.add(newveh)
            
            #initaialize route // TO DO
            
            self.inflow_buffer += -1
            vehicles.add(newveh)
        
            #create next vehicle
            cf_parameters, lc_parameters, kwargs = self.new_vehicle()
            newveh = vehicle(vehid, self, cf_parameters, lc_parameters, **kwargs)
            self.newveh = newveh
            vehid = vehid + 1
            
                
                
        return vehid
        
    return increment_inflow
        
    
class lane: 
    def __init__(self, length, connect_to=None, connect_from = None, connect_left = None, connect_right = None,
                 downstream = {}, increment_inflow = {}, get_inflow = {}, new_vehicle = None):
        self.length = length
        self.connect_to = connect_to
        self.connect_from = connect_from
        self.connect_left = connect_left
        self.connect_right = connect_right
        
        if downstream != {}:
            self.call_downstream = downstream_wrapper(**downstream)
            
        if increment_inflow != {}:
            self.increment_inflow = increment_inflow_wrapper(**increment_inflow)
            
        if get_inflow != {}:
            self.get_inflow = get_inflow_wrapper(**get_inflow)
            
        if new_vehicle != None:
            self.new_vehicle = staticmethod(new_vehicle)
        
        #todo - 
        #need function to initialize roads, which will make roadlen dictionary, 
        #enddist attribute, initialize special vehicles and anchor vehicles 
        #add roads and roadind attribute, handle routes 
    
    def get_headway(self, veh, lead): 
        #distance from front of vehicle to back of lead
        hd = lead.pos - veh.pos - lead.length
        if self.road != lead.road: 
            hd += self.roadlen[(self.road, lead.road)]
        return hd 
    
    def get_dist(self, veh, lead): 
        #distance from front of vehicle to front of lead
        dist = lead.pos-veh.pos
        if self.road != lead.road: 
            dist += self.roadlen[(self.road, lead.road)]
        return dist
            
    def dist_to_end(self, veh):
        #distance from front of vehicle to end of network 
        return self.enddist - veh.pos
    
    def leadfol_find(self, veh, guess):
        #given guess vehicle which is 'close' to veh, returns the leader, follower
        #in that order in the same lane as guess 
        #used to initialize the new lc side follower/leader when new lanes become available
        if guess == None: 
            return None, None
        else: 
            get_dist = self.get_dist
            hd = get_dist(guess, veh)
            if hd > 0: 
                nextguess = guess.lead 
                if nextguess == None:  #None -> reached end of network
                    return nextguess, guess
                nexthd = get_dist(nextguess, veh)
                while nexthd > 0: 
                    guess = nextguess 
                    nextguess = guess.lead
                    if nextguess == None:
                        return nextguess, guess
                    nexthd = get_dist(nextguess, veh)
                return nextguess, guess
            else: 
                nextguess = guess.fol
                if nextguess == None:
                    return guess, nextguess
                nexthd = get_dist(nextguess, veh)
                while nexthd < 0:
                    guess = nextguess
                    nextguess = guess.fol
                    if nextguess.cf_parameters == None: #reached anchor -> beginning of network
                        return guess, nextguess
                    nexthd = get_dist(nextguess,veh)
                return guess, nextguess
        
    
