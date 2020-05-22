
"""
@author: rlk268@cornell.edu

houses all the different models for simulation

models USED TO have the following call signature: 
    veh - list of state of vehicle model is being applied to
    lead - list of state for vehicle's leader
    p - parameters
    leadlen - length of lead vehicle
    *args - additional inputs should be stored in modelinfo dict, and passed through *args
    dt = .1 - timestep 
    
    they return the derivative of the state i.e. how to update in the next timestep 
current documentation is in jupyter notebook 
"""

import numpy as np 
import scipy.optimize as sc 
import math 

def IDM(p, state):
    #state = headway, velocity, lead velocity
    #p = parameters
    #returns acceleration 
    return p[3]*(1-(state[1]/p[0])**4-((p[2]+state[1]*p[1]+(state[1]*(state[1]-state[2]))/(2*(p[3]*p[4])**(1/2)))/(state[0]))**2)

def IDM_free(p, state):
    #state = velocity
    #p = parameters
    return p[3]*(1-(state/p[0])**4)

def IDM_eql(p, v):
    #input is p = parameters, v = velocity, output is s = headway corresponding to eql soln
    s = ((p[2]+p[1]*v)**2/(1- (v/p[0])**4))**.5
    return s 

def IDM_shift_eql(p, v, shiftp, state):
    #p = CF parameters
    #v = velocity 
    #shiftp = list of deceleration, acceleration parameters. eq'l at v goes to n times of normal, where n is the parameter
    #state = if state = 'decel' we use shiftp[0] else shiftp[1]
    
    #for IDM model with parameters p, returns an acceleration such that 
    # the eql headway at speed v will go to shiftp of normal. 
    #e.g. shiftp = 2 -> add an acceleration such that equilibrium headway goes to 2 times its normal 
    if state == 'decel':
        temp = shiftp[0]**2
    else:
        temp = shiftp[1]**2
        
    return (temp - 1)/temp*p[3]*(1 - (v/p[0])**4)
    


def mobil( veh, lc_actions, lside, rside, newlfolhd, newlhd, newrfolhd, newrhd, newfolhd, timeind, dt,
          userelax_cur = True, userelax_new = False, use_coop = True, use_tact = True):
    #LC parameters 
    #0 - safety criterion
    #1 - incentive criteria 
    #2 - politeness
    #3 - bias on left side 
    #4 - bias on right side
    #5 - probability of checking LC while in discretionary state (scaled by timestep)
    
    #naming convention - l/r = left/right respectively, current lane if no l/r
    #new indicates that it would be the configuration after potential change 
    
    p = veh.lc_parameters
    lincentive = rincentive = -math.inf
    
    #calculate cura, fola, newfola
    if not userelax_cur and veh.in_relax: 
        cura = veh.get_cf(veh.hd, veh.speed, veh.lead, veh.lane, timeind, dt, False)
    else: 
        cura = veh.acc #more generally could use a method to return acceleration 
    fola, newfola = mobil_helper(veh.fol, veh, veh.lead, newfolhd, timeind, dt, userelax_cur, userelax_new)
    
    #to compute left side: need to compute lfola, newlfola, newla
    if lside: 
        lfol = veh.lfol
        llead = lfol.lead
        lfola, newlfola = mobil_helper(lfol, llead, veh, newlfolhd, timeind, dt, userelax_cur, userelax_new)
        
        userelax = userelax_new and veh.in_relax
        newla = veh.get_cf(newlhd, veh.speed, llead, veh.llane, timeind, dt, userelax)
        
        lincentive = newla - cura + p[2]*(newlfola - lfola + newfola - fola) + p[3]
    
    #same for right side
    if rside: 
        rfol = veh.rfol
        rlead = rfol.lead
        rfola, newrfola = mobil_helper(rfol, rlead, veh, newrfolhd, timeind, dt, userelax_cur, userelax_new)
        
        userelax = userelax_new and veh.in_relax
        newra = veh.get_cf(newrhd, veh.speed, rlead, veh.rlane, timeind, dt, userelax)
                
        rincentive = newra - cura + p[2]*(newrfola - rfola + newfola - fola) + p[4]
        
    #determine which side we want to potentially intiate LC for 
    if rincentive > lincentive: 
        side = 'r'
        lctype = veh.r
        incentive = rincentive
        
        newhd = newrhd
        newlcsidehd = newrfolhd
        selfsafe = newra
        folsafe = newrfola
    else:
        side = 'l'
        lctype = veh.l
        incentive = lincentive
        
        newhd = newlhd
        newlcsidehd = newlfolhd
        selfsafe = newla
        folsafe = newlfola
    
    #determine if LC can be completed, and if not, determine if we want to enter cooperative or tactical states
    #if wanted to make the function for cooperative/tactical states modular, could return a flag and arguments 
    if lctype == 'discretionary':
        if incentive > p[1]: 
            if selfsafe > p[0] and folsafe > p[0]:
                lc_actions[veh] = side
            else: 
                coop_tact_model(veh, newhd, newlcsidehd, folsafe, selfsafe, side, lctype, use_coop = use_coop, use_tact = use_tact) 
    else: #mandatory state
        if selfsafe > p[0] and folsafe > p[0]:
            lc_actions[veh] = side
        else: 
            coop_tact_model(veh, newhd, newlcsidehd, folsafe, selfsafe, side, lctype, use_coop = use_coop, use_tact = use_tact)
        
    return 
    
def coop_tact_model(veh, newhd, newlcsidehd, folsafe, selfsafe, side, lctype, use_coop = True, use_tact = True):
    #cooperative/tactical model for lane changing
    
    #veh = vehicle to consider
    #newhd - new headway of vehicle if it changed lanes
    #new headway of lcside follower if vehicle changed lanes
    #folsafe - safety condition for follower (float, it must be above a set threshold to be considered safe)
    #selfsafe - safety condition for vehicle
    #side - 'l' or 'r' depending on which side veh wants to change
    #lctype - 'discretionary' or 'mandatory'
    #use_coop - apply cooperative model
    #use_tact - apply tactical model
    
    #explanation of model###########################
    #first we assume that we can give vehicles one of two commands - accelerate or decelerate. 
    #there are three possible options - cooperation and tactical, or only cooperation, or only tactical
    
    #in the tactical model, first we check the safety conditions to see what is preventing us from changing (either lcside fol or lcside lead). 
    #if both are violating safety, and the lcside leader is faster than vehicle, then 
    #the vehicle gets deceleration to try to change behind them. If vehicle is faster than lcside leader, 
    #then the vehicle gets acceleration to try to overtake. 
    #if only one is violating safety, the vehicle moves in a way to prevent that violation. Meaning - 
    #if only the follower's safety is violated, the vehicle accelerates 
    #if the vehicle's own safety is violated; the vehicle decelerates
    #the tactical model only modifies the acceleration of veh. 
    
    #in the cooperative model, we try to identify a cooperating vehicle.
    #a cooperating vehicle always gets a deceleration added so that it will give extra space 
    #If the cooperation is applied without tactical, 
    #then the cooperating vehicle must be the lcside follower, and the newlcsidehd must be > 0
    #if cooperation is applied with tactical,
    #then in addition to the above, it's also possible the cooperating vehicle is the lcside follower's follower, 
    #where additionally the lcsidehd is < 0. Note that in this second case, the lcside follower is directly to the 
    #side of vehicle, and that is why this case is distinct.
    #In the first case where the cooperating vehicle is the lcside follower, the tactical model is applied as normal.
    #In the second case, since the issue is the the lcside follower is directly blocking the vehicle, 
    #the vehicle accelerates if the lcside follower has a slower speed than vehicle, and decelerates otherwise.
    
    #when a vehicle requests cooperation, it has to additionally fulfill a condition which simulates
    #the choice of the cooperating vehicle. ALl vehicles have a innate probability (coopp attribute) of 
    #cooperating, and for a discretionary LC, this innate probability controls whether or not the cooperation is accepted.
    #For a mandatory LC, vehicle can add to this probability, which simulates vehicles forcing the cooperation. 
    #vehicles have a LC_urgency attribute which is updated upon initiating a mandatory change. 
    #LC_urgency is a tuple of two positions, at the first position, only the follower's innate cooperation probability
    #is used. at the second position, the follower is always forced to cooperate, even if it has 0 innate cooperation probability
    
    #when a vehicle has cooperation or tactical components applied, it has a lcside attribute which is 
    #set to either 'l' or 'r' instead of None. This marks the vehicle as having the coop/tact added, and 
    #will make it so the vehicle attempts to complete the LC at every timestep even if its only a discretionary change. 
    #vehicles also have a coop_veh attribute which stores the cooperating vehicle. 
    #A cooperating vehicle does not have any attribute marking it as such
    
    #the LC_urgency attribute needs to be set whenever a mandatory route event begins. 
    #the lcside, coop_veh, and lc_urgency attributes need to be reset to None 
    #whenever the change is completed. 
    #in the road event where a lane ends on the side == lcside, then lcside and coop_veh need to be reset to None
    ###################################################
    #clearly it would be possible to modify different things, such as how the acceleration modifications
    #are obtained, and changing the conditions for entering/exiting the cooperative/tactical conditions
    #in particular we might want to add extra conditions for entering cooperative state
    tact = False
    if side == 'l': 
        lcsidefol = veh.lfol
    else:
        lcsidefol = veh.rfol
    
    if use_coop and use_tact: 
        coop_veh = veh.coop_veh
        if coop_veh != None: 
            #2 possible cases here 
            if coop_veh is lcsidefol: 
                coop_veh.acc += coop_veh.shift_eql(coop_veh.cf_parameters, coop_veh.speed, coop_veh.shiftp, 'decel')
                tact = True
            else: #check condition for coop_veh to be valid in case where coop_veh = lcsidefol.fol
                if coop_veh is lcsidefol.fol and newlcsidehd < 0:
                    coop_veh.acc += coop_veh.shift_eql(coop_veh.cf_parameters, coop_veh.speed, coop_veh.shiftp, 'decel')
                    if lcsidefol.speed > veh.speed: 
                        tactstate = 'decel'
                    else: 
                        tactstate = 'accel'
                    veh.acc += veh.shift_eql(veh.cf_parameters, veh.speed, veh.shiftp, tactstate)
                else: #cooperation is not valid -> reset
                    veh.lcside = None
                    veh.coop_veh = None
                    
        else: #see if we can get vehicle to cooperate
            tact = True
            if newlcsidehd < 0:
                coop_veh = lcsidefol.fol
            else:
                coop_veh = lcsidefol
                
            if coop_veh.cf_parameters != None: 
                temp = coop_veh.coopp
                if lctype == 'mandatory': 
                    start, end = veh.lc_urgency[:]
                    temp += (veh.pos - start)/(end - start)
                
                if temp >= 1 or np.random.rand() < temp: 
                    veh.coop_veh = coop_veh
                    veh.lcside = side
                    #apply cooperation
                    coop_veh.acc += coop_veh.shift_eql(coop_veh.cf_parameters, coop_veh.speed, coop_veh.shiftp, 'decel')
                    #apply tactical 
                    if coop_veh is not lcsidefol: #in this case we need to manually apply; otherwise we go to normal tactical model (tact = True)
                        tact = False
                        if lcsidefol.speed > veh.speed: 
                            tactstate = 'decel'
                        else:
                            tactstate = 'accel'
                        veh.acc += veh.shift_eql(veh.cf_parameters, veh.speed, veh.shiftp, tactstate)
            
            
    elif use_coop and not use_tact:
        
        coop_veh = veh.coop_veh
        if coop_veh != None:
            if coop_veh is lcsidefol and newhd > 0: #conditions for cooperation when there is no tactical
                coop_veh.acc += coop_veh.shift_eql(coop_veh.cf_parameters, coop_veh.speed, coop_veh.shiftp, 'decel')
            else: #cooperating vehicle not valid -> reset 
                veh.lcside = None
                veh.coop_veh = None
        else: #see if we can get vehicle to cooperate
            if newhd > 0 and lcsidefol.cf_parameters != None: #vehicle is valid
                #check cooperation condition 
                temp = lcsidefol.coopp
                if lctype == 'mandatory': 
                    start, end = veh.lc_urgency[:]
                    temp += (veh.pos - start)/(end - start)
                    
                if temp >= 1 or np.random.rand() < temp: #cooperation agreed/forced -> add cooperation
                    veh.coop_veh = lcsidefol
                    veh.lcside = side
                    #apply cooperation 
                    lcsidefol.acc += lcsidefol.shift_eql(lcsidefol.cf_parameters, lcsidefol.speed, lcsidefol.shiftp, 'decel')
                    
    elif tact or (not use_coop and use_tact):
        #mark side if not already 
        if veh.lcside == None:
            veh.lcside = side
        #find vehicle which is preventing change - if both lcsidefol and lcsidelead are prventing, 
        #default to looking at the lcsidelead
        safe = veh.lc_parameters[0]
        if folsafe < safe: 
            if selfsafe < safe: #both unsafe
                if lcsidefol.lead.speed > veh.speed:
                    tactstate = 'decel'
                else:
                    tactstate = 'accel'
            else:  #only follower unsafe
                tactstate = 'accel'
        else: #only leader unsafe
            tactstate = 'decel'
        
#        if tactveh == None: #may need this in very weird edge case where you are at boundary but this is probably just overkill to include
#            return
        
        veh.acc += veh.shift_eql(veh.cf_parameters, veh.speed, veh.shiftp, tactstate)
        
        
        
        
    
def mobil_helper(fol, curlead, newlead, newhd, timeind, dt, userelax_cur, userelax_new):
    #fol is assumed to follow curlead in the current configuration, in potential 
    #new configuration it would follow newlead
    if fol.cf_parameters == None: 
        fola = 0
        newfola = 0
    else: 
        if not userelax_cur and fol.in_relax:
            fola = fol.get_cf(fol.hd, fol.speed, curlead, fol.lane, timeind, dt, False)
        else: 
            fola = fol.acc
            
        userelax = userelax_new and fol.in_relax
        newfola = fol.get_cf(newhd, fol.speed, newlead, fol.lane, timeind, dt, userelax)
        
    return  fola, newfola


def IDM_parameters(*args): 
    cf_parameters = [27, 1.2, 2, 1.1, 1.5] #note speed is supposed to be in m/s
    cf_parameters[0] += np.random.rand()*6
    # cf_parameters[0] += np.random.rand()*25-15 #give vehicles very different speeds for testing purposes
    lc_parameters = [-1.5, .3, .2, 0, .1, .5]
    
    return cf_parameters, lc_parameters
    

#####################stuff for older code 

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

def zhangkim(p, s, leadv):
    #refer to car following theory for multiphase vehicular traffic flow by zhang and kim, model c
    #delay is p[0], then in order, parameters are S0, S1, v_f, h_1
    #recall reg = 0 is reserved for shifted end
    #s is the delayed headway, i.e. lead(t - tau) - lead_len - x(t - tau) 
    #leadv is the delayed velocity 
    
    if s >= p[2]:
        out = p[3]
    
    elif s < p[1]:
        out = s/p[4]
    
    else: 
        if leadv[1] >= p[3]:
            out = p[3]
        else: 
            out = s/p[4]
    
    return out


    
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

def drl_reward(nextstate, vavg, decay = .95, penalty = 1):
#    reward = 0
    for i in [8,9,10]:
#        vavg[i] = vavg[i]*decay + (1 - decay)*nextstate[i][1] #update exponential average of velocity
        vavg[i]= nextstate[i][1]
#        cur = -(nextstate[i][1] - vavg[i])**2 + vavg[i]  #penalize oscillations, reward high average
#        if nextstate[i][2] < 1: 
#            cur = cur - penalty*(2**(-5*(nextstate[i][2]-.2)) - 1) #if headway is small, we give a penalty
#            cur = cur - penalty*(nextstate[i][2]-1)**2 #here is a smaller penalty
#        reward += cur
    return 1+(vavg[9]+vavg[10])/10, vavg

def drl_reward4(nextstate, vavg, decay = .95, penalty = 1):
    reward = []
    for i in nextstate.keys():
        vavg[i] = vavg[i]*decay + (1 - decay)*nextstate[i][1] #update exponential average of velocity
        cur = -(nextstate[i][1] - vavg[i])**2 + vavg[i]  #penalize oscillations, reward high average
        if nextstate[i][2] < .2: 
#            cur = cur - penalty*(2**(-5*(nextstate[i][2]-.2)) - 1) #if headway is small, we give a penalty
            cur = cur - penalty*(nextstate[i][2]-.2)**2 #here is a smaller penalty
        reward.append(cur)
    return np.average(reward)+4, vavg

def drl_reward2(nextstate, vavg, decay = .95, lowereql = 11, eql = 15):
    vlist = [i[1] for i in nextstate.values()]
    vavg = np.average(vlist)
    return 1+ np.interp(vavg, (lowereql, eql), (0, 1)), vavg

def drl_reward3(nextstate, vavg, decay = .95, lowereql = 0, eql = 15):
    vlist = [i[1] for i in nextstate.values()]
    vavg = np.average(vlist)
    return 1+ 1/30*vavg**3, vavg

def drl_reward5(nextstate, vavg, decay = .95, lowereql = 0, eql = 15, avid = 9):
#    vlist = [i[1] for i in nextstate.values()]
    for i in [avid]:
        vavg[i] = vavg[i]*decay + (1 - decay)*nextstate[i][1]
    return 1+1/15*nextstate[avid][1]**2 - 1/5*(nextstate[avid][1] - vavg[avid])**2 , vavg

def drl_reward8(nextstate, vavg, decay = .95, lowereql = 11, eql = 15, avid = 9):
    vlist = [i[1] for i in nextstate.values()]
    vavg = np.average(vlist)
    hdpenalty = 0
    hd = nextstate[avid][2]
    if hd < 1.75:
        hdpenalty = -1.5
    return 1+ np.interp(vavg, (lowereql, eql), (0, 5)) + vlist[avid] +hdpenalty, vavg #1 + np.interp(vavg, (lowereql, eql), (0, 1)) + vlist[avid],

def drl_reward88(nextstate, vavg, decay = .95, lowereql = 11, eql = 15, avid = 9):
    vlist = [i[1] for i in nextstate.values()]
    vavg[avid] = vavg[avid]*decay + (1 - decay)*nextstate[avid][1] #update exponential average of velocity
#    vavg = np.average(vlist)
    #constant reward + strictly positive reward when going 15 + strictly positive reward for minimizing oscillations
#    return 1 + (-(min([vlist[avid],30]) - 15)**2+15**2)/30 + (-(min([vavg[avid],30]) - 15)**2+15**2)/30, vavg
    return 1 + (-(min([vlist[avid],30]) - 15)**2+15**2)/30, vavg
#term to penalize oscillations? 
#    return 1 + (-(min([vlist[avid],30]) - 15)**2+15**2)/30 + (-(vlist[avid] - vavg[avid])**2+15**2)/30, vavg

def drl_reward7(nextstate, vavg, decay = .95, lowereql = 0, eql = 15, avid = 9):
#    vlist = [i[1] for i in nextstate.values()]
#    vavg[avid] =  = np.average(vlist)
    return 1+1/5*nextstate[avid][1], vavg
        

def drl_reward6(nextstate, vavg, decay = .95, penalty = 1):
    reward = 0
    for i in [9]:
        vavg[i] = vavg[i]*decay + (1 - decay)*nextstate[i][1] #update exponential average of velocity
        cur = -5*(nextstate[i][1] - vavg[i])**2 + vavg[i]**2  #penalize oscillations, reward high average
        if nextstate[i][2] < .2: 
#            cur = cur - penalty*(2**(-5*(nextstate[i][2]-.2)) - 1) #if headway is small, we give a penalty
            cur = cur - penalty*(nextstate[i][2]-.2)**2 #here is a smaller penalty
        reward += cur
    return reward, vavg

def drl_reward9(nextstate, vavg, decay = .95, penalty = 1):
    reward = 0
    a = []
#    for i in [9]:
#        vavg[i] = vavg[i]*decay + (1 - decay)*nextstate[i][1] #update exponential average of velocity
#        cur = -5*(nextstate[i][1] - vavg[i])**2 + vavg[i]**2  #penalize oscillations, reward high average
#        if nextstate[i][2] < .2: 
##            cur = cur - penalty*(2**(-5*(nextstate[i][2]-.2)) - 1) #if headway is small, we give a penalty
#            cur = cur - penalty*(nextstate[i][2]-.2)**2 #here is a smaller penalty
#        reward += cur
#    for i in [8,9,10]:
#        reward += (max(nextstate[i][1],0)-15)**2
#    return 1 - (reward**.5)/25.9807, vavg #25.9807 = ((15**2)*3)**.5 => max reward at all 15 speed, mapped to 0,1
    for i in [8,9,10]:
        reward += abs(nextstate[i][1]-15)/15
        a.append(nextstate[i][1])
    return 1.1 - reward/3 + np.average(a)/15, None
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
    if acc > 3:
        acc = 3
    elif acc < -7:
        acc = -7
    return [veh[1],acc]