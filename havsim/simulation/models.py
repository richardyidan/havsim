
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

def IDM_shift_eql(p, v, shift_parameters, state):
    #p = CF parameters
    #v = velocity
    #shift_parameters = list of deceleration, acceleration parameters. eq'l at v goes to n times of normal, where n is the parameter
    #state = if state = 'decel' we use shift_parameters[0] else shift_parameters[1]

    #for IDM model with parameters p, returns an acceleration such that
    # the eql headway at speed v will go to shift_parameters of normal.
    #e.g. shift_parameters = 2 -> add an acceleration such that equilibrium headway goes to 2 times its normal
    if state == 'decel':
        temp = shift_parameters[0]**2
    else:
        temp = shift_parameters[1]**2

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
        lctype = veh.r_lc
        incentive = rincentive

        newhd = newrhd
        newlcsidehd = newrfolhd
        selfsafe = newra
        folsafe = newrfola
    else:
        side = 'l'
        lctype = veh.l_lc
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
    #the choice of the cooperating vehicle. ALl vehicles have a innate probability (coop_parameters attribute) of
    #cooperating, and for a discretionary LC, this innate probability controls whether or not the cooperation is accepted.
    #For a mandatory LC, vehicle can add to this probability, which simulates vehicles forcing the cooperation.
    #vehicles have a LC_urgency attribute which is updated upon initiating a mandatory change.
    #LC_urgency is a tuple of two positions, at the first position, only the follower's innate cooperation probability
    #is used. at the second position, the follower is always forced to cooperate, even if it has 0 innate cooperation probability

    #when a vehicle has cooperation or tactical components applied, it has a lc_side attribute which is
    #set to either 'l' or 'r' instead of None. This marks the vehicle as having the coop/tact added, and
    #will make it so the vehicle attempts to complete the LC at every timestep even if its only a discretionary change.
    #vehicles also have a coop_veh attribute which stores the cooperating vehicle.
    #A cooperating vehicle does not have any attribute marking it as such

    #the LC_urgency attribute needs to be set whenever a mandatory route event begins.
    #the lc_side, coop_veh, and lc_urgency attributes need to be reset to None
    #whenever the change is completed.
    #in the road event where a lane ends on the side == lc_side, then lc_side and coop_veh need to be reset to None
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
                coop_veh.acc += coop_veh.shift_eql(coop_veh.cf_parameters, coop_veh.speed, coop_veh.shift_parameters, 'decel')
                tact = True
            else: #check condition for coop_veh to be valid in case where coop_veh = lcsidefol.fol
                if coop_veh is lcsidefol.fol and newlcsidehd < 0:
                    coop_veh.acc += coop_veh.shift_eql(coop_veh.cf_parameters, coop_veh.speed, coop_veh.shift_parameters, 'decel')
                    if lcsidefol.speed > veh.speed:
                        tactstate = 'decel'
                    else:
                        tactstate = 'accel'
                    veh.acc += veh.shift_eql(veh.cf_parameters, veh.speed, veh.shift_parameters, tactstate)
                else: #cooperation is not valid -> reset
                    veh.lc_side = None
                    veh.coop_veh = None

        else: #see if we can get vehicle to cooperate
            tact = True
            if newlcsidehd < 0:
                coop_veh = lcsidefol.fol
            else:
                coop_veh = lcsidefol

            if coop_veh.cf_parameters != None:
                temp = coop_veh.coop_parameters
                if lctype == 'mandatory':
                    start, end = veh.lc_urgency[:]
                    temp += (veh.pos - start)/(end - start)

                if temp >= 1 or np.random.rand() < temp:
                    veh.coop_veh = coop_veh
                    veh.lc_side = side
                    #apply cooperation
                    coop_veh.acc += coop_veh.shift_eql(coop_veh.cf_parameters, coop_veh.speed, coop_veh.shift_parameters, 'decel')
                    #apply tactical
                    if coop_veh is not lcsidefol: #in this case we need to manually apply; otherwise we go to normal tactical model (tact = True)
                        tact = False
                        if lcsidefol.speed > veh.speed:
                            tactstate = 'decel'
                        else:
                            tactstate = 'accel'
                        veh.acc += veh.shift_eql(veh.cf_parameters, veh.speed, veh.shift_parameters, tactstate)


    elif use_coop and not use_tact:

        coop_veh = veh.coop_veh
        if coop_veh != None:
            if coop_veh is lcsidefol and newhd > 0: #conditions for cooperation when there is no tactical
                coop_veh.acc += coop_veh.shift_eql(coop_veh.cf_parameters, coop_veh.speed, coop_veh.shift_parameters, 'decel')
            else: #cooperating vehicle not valid -> reset
                veh.lc_side = None
                veh.coop_veh = None
        else: #see if we can get vehicle to cooperate
            if newhd > 0 and lcsidefol.cf_parameters != None: #vehicle is valid
                #check cooperation condition
                temp = lcsidefol.coop_parameters
                if lctype == 'mandatory':
                    start, end = veh.lc_urgency[:]
                    temp += (veh.pos - start)/(end - start)

                if temp >= 1 or np.random.rand() < temp: #cooperation agreed/forced -> add cooperation
                    veh.coop_veh = lcsidefol
                    veh.lc_side = side
                    #apply cooperation
                    lcsidefol.acc += lcsidefol.shift_eql(lcsidefol.cf_parameters, lcsidefol.speed, lcsidefol.shift_parameters, 'decel')

    elif tact or (not use_coop and use_tact):
        #mark side if not already
        if veh.lc_side == None:
            veh.lc_side = side
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

        veh.acc += veh.shift_eql(veh.cf_parameters, veh.speed, veh.shift_parameters, tactstate)





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
    lc_parameters = [-1.5, .2, .2, 0, .1, .5]

    return cf_parameters, lc_parameters
