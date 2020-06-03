"""Houses all the different models for simulation."""

import math
import numpy as np


def IDM(p, state):
    """Intelligent Driver Model (IDM), second order ODE.

    Args:
        p: parameters - [max speed, comfortable time headway, jam spacing, max comfortable acceleration,
                         emergency acceleration]
        state: list of [headway, self velocity, leader velocity]

    Returns:
        acceleration
    """
    return p[3]*(1-(state[1]/p[0])**4-((p[2]+state[1]*p[1]+(state[1]*(state[1]-state[2]))
                                        / (2*(p[3]*p[4])**(1/2)))/(state[0]))**2)


def IDM_free(p, state):
    """Free flow model for IDM, state = self velocity, p = parameters. Returns acceleration."""
    return p[3]*(1-(state/p[0])**4)


def IDM_eql(p, v):
    """Equilibrium solution for IDM, v = velocity, p = parameters. Returns headway corresponding to eql."""
    s = ((p[2]+p[1]*v)**2/(1 - (v/p[0])**4))**.5
    return s


def IDM_shift_eql(p, v, shift_parameters, state):
    """Calculates an acceleration which shifts the equilibrium solution for IDM.

    Args:
        p: parameters
        v: velocity
        shift_parameters: list of two parameters, 0 index is 'decel' which is > 1, 1 index is 'accel' which
            is < 1. Equilibrium goes to n*s, where s is the old equilibrium, and n is the parameter.
            E.g. shift_parameters = [2, .4], if state = 'decel' we return an acceleration which makes
            the equilibrium goes to 2 times its normal distance.
        state: one of 'decel' if we want equilibrium headway to increase, 'accel' if we want it to decrease.

    Returns:
        acceleration which shifts the equilibrium
    """
    if state == 'decel':
        temp = shift_parameters[0]**2
    else:
        temp = shift_parameters[1]**2

    return (temp - 1)/temp*p[3]*(1 - (v/p[0])**4)


def mobil(veh, lc_actions, lside, rside, newlfolhd, newlhd, newrfolhd, newrhd, newfolhd, timeind, dt,
          userelax_cur=True, userelax_new=False, use_coop=True, use_tact=True):
    """Minimizing total braking during lane change (MOBIL) lane changing decision model.

    parameters - [safety criteria (maximum deceleration allowed after LC, more negative = less strict),
                  incentive criteria (>0, larger = more strict. smaller = discretionary changes more likely),
                  politeness (taking other vehicles into account, 0 = ignore other vehicles, ~.2 = realistic),
                  bias on left side (can add a bias to make vehicles default to a certain side of the road),
                  bias on right side,
                  probability of checking LC while in discretionary state (not in original model. set to
                  1/dt to always check discretionary. probability of checking per 1 second)]
    naming convention - l/r = left/right respectively, current lane if no l/r
    new indicates that it would be the configuration after potential change

    Args:
        veh: ego vehicle
        lc_actions: dictionary where lane changing actions are stored, keys = vehicles, values = 'l' or 'r'
        lside: bool whether model evaluates a left lane change
        rside: bool whether model evaluates a right lane change
        newlfolhd: new left follower headway
        newlhd: new vehicle headway for left change
        newrfolhd: new right follower headway
        newrhd: new vehicle headway for right change
        newfolhd: new follower headway
        timeind: time index
        dt: time step
        userelax_cur: If True, we include relaxation in the evaluation of current acceleration, so that the
            cf model is not reevaluated if vehicle is in relaxation. True is recommended
        userelax_new: If True, we include relaxation in the evaluation of the new acceleration. False is
            recommended.
        use_coop: Controls whether cooperative model is applied (see coop_tact_model)
        use_tact: Controls whether tactical model is applied (see coop_tact_model)

    Returns:
        None. (modifies lc_actions in place)
    """
    p = veh.lc_parameters
    lincentive = rincentive = -math.inf

    # calculate cura, fola, newfola
    if not userelax_cur and veh.in_relax:
        cura = veh.get_cf(veh.hd, veh.speed, veh.lead, veh.lane, timeind, dt, False)
    else:
        cura = veh.acc  # more generally could use a method to return acceleration
    fola, newfola = mobil_helper(veh.fol, veh, veh.lead, newfolhd, timeind, dt, userelax_cur, userelax_new)

    # to compute left side: need to compute lfola, newlfola, newla
    if lside:
        lfol = veh.lfol
        llead = lfol.lead
        lfola, newlfola = mobil_helper(lfol, llead, veh, newlfolhd, timeind, dt, userelax_cur, userelax_new)

        userelax = userelax_new and veh.in_relax
        newla = veh.get_cf(newlhd, veh.speed, llead, veh.llane, timeind, dt, userelax)

        lincentive = newla - cura + p[2]*(newlfola - lfola + newfola - fola) + p[3]

    # same for right side
    if rside:
        rfol = veh.rfol
        rlead = rfol.lead
        rfola, newrfola = mobil_helper(rfol, rlead, veh, newrfolhd, timeind, dt, userelax_cur, userelax_new)

        userelax = userelax_new and veh.in_relax
        newra = veh.get_cf(newrhd, veh.speed, rlead, veh.rlane, timeind, dt, userelax)

        rincentive = newra - cura + p[2]*(newrfola - rfola + newfola - fola) + p[4]

    # determine which side we want to potentially intiate LC for
    if rincentive > lincentive:
        side = 'r'
        lctype = veh.r_lc
        incentive = rincentive

        newhd = newrhd
        newlcsidefolhd = newrfolhd
        selfsafe = newra
        lcsidefolsafe = newrfola
    else:
        side = 'l'
        lctype = veh.l_lc
        incentive = lincentive

        newhd = newlhd
        newlcsidefolhd = newlfolhd
        selfsafe = newla
        lcsidefolsafe = newlfola

    # determine if LC can be completed, and if not, determine if we want to enter cooperative or
    # tactical states
    if lctype == 'discretionary':
        if incentive > p[1]:
            if selfsafe > p[0] and lcsidefolsafe > p[0]:
                lc_actions[veh] = side
            else:
                coop_tact_model(veh, newhd, newlcsidefolhd, lcsidefolsafe, selfsafe, side, lctype,
                                use_coop=use_coop, use_tact=use_tact)
    else:  # mandatory state
        if selfsafe > p[0] and lcsidefolsafe > p[0]:
            lc_actions[veh] = side
        else:
            coop_tact_model(veh, newhd, newlcsidefolhd, lcsidefolsafe, selfsafe, side, lctype,
                            use_coop=use_coop, use_tact=use_tact)
    return


def mobil_helper(fol, curlead, newlead, newhd, timeind, dt, userelax_cur, userelax_new):
    """Helper function for MOBIL computes new accelerations for fol after a potential lane change.

    Args:
        fol: follower is assumed to follow curlead in the current configuration in new potential
            configuration it would follow newlead
        curlead: current leader for fol
        newlead: new leader for fol
        newhd: new headway for fol
        timeind: time index
        dt: time step
        userelax_cur: If True, apply relaxation in current acceleration. True recommended.
        userelax_new: If True, apply relaxation in new acceleration. False recommended.

    Returns:
        fola: The current acceleration for fol
        newfola: new acceleration for fol after the lane change
    """
    if fol.cf_parameters is None:
        fola = 0
        newfola = 0
    else:
        if not userelax_cur and fol.in_relax:
            fola = fol.get_cf(fol.hd, fol.speed, curlead, fol.lane, timeind, dt, False)
        else:
            fola = fol.acc

        userelax = userelax_new and fol.in_relax
        newfola = fol.get_cf(newhd, fol.speed, newlead, fol.lane, timeind, dt, userelax)

    return fola, newfola


def coop_tact_model(veh, newhd, newlcsidefolhd, lcsidefolsafe, selfsafe, side, lctype, use_coop=True,
                    use_tact=True):
    """Cooperative and tactical model for a lane changing decision model.

    Explanation of model -
    first we assume that we can give vehicles one of two commands - accelerate or decelerate.
    there are three possible options - cooperation and tactical, or only cooperation, or only tactical

    In the tactical model, first we check the safety conditions to see what is preventing us from
    changing (either lcside fol or lcside lead). if both are violating safety, and the lcside leader is
    faster than vehicle, then the vehicle gets deceleration to try to change behind them. If vehicle is
    faster than lcside leader, then the vehicle gets acceleration to try to overtake. if only one is
    violating safety, the vehicle moves in a way to prevent that violation. Meaning -
    if only the follower's safety is violated, the vehicle accelerates
    if the vehicle's own safety is violated; the vehicle decelerates
    the tactical model only modifies the acceleration of veh.

    in the cooperative model, we try to identify a cooperating vehicle. a cooperating vehicle always gets a
    deceleration added so that it will give extra space.
    If the cooperation is applied without tactical, then the cooperating vehicle must be the lcside follower,
    and the newlcsidefolhd must be > 0.
    if cooperation is applied with tactical, then in addition to the above, it's also possible the
    cooperating vehicle is the lcside follower's follower, where additionally the newlcsidefolhd is < 0.
    Note that in this second case, the lcside follower is directly to the side of vehicle, and that is why
    this case is distinct.
    In the first case where the cooperating vehicle is the lcside follower, the tactical model is applied
    as normal.
    In the second case, since the issue is the the lcside follower is directly blocking the vehicle,
    the vehicle accelerates if the lcside follower has a slower speed than vehicle, and decelerates otherwise.

    when a vehicle requests cooperation, it has to additionally fulfill a condition which simulates
    the choice of the cooperating vehicle. All vehicles have a innate probability (coop_parameters attribute)
    of cooperating, and for a discretionary LC, this innate probability controls whether or not the
    cooperation is accepted.
    For a mandatory LC, vehicle can add to this probability, which simulates forcing the cooperation.
    vehicles have a lc_urgency attribute which is updated upon initiating a mandatory change. lc_urgency is a
    tuple of two positions, at the first position, only the follower's innate cooperation probability
    is used. at the second position, the follower is always forced to cooperate, even if it has 0
    innate cooperation probability, and the additional cooperation probability is interpolated linearally
    between these two positions.

    Implementation details -
    when a vehicle has cooperation or tactical components applied, it has a lc_side attribute which is
    set to either 'l' or 'r' instead of None. This marks the vehicle as having the coop/tact added, and
    will make it so the vehicle attempts to complete the LC at every timestep even if its only a discretionary
    change. vehicles also have a coop_veh attribute which stores the cooperating vehicle. A cooperating
    vehicle does not have any attribute marking it as such.
    the lc_urgency attribute needs to be set whenever a mandatory route event begins. the lc_side, coop_veh,
    and lc_urgency attributes need to be reset to None whenever the change is completed.
    in the road event where a lane ends on the side == lc_side, then lc_side and coop_veh need to be reset.

    Args:
        veh: vehicle which wants to change lanes
        newhd: new headway of vehicle after potential lane change
        newlcsidefolhd: new lcside follower headway
        lcsidefolsafe: acceleration for lcside follower, used to check safety condition against
            veh.lc_parameters[0]
        selfsafe: acceleration for vehicle, used to check safety condition
        side: either 'l' or 'r', side vehicle wants to change
        lctype:either 'discretionary' or 'mandatory'
        use_coop: bool, whether to apply cooperation model. if use_coop and use_tact are both False,
            this function does nothing.
        use_tact: bool, whether to apply tactical model

    Returns:
        None (modifies veh, veh.coop_veh)
    """
    # clearly it would be possible to modify different things, such as how the acceleration modifications
    # are obtained, and changing the conditions for entering/exiting the cooperative/tactical conditions
    # in particular we might want to add extra conditions for entering cooperative state
    tact = False
    if side == 'l':
        lcsidefol = veh.lfol
    else:
        lcsidefol = veh.rfol

    if use_coop and use_tact:
        coop_veh = veh.coop_veh
        if coop_veh is not None:
            # 2 possible cases here
            if coop_veh is lcsidefol:
                coop_veh.acc += coop_veh.shift_eql(coop_veh.cf_parameters, coop_veh.speed,
                                                   coop_veh.shift_parameters, 'decel')
                tact = True
            else:  # check condition for coop_veh to be valid in case where coop_veh = lcsidefol.fol
                if coop_veh is lcsidefol.fol and newlcsidefolhd < 0:
                    coop_veh.acc += coop_veh.shift_eql(coop_veh.cf_parameters, coop_veh.speed,
                                                       coop_veh.shift_parameters, 'decel')
                    if lcsidefol.speed > veh.speed:
                        tactstate = 'decel'
                    else:
                        tactstate = 'accel'
                    veh.acc += veh.shift_eql(veh.cf_parameters, veh.speed, veh.shift_parameters, tactstate)
                else:  # cooperation is not valid -> reset
                    veh.lc_side = None
                    veh.coop_veh = None

        else:  # see if we can get vehicle to cooperate
            tact = True
            if newlcsidefolhd < 0:
                coop_veh = lcsidefol.fol
            else:
                coop_veh = lcsidefol

            if coop_veh.cf_parameters is not None:
                temp = coop_veh.coop_parameters
                if lctype == 'mandatory':
                    start, end = veh.lc_urgency[:]
                    temp += (veh.pos - start)/(end - start)

                if temp >= 1 or np.random.rand() < temp:
                    veh.coop_veh = coop_veh
                    veh.lc_side = side
                    # apply cooperation
                    coop_veh.acc += coop_veh.shift_eql(coop_veh.cf_parameters, coop_veh.speed,
                                                       coop_veh.shift_parameters, 'decel')
                    # apply tactical
                    if coop_veh is not lcsidefol:  # in this case we need to manually apply
                        # otherwise we go to normal tactical model (tact = True)
                        tact = False
                        if lcsidefol.speed > veh.speed:
                            tactstate = 'decel'
                        else:
                            tactstate = 'accel'
                        veh.acc += veh.shift_eql(veh.cf_parameters, veh.speed, veh.shift_parameters,
                                                 tactstate)

    elif use_coop and not use_tact:
        coop_veh = veh.coop_veh
        if coop_veh is not None:
            if coop_veh is lcsidefol and newhd > 0:  # conditions for cooperation when there is no tactical
                coop_veh.acc += coop_veh.shift_eql(coop_veh.cf_parameters, coop_veh.speed,
                                                   coop_veh.shift_parameters, 'decel')
            else:  # cooperating vehicle not valid -> reset
                veh.lc_side = None
                veh.coop_veh = None
        else:  # see if we can get vehicle to cooperate
            if newhd > 0 and lcsidefol.cf_parameters is not None:  # vehicle is valid
                # check cooperation condition
                temp = lcsidefol.coop_parameters
                if lctype == 'mandatory':
                    start, end = veh.lc_urgency[:]
                    temp += (veh.pos - start)/(end - start)

                if temp >= 1 or np.random.rand() < temp:  # cooperation agreed/forced -> add cooperation
                    veh.coop_veh = lcsidefol
                    veh.lc_side = side
                    # apply cooperation
                    lcsidefol.acc += lcsidefol.shift_eql(lcsidefol.cf_parameters, lcsidefol.speed,
                                                         lcsidefol.shift_parameters, 'decel')

    elif tact or (not use_coop and use_tact):
        # mark side if not already
        if veh.lc_side is None:
            veh.lc_side = side
        # find vehicle which is preventing change - if both lcsidefol and lcsidelead are prventing,
        # default to looking at the lcsidelead
        safe = veh.lc_parameters[0]  # this part is more specific to MOBIL
        if lcsidefolsafe < safe:
            if selfsafe < safe:  # both unsafe
                if lcsidefol.lead.speed > veh.speed:
                    tactstate = 'decel'
                else:
                    tactstate = 'accel'
            else:  # only follower unsafe
                tactstate = 'accel'
        else:  # only leader unsafe
            tactstate = 'decel'
#        if tactveh == None: #may need this in very weird edge case where you are at boundary
#            return
        veh.acc += veh.shift_eql(veh.cf_parameters, veh.speed, veh.shift_parameters, tactstate)


def IDM_parameters(*args):
    """Suggested parameters for the IDM/MOBIL."""
    cf_parameters = [27, 1.2, 2, 1.1, 1.5]  # note speed is supposed to be in m/s
    cf_parameters[0] += np.random.rand()*6
    # cf_parameters[0] += np.random.rand()*25-15 #give vehicles very different speeds for testing purposes
    lc_parameters = [-1.5, .2, .2, 0, .1, .5]

    return cf_parameters, lc_parameters
