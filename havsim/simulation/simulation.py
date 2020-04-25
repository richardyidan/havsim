
"""
@author: rlk268@cornell.edu
houses the main code for running simulations 
    
    
"""

import numpy as np 
import math 
import scipy.optimize as sc 
    
def update_net(vehicles, lc_actions, inflow_lanes, merge_lanes, vehid, timeind, dt): 
    #update followers/leaders for all lane changes 
    for veh in lc_actions.keys(): 
        update_change(veh, timeind) #this cannot be done in parralel
    
        #apply relaxation 
        new_relaxation(veh, timeind, dt)
        
        #update a vehicle's road events here // TO DO
        #add cooperative/tactical to update_change // TO DO 
        
        #update a vehicle's route events, adding new events if necessary 
        set_new_route(veh)
    
    #update all states, memory and headway 
    for veh in vehicles: 
        veh.update(timeind, dt)
    for veh in vehicles:
        if veh.lead is not None: 
            veh.hd = veh.lane.get_headway(veh, veh.lead)
        else: #don't actually need this but for robustness
            veh.hd = None
        
    #update left and right followers
    for veh in vehicles:
        update_lrfol(veh)
    
    #update merge_anchors // TO DO update this 
    for lane in merge_lanes:
        for i in range(len(lane.merge_anchors)):
            veh, pos = lane.merge_anchors[i]
            if pos == None: #some merge anchors may not need to be updated 
                continue
            if veh.cf_parameters == None:  
                lead = veh.lead
                if lead is not None and ((lead.lane is lane and lead.pos < pos) or lane.roadlen[lead.road]+lead.pos - pos < 0):
                    lane.merge_anchors[i][0] = lead
                    
            elif veh in lc_actions: 
                if lc_actions[veh] == 'l': 
                    lane.merge_anchors[i][0] = veh.rfol
                else: 
                    lane.merge_anchors[i][0] = veh.lfol
            
            else:
                if veh.pos > pos: 
                    lane.merge_anchors[i][0] = veh.fol
                    
    #inflow goes here
    for lane in inflow_lanes: 
        vehid = lane.increment_inflow(vehicles, vehid, timeind, dt)
                    
    #update roads and routes last
    for veh in vehicles: 
        #check vehicle's lane events and act if necessary 
        #check vehicle's route events and act if neccessary
        update_route(veh)
        pass
                    
    return 

def update_route(veh):
    #will check a vehicle's current route events and see if we need to do anything 
    #returns True if we make a change, False otherwise
    #expect a list of dictionarys, each dict is an event with keys of pos, event, side 
    #giving the position when the event occurs, event type, and side of the event 
    curevent = veh.route_events[0]
    if veh.pos > curevent['pos']:
        if curevent['event'] == 'end discretionary': 
            setattr(veh, curevent['side'],None)
        elif curevent['event'] == 'mandatory': 
            setattr(veh, curevent['side'],'mandatory')
        veh.route_events.pop(0)
        return True
    return False

def update_lane_events(veh, timeind): 
    curevent = veh.lane_events[0]
    if veh.pos > curevent['pos']:
        if curevent['event'] == 'new road': 
            #update lane/road/position
            newlane = veh.lane.connect_to
            newroad = newlane.road
            veh.pos += -veh.lane.roadlen[newroad]
            veh.lane = newlane
            veh.road = newroad
            veh.lanemem.append((newlane, timeind+1))
            
            #need to update vehicle orders and lane change state
            if curevent['left'] == 'remove': 
                veh.lfol.rlead.remove(veh)
                veh.lfol = None
                veh.l = None
            elif curevent['left'] == 'add':
                newllane = newlane.get_connect_left(curevent['pos'])
                merge_anchor = newllane.merge_anchors[curevent['left merge']]
                unused, newfol = newlane.leadfol_find(veh,merge_anchor,'l')
                
                veh.lfol = newfol
                newfol.rlead.add(veh)
                veh.l = 'discretionary'
            
            #same thing for right
            if curevent['right'] == 'remove: 
                veh.rfol.llead.remove(veh)
                veh.rfol = None
                veh.r = None
            elif curevent['right'] == 'add': 
                newrlane  = newlane.get_connect_right(curevent['pos'])
                merge_anchor = newrlane.merge_anchors[curevent['right merge']]
                unused, newfol = newlane.leadfol_find(veh, merge_anchor, 'r')
                
                veh.rfol = newfol
                newfol.llead.add(veh)
                veh.r = 'discretionary'
            
            #update vehicle's routes and lane events 
        
        elif curevent['event'] == 'add':
            pass
        
        elif curevent['event'] == 'exit':
            pass
                
def set_lane_events(veh):
    pass

def make_cur_route(p, lane, nextroadname): 
    #generates route events with parameters p in lane lane 
    #p - parameters - currently len 2 list with constant, p[0] is a constant which is a like a safety buffer, and p[1]
    #controls the distance you need for a change (also a constant)
    #lane- lane object that the route events start on 
    #nextroad - once you leave lane.road, you want to end up on nextroad
    
    #output - dictionary where keys are lanes, values are the route events a vehicle with 
    #parameters p needs to follow on that lane. 
    
    #explanation of current model - 
    #if  you need to be in lane '2' by position 'x' and start in lane '1', 
    #then starting at x - p[0] - 2*p[1] you will end discretionary changing into lane '0'
    #at x - p[0] - p[1] you wil begin mandatory changing into lane '2'
    #at x - p[0] your mandatory change will have urgency of 100% which will always force cooperation of your l/rfol 
    #for merging onto/off an on-ramp which begins at 'x' and ends at 'y', you will start mandatory at 'x' always, 
    #reaching 100% cooperation by 'y' - p[0]
    
    #we only get the route for the current road - no look ahead to take into account future roads. // TO DO low priority 
    #nothing to handle cases where LC cannot be completed successfully // TO DO
    #would need to know latest point when change can take place ('pos' for 'continue' type), 
    #need to add an attribute for 'merge' type giving this 
    #in lane changing model, it would need to check if we are getting too close and act accordingly (e.g. slow down) if so 
    #in this function, would need to add events if you miss the change, and in that case you would need to be given a new route 
    
    curroad = lane.road
    curlaneind = lane.laneind
    #position, str, tuple of 2 ints or single int, str, dict for the next road
    pos, change_type, laneind, side, nextroad  = curroad['connect to'][nextroadname][:]
    
    cur_route = {}
    
    if change_type == 'continue': #-> vehicle needs to reach end of lane to transition to next road 
        #initialize for lanes which vehicle needs to continue on 
        for i in range(laneind[0], laneind[1]+1):
            cur_route[curroad[i]] = []
            
        templane = curroad[laneind[0]]
        cur_route[templane].append({'pos': templane.end - p[0] - p[1], 'event': 'end discretionary', 'side': 'l'})
        
        templane = curroad[laneind[1]]
        cur_route[templane].append({'pos': templane.end - p[0] - p[1], 'event': 'end discretionary', 'side': 'r'})
        
        if curlaneind >= laneind[0] and curlaneind <= laneind[1]: #if on correct lane(s) already, do no more work 
            return cur_route
        
        elif curlaneind < laneind[0]: #need to change right possibly multiple times
            uselaneind = laneind[0]
        else: 
            uselaneind = laneind[1]
            
        cur_route = make_route_helper(p, cur_route, curroad, curlaneind, uselaneind, curroad[uselaneind].end, curroad[uselaneind].start)
                
            
    elif change_type =='merge': #logic is similar and also uses make_route_helper
        if side == 'l': 
            opside = 'r'
        else: 
            opside= 'l'
            
        templane = curroad[laneind]
        cur_route[templane].append({'pos': pos -p[0] - p[1], 'event':'end discretionary', 'side':opside})
        cur_route[templane].append({'pos': pos, 'event':'mandatory', 'side':side})
        
        cur_route = make_route_helper(p, cur_route, curroad, curlaneind, laneind, pos, templane.start)
    
    return cur_route
    
def make_route_helper(p, cur_route, curroad, curlaneind, laneind, curpos, mincurpos):
    #p - parameters for route 
    #cur_route - dictionary to add entries to 
    #curroad - current road 
    #curlaneind - current index of lane you start in 
    #laneind, curpos - index of lane you want to be in by position curpos 
    #mincurpos - for edge case to prevent you from changing onto lane before it starts 
    
    #starting on curroad in lane with index curlaneind, and wanting to be in laneind by curpos position, 
    #generates routes cur all roads in [curlaneind, laneind)
    #assumes you already have the route for laneind
    if curlaneind < laneind:
        curind = laneind - 1
        templane = curroad[curind]
        cur_route[templane] = []
        while not (curind < curlaneind):
            #determine curpos = where the mandatory change starts 
            if templane.end < curpos: 
                curpos = templane.end
            curpos += -p[0] - p[1]
            curpos = max(mincurpos, curpos)
            enddiscpos = curpos - p[0] - p[1]
            
            #append the two events - end discretionary and being mandatory 
            cur_route[templane].append({'pos': enddiscpos, 'event': 'end discretionary', 'side': 'l'})
            cur_route[templane].append({'pos': curpos, 'event': 'mandatory', 'side': 'r'})
            
            #update iteration 
            mincurpos = templane.start
            curind += -1 
            templane = curroad[curind]
            
        
    elif curlaneind > laneind: 
        curind = laneind +1
        templane = curroad[curind]
        cur_route[templane] = []
        while not (curind > curlaneind):
            #determine curpos = where the mandatory change starts 
            if templane.end < curpos: 
                curpos = templane.end
            curpos += -p[0] - p[1]
            curpos = max(mincurpos, curpos)
            enddiscpos = curpos - p[0] - p[1]
            
            #append the two events - end discretionary and being mandatory 
            cur_route[templane].append({'pos': enddiscpos, 'event': 'end discretionary', 'side': 'r'})
            cur_route[templane].append({'pos': curpos, 'event': 'mandatory', 'side': 'l'})
            
            #update iteration 
            mincurpos = templane.start
            curind += -1 
            templane = curroad[curind]

    return cur_route
        
def set_new_route(veh):
    
    
    #get new route events if they are stored in memory already 
    newlane = veh.lane 
    if newlane in veh.cur_route: 
        veh.route_events = veh.cur_route[newlane].copy() #use shallow copy - copy references only
    #otherwise we will make it 
    else:
        p = veh.route_parameters
        prevlane = veh.lanemem[-2][0]
        if prevlane.road is newlane.road: #on same road - we can just use helper function to update cur_route
            curpos = veh.cur_route[prevlane][0]['pos'] + p[0] + p[1]
            make_route_helper(veh.route_parameters, veh.cur_route, veh.road, newlane.laneind, prevlane.laneind, curpos,prevlane.start)
        else: #on new road - we need to generate new cur_route and update the vehicle's route
            veh.cur_route = make_cur_route(p, newlane, veh.route.pop(0))
        
        veh.route_events = veh.cur_route[newlane].copy()
    
    curbool = True
    while curbool: 
        curbool = update_route(veh)
        
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
    elif veh.lane.get_dist(veh,rfol) > 0: 
        veh.rfol = rfol.fol
        veh.rfol.llead.add(veh)
        rfol.llead.remove(veh)
        
        rfol.lfol.rlead.remove(rfol)
        rfol.lfol = veh
        veh.rlead.add(rfol)
        
        
        
    

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

#in current logic, main cost per timestep is just one distance compute in update_lrfol
#whenever there is a lane change, there are a fair number of extra updates we have to do to keep all 
#of the rlead/llead updated. Also, whenever an lfol/rfol changes, there are two rlead/lead attributes 
#that need to be changed as well. 
#Thus this strategy is very efficient assuming we want to keep lfol/rfol updated (call lc every timestep), lane changes aren't 
#super common, and all vehicles travel at around the same speed. (which are all reasonable assumptions)
    
#naive way would be like having to do something like keep a sorted list, every time we want lfol/rfol
#we have to do log(n) dist computes, where n is the number of vehicles in the current lane. 
#whenever a vehicle changes lanes, you need to remove from the current list and add to the new, 
#so it is log(n) dist computations + 2n for searching/updating the 2 lists. Thus the current implementation is definitely much better than the naive way. 
    
#Another option you could do is to only store lfol/rfol, to keep it updated you would have to 
#do 2 dist calculations per side per timestep (do a call of leadfol find where we already have either a follower or leader as guess). 
#When there is a lane change store a dict which has the lane changing vehicle as a key, and store as the value the new guess to use. 
#in lfol/rfol update, you know there was a lane change if your fol is in the wrong lane Then can get a new guess for the fol from the dict. 
#This strategy would have higher costs per timestep to keep lfol/rfol updated, but would be simpler to update when there is a lane change. 
#Thus it might be more efficient if the timesteps are long relative to the number of lane changes. 
#Overall I doubt there would be much practical difference between this option and the first option unless the timesteps are very long (~10-15 sec) and changes very often (every ~30 seconds)
def update_change(veh, timeind): 
    #logic for updating - logic is complicated because we avoid doing any sorts - faster this way 
    
    #no check for vehicles moving into same gap // TO DO low priority 
    #no cooperative tactical components // TO DO
    
    #initialization and update l/r attributes
    lane = veh.lane
    if veh.lc == 'l':
        lcsidefol, opsidefol, lcsidelead, opsidelead = 'lfol', 'rfol', 'llead', 'rlead'
        lcsidelane = lane.get_connect_left(veh.pos)
        newroad = lcsidelane.road
        newlcsidelane = lcsidelane.get_connect_left(veh.pos)
        veh.r = 'discretionary'
        if newroad is newlcsidelane.road: 
            veh.l = 'discretionary'
        
    else: 
        lcsidefol, opsidefol, lcsidelead, opsidelead = 'rfol', 'lfol', 'rlead', 'llead'
        lcsidelane = lane.get_connect_right(veh.pos)
        newroad = lcsidelane.road
        newlcsidelane = lcsidelane.get_connect_right(veh.pos)
        veh.l = 'discretionary'
        if newroad is newlcsidelane.road: 
            veh.r = 'discretionary'
        
    #update all leader/follower relationships###
    
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

    if lclead is not None: 
        lclead.fol = veh
    #update for new left/right leaders - opside first 
    newleads = set()
    oldleads = getattr(lcfol, opsidelead)
    for j in oldleads.copy(): 
        curdist = lane.get_dist(veh,j)
        if curdist > 0: 
            setattr(j, lcsidefol, veh)
            newleads.add(j)
            oldleads.remove(j)
    setattr(veh, opsidelead, newleads)
    #lcside 
    newleads = set()
    oldleads = getattr(lcfol, lcsidelead)
    mindist = math.inf
    minveh = None
    for j in oldleads.copy():
        curdist = lane.get_dist(veh, j)
        if curdist > 0: 
            setattr(j, opsidefol, veh)
            newleads.add(j)
            oldleads.remove(j)
            if curdist < mindist: 
                mindist = curdist 
                minveh = j #minveh is the leader of new lc side follower 
    setattr(veh, lcsidelead, newleads)
    
    #update lane
    veh.lane = lcsidelane
    veh.lanemem.append((lcsidelane, timeind+1))
    #road/position
    if newroad is not veh.road: 
        veh.pos += - lane.roadlen[newroad]
        veh.road = newroad
    
    #update new lcside leaders/follower
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
            

            
    return 
        
def get_guess(lcfol, lclead, veh, lcsidefol, newlcsidelane):
    #need to find new lcside follower for veh
    #not sure if this is going to behave properly 
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
        
def eql_wrapper(eqlfun, eql_type = 'v', bounds = (1e-10, 120), tol = .1, **kwargs):
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

#get_flow is currently not needed anywhere in the code, but could be useful to have 
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
                
#solving for the headway/speed given flow must be done numerically, 
    #and thus has some extra computational costs associated with it. 
    #if eql_type = 'both', it should be possible to solve for the inverse flow function 
    #analytically and thus one should just define the inv\_flow method using that 
def invFD_wrapper(eqlfun, eql_type = 'v', bounds = (1e-10, 120), tol = .1, ftol = .01, invflowfun = None):
    #same call signature as eql_wrapper, tol is for headway/speed tolerance, ftol is for flow tolerance 
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
            
            res = sc.minimize_scalar(maxfun, bracket = bounds, tol = ftol)
            if res['converged']:
                if congested: 
                    invbounds = (bounds[0], res['x'])
                else:
                    invbounds = (res['x'], bounds[1])
            else:
                raise RuntimeError('could not find maximum flow')
                
                
            if eql_type == 'v':
                def invfun(y): 
                    return x - y/(eqlfun(self.cf_parameters, y) + leadlen)
            elif eql_type == 's':
                def invfun(y):
                    return x - eqlfun(self.cf_parameters, y)/(y+leadlen)
            
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
        def inv_flow(self, x, leadlen = None, output_type = 'v', congested = True):
            return invflowfun(x, leadlen, output_type, congested)
    
    return inv_flow
        
        
    


def CF_wrapper(cfmodel, acc_bounds = [-7,3]): 
    #acc_bounds controls [lower, upper] bounds on acceleration 
    #assumes a second order model which has inputs of (p, state), where state
    #is a list of all values needed, p is a list of parameters, and
    #output is a float giving the acceleration 
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
    
    #// TO DO get rid of dist to end, just use None
    #don't think anchor vehicles should give a 0 headway either, should just be None? 
    
    def call_lc(self, chk_lc, timeind, dt):
        lfol, rfol, lane = self.lane = self.lfol, self.rfol, self.lane
        if lfol== '' and rfol == '':
            return 
        
        if chk_lc == 0:
            return
        elif chk_lc == 1:
            pass
        elif np.random.rand() > chk_lc: 
            return 
        
        if lfol != '': 
            llead, newlfolhd, newlhd = call_lc_helper(lfol, self, lane.get_connect_left(self.pos))
        else:
            llead = newlfolhd = newlhd = None
        
        if rfol != '': 
            rlead, newrfolhd, newrhd = call_lc_helper(rfol, self, lane.get_connect_right(self.pos))
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

#2 options for implementing your own LC model and CF model - 
    #option 1 - we have decorators for book keeping work for the LC/CF functions, user specifies the model 
    #and what the vehicle object calls is the decorated model which will handle formatting/bookkeeping issues
    #the default class will use this design so users can write their own model in a standard format 
    #and directly feed that in (easier but more restrictive)
    
    #option 2 - you can inherit this class and write your own functions (more work but less restrictive)
    #(also possible that you may use the decorators for your own custom methods)
class vehicle: 
    
    def __init__(self, vehid,lane, p, lcp, length = 2, relaxp = None,
                 cfmodel = None, free_cf = None, lcmodel = None, eqlfun = None, check_lc = .25,
                 eql_kwargs = {}): 
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
            self.get_eql = eql_wrapper(eqlfun, **eql_kwargs)
            self.get_flow = FD_wrapper(eqlfun)
            self.inv_flow = invFD_wrapper(eqlfun, **eql_kwargs)
            
        if lcmodel is not None: 
            self.call_lc = LC_wrapper(lcmodel)
            
    def __hash__(self):
        return hash(self.vehid)
        
    def __eq__(self, other):
        return self.vehid == other.vehid
    
    def __ne__(self, other):
        return not(self is other)
            
    def update(self, timeind, dt): 
        #update state
        temp = self.action*dt
        self.pos += self.speed*dt + .5*temp*dt
        self.speed += temp
        
        #update memory and relax
        self.posmem.append(self.pos)
        self.speedmem.append(self.speed)
        if self.in_relax:
            if timeind == self.relax_start + len(self.relax) - 1:
                self.in_relax = False
                self.relaxmem.append((self.relax_start, timeind, self.relax))
                
    
            

def downstream_wrapper(speed_fun = None, method = 'speed', congested = True, 
                       mergeside = 'l', merge_anchor_ind = None, anchor = None, shift = 1):
    #downstream function -> method of lane, takes in (veh, timeind, dt)
    #and returns action (acceleration) for the vehicle 
    
    if method == 'speed': #specify a function speedfun which takes in time and returns the speed
        @staticmethod
        def call_downstream(veh, timeind, dt):
            speed = speed_fun(timeind)
            return (speed - veh.speed)/dt
        return call_downstream
    
    elif method == 'free': #use free flow method of the vehicle 
        @staticmethod
        def free_downstream(veh, *args):
            return veh.free_cf(veh.cf_parameters, veh.speed)
        return free_downstream
    
    elif method == 'flow': #specify a function which gives the flow, we invert the flow to obtain speed
        @staticmethod
        def call_downstream(veh, timeind, dt):
            flow = speed_fun(timeind)
            speed = veh.inv_flow(flow, output_type = 'v', congested = congested)
            return (speed - veh.speed)/dt
        return call_downstream
    
     #this is meant to give a longitudinal update in congested conditions 
     #when on a bottleneck (on ramp or lane ending) and you have no leader 
     #not because you are leaving the network but because the lane is ending and you need to move over
    elif method == 'merge':
        #first try to get a vehicle and use its shifted speed. By default use the l/rfol (controlled by mergeside)
        #can also try using the merge anchor (if merge_anchor_ind is not None) or another anchor's lead (if anchor is not None)
        #it has to be a vehicle (not an anchor vehicle) as we want its speedmem
        #if we fail to find such a vehicle and speed_fun is not None, we will use that; 
        #otherwise we will use the vehicle's free_cf method
        
        #the vehicle won't slow down if approaching the end of the lane // TO DO
        #would be simple to add this modification - at end of function do a check if youre getting close, 
        #then can just use the car following model with the headway to the end 
        if mergeside == 'l': 
            folside = 'lfol'
        elif mergeside == 'r':
            folside = 'rfol'
        def call_downstream(self, veh, timeind, dt): 
            fol = getattr(veh, folside) #first check if we can use your current change side follower
            if fol.cf_parameters == None: 
                fol = fol.lead
                if fol == None and merge_anchor_ind != None:
                    fol = self.merge_anchor[merge_anchor_ind][0]
                    if fol.cf_parameters == None: 
                        fol = fol.lead
                        if fol == None and anchor != None: 
                            fol = anchor.lead
            if fol != None: 
                speed = shift_speed(fol.speedmem, shift, dt)
            elif speed_fun != None:
                speed = speed_fun(timeind)
            else: 
                return veh.free_cf(veh.cf_parameters, veh.speed)
            return (speed - veh.speed)/dt
                
    
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
    spd = shift_speed(lead.speedmem, shift, dt)
        
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

def shift_speed(speedseries, shift, dt):
    #speedseries is timeseries with a constant discretization of dt
    #we want the measurement from shift time ago
    #outputs speed
    ind = shift // dt 
    if ind+1 > len(speedseries):
        return speedseries[0]
    remainder = shift - ind*dt
    spd = (speedseries[-ind-1]*(dt - remainder) + speedseries[-ind]*remainder)/dt #weighted average
    return spd

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
            
            #initaialize route // TO DO #also some of the initialization above will need to be changed as well
            
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
    def __init__(self, laneid, start, end, road, laneindex, connect_left = [(0, None)], connect_right = [(0, None)],
                 downstream = {}, increment_inflow = {}, get_inflow = {}, new_vehicle = None):
        
        self.laneid = laneid
        self.laneindex = laneindex
        self.road = road
        #starting position/end (float)
        self.start = start
        self.end = end
        #connect_left/right has format of list of (pos (float), lane (object)) tuples where lane is the connection starting at pos 
        self.connect_left = connect_left
        self.connect_right = connect_right
        self.connect_to = None
        self.connect_from = None
        
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
        #assumes veh.road = self.road
        hd = lead.pos - veh.pos - lead.length
        if self.road != lead.road: 
            hd += self.roadlen[lead.road]
        return hd 
    
    def get_dist(self, veh, lead): 
        #distance from front of vehicle to front of lead
        #assumes veh.lane.road = self.road
        dist = lead.pos-veh.pos
        if self.road != lead.road: 
            dist += self.roadlen[lead.road]
        return dist
            
    def dist_to_end(self, veh):
        #distance from front of vehicle to end of network 
        return self.enddist - veh.pos
    
    def leadfol_find(self, veh, guess, side):
        #given guess vehicle which is 'close' to veh
        #returns the leader, follower in that order in the same track of lanes as guess 
        #side is the side of veh we are looking at - e.g. side = 'r' means we are looking to the right of veh
        
        #used to initialize the new lc side follower/leader when new lanes become available
        #because this is only used when a new lane becomes available, there will always be a follower returned
        #it is possible that the leader is None, or that there is a leader but it can't have veh as a follower. 
        
#        if guess == None: #I don't remember what case this is for or if its even still necessary or useful 
#            return None, None
#        else: 
        
        if side == 'r':
            checkfol = 'lfol'
        else:
            checkfol = 'rfol'
        get_dist = self.get_dist
        hd = get_dist(veh,guess)
        if hd < 0: 
            nextguess = guess.lead 
            if nextguess == None:  #None -> reached end of network
                return nextguess, guess
            nexthd = get_dist(veh, nextguess)
            while nexthd < 0: 
                guess = nextguess 
                nextguess = guess.lead
                if nextguess == None:
                    return nextguess, guess
                nexthd = get_dist(veh, nextguess)
                
            if getattr(nextguess,checkfol) == None: 
                nextguess = None
            return nextguess, guess
        else: 
            nextguess = guess.fol
            if nextguess == None:
                return guess, nextguess
            nexthd = get_dist(veh, nextguess)
            while nexthd > 0:
                guess = nextguess
                nextguess = guess.fol
                if nextguess.cf_parameters == None: #reached anchor -> beginning of network
                    return guess, nextguess
                nexthd = get_dist(veh, nextguess)
            if getattr(guess,checkfol) == None: 
                guess = None
            return guess, nextguess
    
        
    def get_connect_left(self, pos):
        #given position, returns the connection to left 
        #output is either lane object or None
        return connect_helper(self.connect_left, pos)

    def get_connect_right(self, pos):
        return connect_helper(self.connect_right,pos)
    
    
    def __hash__(self):
        return hash((self.road['name'], self.laneindex))
    
    def __eq__(self, other):
        return self.road['name'] == other.road['name'] and self.laneindex == other.laneindex
    
    def __ne__(self, other):
        return not(self is other)
    
    
def connect_helper(connect, pos):
    out = connect[-1][1] #default to last lane for edge case or case when there is only one possible connection 
    for i in range(len(connect)-1):
        if pos < connect[i+1][0]:
            out = connect[i+1][1]
            break
    return out 
