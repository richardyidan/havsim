
"""
where all the plotting functions go

@author: rlk268@cornell.edu
"""

import numpy as np
import copy 
import math

import matplotlib.pyplot as plt 
from matplotlib.collections import LineCollection
import matplotlib.animation as animation
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import pywt #package for wavelet transforms in python 
from matplotlib import cm
from statistics import harmonic_mean #harmonic mean function
from matplotlib.widgets import RectangleSelector
from matplotlib.widgets import PolygonSelector
import matplotlib.transforms as mtransforms

import palettable

from .calibration import helper
from .calibration.opt import r_constant
from .calibration.helper import sequential, indtotimes

def optimalvelocity(s,p):
    V = p[0]*(np.tanh(p[1]*(s)-p[2]-p[4]) - np.tanh(-p[2]))
    return V

def optimalvelocityplot(p):
    s = np.linspace(0,200,2000)
    v = np.zeros(len(s))
    
    for i in range(len(v)): 
        v[i] = optimalvelocity(s[i],p)
    
    plt.figure()
    plt.plot(s,v)
    plt.xlabel('headway (ft)')
    plt.ylabel('speed (ft/s)')
    #plt.savefig('ovm3.png')
    return 

def checksort(vehlist, meas,lane):
    #very simple function plots trajectory in a line, allows you to debug the platoon order 
    fig = plt.figure()
    for i in range(len(vehlist)):
        cur = meas[vehlist[i]]
        cur = cur[cur[:,7]==lane]
        plt.plot(cur[:,1],cur[:,2],picker = 5)
        
    def on_pick(event):
        ax = event.artist.axes
        curind = ax.lines.index(event.artist) #artist index
        plt.title('selected vehicle '+str(vehlist[curind])+' which has order '+str(curind))
        plt.draw()
        
    fig.canvas.callbacks.connect('pick_event',on_pick)
    pass

def optplot(out, meas,sim,followerchain, platoonlist, model, modeladj, modeladjsys, makeleadfolfun, platoonobjfn,args,
            speed = False, newfig = True,clr = ['C0', 'C1'],fulltraj=True, lane = None): #plot platoon in space-time
    #CURRENT DOCUMENTATION 
    #out - results from optimization algorithm
    #meas - measurements in np array, rows are observations
    #sim - simulation in same format as meas. can pass in None and we will get the simulation from out. otherwise we will assume sim is loaded in correctly 
    #
    #in different colors. 
    #followerchain (platooninfo) - dictionary containing information on each vehicle ID
    #platoonlist - input the list of platoons used in the optimization 
    #model
    #modeladj
    #modeladjsys
    #makeleadfolfun
    #platoonobjfn
    #args
    
    
    
    #speed = False - if True will plot the speed instead of position
    #newfig = True - if True will create a new figure, otherwise it will use the current figure 
    #clr = 'C0', assuming Colors = False, clr will control what colors will be used. Default is ['C0','C1'] which are the default matplotlib colors
    #fulltraj = True - assuming Colors = False, if True, will plot the entire trajectory, if False, will plot only the simulation 
    #lane = None - If passed in as a laneID, the parts of trajectories not in the lane ID given will be made opaque 
    
    #this is meant to examine the output from using an optimization algorithm - visualize results and see which vehicles 
    
    
    
    ind = 2
    
    if sim is None: 
        sim = copy.deepcopy(meas)
        sim = helper.obj_helper(out[0],model,modeladj,modeladjsys,meas,sim,followerchain,platoonlist,makeleadfolfun,platoonobjfn,args)
    
    artist2veh = [] #artist2veh takes artist index and converts it to vehicle ID index (for followerlist)
    veh2platoon = [] #takes veh ID index and converts to platoon index (for platoonlist)
    platoonrmse = out[2] #convert platoon index to platoon rmse
    vehrmse = [] #converts veh ID index to rmse for vehicle 

    #vehrmse
    for i in platoonlist: 
        obj = helper.SEobj_pervehicle(meas,sim,followerchain,i) #list of (individual) objective functions. This is needed for next step
        for j in range(len(i)-1): #convert each individual objective to individual rmse
            temp = helper.convert_to_rmse(obj[j],followerchain,[[],i[j+1]])
            vehrmse.append(temp)
    
#    indcounter = np.asarray([],dtype = int64) #keeps track of which artists correspond to which vehicle
    if speed: 
        ind = 3
    if platoonlist != []:
        followerchain = helper.platoononly(followerchain, platoonlist)
    followerlist = list(followerchain.keys()) #list of vehicle ID
    
    for count,i in enumerate(platoonlist): 
        for j in i[1:]:
            veh2platoon.append(count)
        
    
    if newfig:
        fig = plt.figure()
    
    counter = 0 
    for i in followerlist: #iterate over each vehicle
        veh = meas[i]   
        t_nstar, t_n, T_nm1, T_n = followerchain[i][0:4] #note followerchain and platooninfo have same 
        
        if fulltraj: #entire trajectory including pre simulation and shifted end
            start = 0
            end = T_n-t_nstar
        else: #only show trajectory which is simulated 
            start = t_n-t_nstar
            end = T_nm1-t_nstar
        veh = veh[start:end,:]
        x = veh[:,1]
        y = veh[:,ind]
        
        if lane is not None: 
        
            #LCind is a list of indices where the lane the vehicle is in changes. Note that it includes the first and last index. 
            LCind = np.diff(veh[:,7])
            LCind = np.nonzero(LCind)[0]+1
            LCind = list(LCind)
            LCind.insert(0,0); LCind.append(len(veh[:,7])+1)
            
            for j in range(len(LCind)-1):
                kwargs = {}
                if meas[i][LCind[j],7] !=lane:
                    kwargs = {'linestyle':'--','alpha':.4} #dashed line .4 opacity (60% see through)
                plt.plot(x[LCind[j]:LCind[j+1]],y[LCind[j]:LCind[j+1]],clr[0],picker=5, **kwargs)
                artist2veh.append(counter) #artist2veh takes artist index and converts it to vehicle ID index (for followerlist)

                
            
        else: 
            plt.plot(x,y,clr[0])
            artist2veh.append(counter)
        counter += 1
            
    if sim is not None: 
        counter = 0
        for i in followerlist: #iterate over each vehicle
            veh = sim[i]   
            t_nstar, t_n, T_nm1, T_n = followerchain[i][0:4] #note followerchain and platooninfo have same 
            
            if fulltraj: #entire trajectory including pre simulation and shifted end
                start = 0
                end = T_n-t_nstar
            else: #only show trajectory which is simulated 
                start = t_n-t_nstar
                end = T_nm1-t_nstar
            veh = veh[start:end,:]
            x = veh[:,1]
            y = veh[:,ind]
            
            if lane is not None: 
            
                #LCind is a list of indices where the lane the vehicle is in changes. Note that it includes the first and last index. 
                LCind = np.diff(veh[:,7])
                LCind = np.nonzero(LCind)[0]+1
                LCind = list(LCind)
                LCind.insert(0,0); LCind.append(len(veh[:,7]))
                
                for j in range(len(LCind)-1):
                    kwargs = {}
                    if sim[i][LCind[j],7] !=lane:
                        kwargs = {'linestyle':'--','alpha':.4} #dashed line .4 opacity (60% see through)
                    plt.plot(x[LCind[j]:LCind[j+1]],y[LCind[j]:LCind[j+1]],clr[1],picker=5, **kwargs)
                    artist2veh.append(counter)
            else: 
                plt.plot(x,y,clr[1])
                artist2veh.append(counter)
            counter += 1
            
    #        plt.plot(x2,y2)
    #        plt.plot(x3,y3)
    
    
    
        
    plt.xlabel('time (frameID )')
    plt.ylabel('space (ft)')
    if speed: 
        plt.ylabel('speed (ft/s)')
#    plt.show()
#    plt.savefig('platoonspacetime.png')
        
    find_artists = []
    nartists = len(artist2veh)
    halfn = int(nartists/2)
    def on_pick(event):
        nonlocal find_artists
        ax = event.artist.axes
        curind = ax.lines.index(event.artist) #artist index
        
        if event.mouseevent.button ==1: #left click selects vehicle
            #deselect old vehicle 
            for j in find_artists:
                if j < nartists/2: 
                    ax.lines[j].set_color('C0')
                else: 
                    ax.lines[j].set_color('C1')
            
            #select new vehicle
            vehind = artist2veh[curind] #convert from artist to vehicle index
            platoonind = veh2platoon[vehind]
            find_artists = np.asarray(artist2veh)
            find_artists = np.nonzero(find_artists == vehind)[0] #all artist indices which are associated with vehicle 
#            nartists = len(ax.lines)

            for j in find_artists: 
                if j < nartists/2: #part of measurements because in first half
                    ax.lines[j].set_color('C2')
                else:
                    ax.lines[j].set_color('C3')
            plt.title('Vehicle ID '+str(followerlist[vehind])+ ' has RMSE '+str(round(vehrmse[vehind],2))+' in platoon '+str(platoonlist[platoonind])+' with RMSE '+str(round(platoonrmse[platoonind],2)))
            plt.draw()
            
            
        if event.mouseevent.button ==3: #right click selects platoon
#            print(find_artists)
            #deselect old vehicle 
            for j in find_artists:
                if j < nartists/2: 
                    ax.lines[j].set_color('C0')
                else: 
                    ax.lines[j].set_color('C1')
            
            #select new vehicle
            vehind = artist2veh[curind] #convert from artist to vehicle index
            platoonind = veh2platoon[vehind] #get platoonindex 
#            print(platoonind)
            
            platoonindlist = [] #we will get a list of the vehicle indices for all vehicles in the platoon 
            countvehs = 0 #we first find how many vehicles come before the platoon
            for i in range(platoonind):
                countvehs += len(platoonlist[i][1:])
            for i in range(len(platoonlist[platoonind][1:])):
                platoonindlist.append(countvehs+i)
                
            find_artists = [] #now find all artists which are associated with platoon
            for count, i in enumerate(ax.lines): #get artist index
                if artist2veh[count] in platoonindlist: #convert artist index to vehicle index, check if vehicle index is included in platoon, if it is we keep track of it 
                    find_artists.append(count)
                
#            print(find_artists)
            #select new vehicles 
            for j in find_artists: 
                if j < nartists/2: #part of measurements because in first half
                    ax.lines[j].set_color('C2')
                else:
                    ax.lines[j].set_color('C3')
            plt.title('Vehicle ID '+str(followerlist[vehind])+ ' has RMSE '+str(round(vehrmse[vehind],2))+' in platoon '+str(platoonlist[platoonind])+
                      ' with RMSE '+str(round(platoonrmse[platoonind],2)))
            plt.draw()
            
        return
    
    toggle1 = 1
    def key_press(event):
        nonlocal toggle1
        #toggle out of lane trajectories  for all vehicles
        if event.key in ['T','t']:
            first = True
            visbool = None
            ax = event.inaxes #get axes 
#            print(ax)
#            print(ax.lines)
            for i in ax.lines: 
                if i.get_alpha() != None:  #line has alpha then we do something
                    if first:  #for first line, we need to check whether it is currently turned on or off
                        visbool = i.get_visible() #get visibility
                        visbool = not visbool #set visbility to opposite
                        first = False #after doing this check we don't need to do it again 
                    i.set_visible(visbool) #set desired visbility
            plt.draw()
            
        #toggle out of lane trajectories for current selected vehicle/platoon
        if event.key in ['U','u']:
            first = True
            visbool = None
            ax = event.inaxes #get axes 
#            print(ax)
#            print(ax.lines)
            for z in find_artists: 
                i = ax.lines[z]
                if i.get_alpha() != None:  #line has alpha then we do something
                    if first:  #for first line, we need to check whether it is currently turned on or off
                        visbool = i.get_visible() #get visibility
                        visbool = not visbool #set visbility to opposite
                        first = False #after doing this check we don't need to do it again 
                    i.set_visible(visbool) #set desired visbility
            plt.draw()
            
        #toggle between showing sim, meas, and both 
        if event.key in ['Y','y']:
            ax = event.inaxes
            curtoggle = toggle1 % 3
            if curtoggle == 1: 
                for j in range(0,halfn):
                    ax.lines[j].set_visible(0)
                for j in range(halfn,nartists):
                    ax.lines[j].set_visible(1)
            if curtoggle == 2: 
                for j in range(0,halfn):
                    ax.lines[j].set_visible(1)
                for j in range(halfn,nartists):
                    ax.lines[j].set_visible(0)
            if curtoggle == 0: 
                for j in range(nartists):
                    ax.lines[j].set_visible(1)
            toggle1 += 1
            plt.draw()
        
        return
    
    fig.canvas.callbacks.connect('pick_event',on_pick)
    fig.canvas.callbacks.connect('key_press_event',key_press)
    return 

def plotColorLines(X, Y, SPEED, speed_limit):
	axs = plt.gca()
	c = SPEED
	points = np.array([X, Y]).T.reshape(-1, 1, 2)
	segments = np.concatenate([points[:-1], points[1:]], axis=1)
	# if speed_limit:
	# 	norm = plt.Normalize(speed_limit[0], speed_limit[1])
	# else:
	# 	norm = plt.Normalize(c.min(), c.max())
	norm = plt.Normalize(speed_limit[0], speed_limit[1])
	lc = LineCollection(segments, cmap=palettable.colorbrewer.diverging.RdYlGn_4.mpl_colormap, norm=norm)
	# lc = LineCollection(segments, cmap=cm.get_cmap('RdYlBu'), norm=norm)
	lc.set_array(c)
	lc.set_linewidth(1)
	line = axs.add_collection(lc)
	return line


def platoonplot(meas, sim, followerchain, platoon=[], Colors=False, speed=False, newfig=True, clr=['C0', 'C1'],
                fulltraj=True, lane=None, opacity=.4, colorCode=True, speed_limit=[]):  # plot platoon in space-time
	# CURRENT DOCUMENTATION 
	# meas - measurements in np array, rows are observations
	# sim - simulation in same format as meas. can pass in None and only meas will be shown, or can pass in the data and they will be plotted together
	# in different colors. 
	# followerchain (platooninfo) - dictionary containing information on each vehicle ID
	# platoon - default is [], in which case all keys of followerchain are plotted. If passed in as a platoon (list of vehicle ID as [1:] so first entry not included)
	# only those vehicles will be plotted. 
	# Colors = False - if True, the different parts of each trajectory (pre simulation, simulation, shifted end) will be plotted as different colors
	# speed = False - if True will plot the speed instead of position
	# newfig = True - if True will create a new figure, otherwise it will use the current figure 
	# clr = 'C0', assuming Colors = False, clr will control what colors will be used. Default is ['C0','C1'] which are the default matplotlib colors
	# fulltraj = True - assuming Colors = False, if True, will plot the entire trajectory, if False, will plot only the simulation 
	# lane = None - If passed in as a laneID, the parts of trajectories not in the lane ID given will be made opaque 

	# plots a platoon of vehicles in space-time plot. 

	# OLD DOCUMENTATION
	# this will plot everything in followerchain.keys() if platoon is empty. 
	# otherwise this plots only everything in platoon[1:] 
	# if Colors = True this will color the pre, sim, and post trajectories in different colors. Otherwise 
	# it only shows the simulation time in a single color. 
	# this plots everything on the same plot, so if you have stuff in different lanes you will get crisscrossing type stuff. 
	# you can make some nice plots with this if you make a platoon using makefollowerchain (which will put everything in a single lane).
	# I have an example of that in makeposter.py I believe. 

	# this does not handle delay currently. plotspeed/plotdist do, but those only do single vehicles. 

	# this function works for making a platoon in a single lane  #old output has variable 'lead', we just have 'meas' now
	# if using makefollowerchain leave platoon as empty array, otherwise you need to put in the platoon made from makeplatoon
	# can pass either meas or sim
	#    x = lead[:,1] #time 
	#    y = lead[:,dataind[0]] #space 
	#    plt.plot(x,y)

	c = None

	ind = 2
	artist2veh = []

	indcounter = np.asarray([], dtype=np.int64)  # keeps track of which artists correspond to which vehicle
	if speed:
		ind = 3
	if platoon != []:
		followerchain = helper.platoononly(followerchain, platoon)
	followerlist = followerchain.keys()  # list of vehicle ID

	if newfig:
		fig = plt.figure()

	# this is all deprecated now 
	if Colors:  # these colors are pretty ugly imo 
		for i in followerlist:  # iterate over each vehicle
			veh = meas[i]
			t_nstar, t_n, T_nm1, T_n = followerchain[i][0:4]
			#        t_n = int(followerchain[i][1])
			#        T_nm1 = int(followerchain[i][2])
			#        T_n = int(followerchain[i][3])
			x1 = range(t_nstar, t_n)  # vehicle has leader during this time so we will color it differently in plots 
			x2 = range(t_n, T_nm1 + 1)  # this is the portion of the trajectory that will be shifted during calibration 
			x3 = range(T_nm1 + 1, T_n + 1)

			y1 = veh[0:t_n - t_nstar, ind]
			y2 = veh[t_n - t_nstar:T_nm1 - t_nstar + 1:, ind]
			y3 = veh[T_nm1 - t_nstar + 1:T_n - t_nstar + 1, ind]

			plt.plot(x1, y1, 'k', x2, y2, 'b', x3, y3, 'r')
	#        plt.plot(x2,y2)
	#        plt.plot(x3,y3)

	if not Colors:
		# fig, axs = plt.subplots(1, 1, sharex=True, sharey=True)

		counter = 0

		mymin = 1e10
		mymax = 0
		for i in followerlist:
		    curmin = min(meas[i][:,3])
		    curmax = max(meas[i][:,3])
		    if mymin > curmin:
		        mymin = curmin
		    if mymax < curmax:
		        mymax = curmax

		if not speed_limit:
			speed_limit = [mymin, mymax]

		for i in followerlist:  # iterate over each vehicle
			veh = meas[i]
			t_nstar, t_n, T_nm1, T_n = followerchain[i][0:4]  # note followerchain and platooninfo have same 

			if fulltraj:  # entire trajectory including pre simulation and shifted end
				start = 0
				end = T_n - t_nstar
			else:  # only show trajectory which is simulated 
				start = t_n - t_nstar
				end = T_nm1 - t_nstar
			veh = veh[start:end, :]
			x = veh[:, 1]
			y = veh[:, ind]
			speed_list = veh[:, 3]

			if lane is not None:

				# LCind is a list of indices where the lane the vehicle is in changes. Note that it includes the first and last index. 
				LCind = np.diff(veh[:, 7])
				LCind = np.nonzero(LCind)[0] + 1
				LCind = list(LCind)
				LCind.insert(0, 0);
				LCind.append(len(veh[:, 7]) + 1)

				for j in range(len(LCind) - 1):
					kwargs = {}
					if meas[i][LCind[j], 7] != lane:
						kwargs = {'linestyle': '--', 'alpha': opacity}  # dashed line .4 opacity (60% see through)
						plt.plot(x[LCind[j]:LCind[j+1]],y[LCind[j]:LCind[j+1]],clr[0], **kwargs)
					else:

						X = x[LCind[j]:LCind[j + 1]]
						Y = y[LCind[j]:LCind[j + 1]]
						SPEED = speed_list[LCind[j]:LCind[j + 1]]
						if colorCode:
							# axs = plt.gca()
							# c = Y
							# points = np.array([X, Y]).T.reshape(-1, 1, 2)
							# segments = np.concatenate([points[:-1], points[1:]], axis=1)
							# norm = plt.Normalize(c.min(), c.max())
							# lc = LineCollection(segments, cmap=cm.get_cmap('RdYlBu'), norm=norm)
							# lc.set_array(c)
							# lc.set_linewidth(1)
							# line = axs.add_collection(lc)

							line = plotColorLines(X, Y, SPEED, speed_limit=speed_limit)

							# axs.set_xlim(X.min(), X.max())
							# axs.set_ylim(min(Y), max(Y))
						else:
							plt.plot(x[LCind[j]:LCind[j + 1]], y[LCind[j]:LCind[j + 1]], clr[0], picker=5, **kwargs)
							artist2veh.append(counter)

						indcounter = np.append(indcounter, counter)




			# else:

				# plt.plot(x,y,clr[0])
				# if colorCode:
				# 	line = plotColorLines()

				# else:
				# 	plt.plot(x, y, clr[0], picker=5)
				# 	artist2veh.append(counter)
			counter += 1

		# plt.show()

		if sim is not None:
			counter = 0
			for i in followerlist:  # iterate over each vehicle
				veh = sim[i]
				t_nstar, t_n, T_nm1, T_n = followerchain[i][0:4]  # note followerchain and platooninfo have same 

				if fulltraj:  # entire trajectory including pre simulation and shifted end
					start = 0
					end = T_n - t_nstar
				else:  # only show trajectory which is simulated 
					start = t_n - t_nstar
					end = T_nm1 - t_nstar
				veh = veh[start:end, :]
				x = veh[:, 1]
				y = veh[:, ind]

				if lane is not None:

					# LCind is a list of indices where the lane the vehicle is in changes. Note that it includes the first and last index. 
					LCind = np.diff(veh[:, 7])
					LCind = np.nonzero(LCind)[0] + 1
					LCind = list(LCind)
					LCind.insert(0, 0);
					LCind.append(len(veh[:, 7]))

					for j in range(len(LCind) - 1):
						kwargs = {}
						if sim[i][LCind[j], 7] != lane:
							kwargs = {'linestyle': '--', 'alpha': .4}  # dashed line .4 opacity (60% see through)
						plt.plot(x[LCind[j]:LCind[j + 1]], y[LCind[j]:LCind[j + 1]], clr[1], **kwargs)
						artist2veh.append(counter)
				else:
					plt.plot(x, y, clr[1])
					artist2veh.append(counter)
				counter += 1

		find_artists = []
		nartists = len(artist2veh)
		halfn = int(nartists / 2)

		def on_pick(event):
			nonlocal find_artists
			ax = event.artist.axes
			curind = ax.lines.index(event.artist)  # artist index

			if event.mouseevent.button == 1:  # left click selects vehicle
				# deselect old vehicle
				for j in find_artists:
					ax.lines[j].set_color('C0')

				# select new vehicle
				vehind = artist2veh[curind]  # convert from artist to vehicle index
				find_artists = np.asarray(artist2veh)
				find_artists = np.nonzero(find_artists == vehind)[
					0]  # all artist indices which are associated with vehicle
				#            nartists = len(ax.lines)

				for j in find_artists:
					ax.lines[j].set_color('C1')
				plt.title('Vehicle ID ' + str(list(followerlist)[vehind]))
				plt.draw()

			if event.mouseevent.button == 3:  # right click selects platoon
				#            print(find_artists)
				# deselect old vehicle
				for j in find_artists:
					ax.lines[j].set_color('C0')
			plt.draw()
			# recolor the selected artist and all other artists associated with the vehicle ID so you can see what line you clicked on
			# change the title to say what vehicle you selected.

		fig.canvas.callbacks.connect('pick_event', on_pick)

	#        plt.plot(x2,y2)
	#        plt.plot(x3,y3)

	axs = plt.gca()

	plt.xlabel('time (frameID )')
	plt.ylabel('space (ft)')
	if speed:
		plt.ylabel('speed (ft/s)')

	if colorCode:
		fig.colorbar(line, ax=axs)


	axs.autoscale(axis='x')

	# if speed_limit:
	# 	axs.autoscale(axis='x')
	# 	axs.set_ylim(speed_limit[0], speed_limit[1])
	# else:
	# 	axs.autoscale()
	# modified

#	plt.savefig('platoonspacetime.png')
#	plt.show()
	return



def calculateflows(meas, spacea, timea, agg):
	q = [[] for i in spacea]
	k = [[] for i in spacea]
    #ronan modification - calculate minimum and maximum space, minimum and maximum time
#spacealist = []
#for i in spacea: 
#    spacealist.extend(i)
#spacemin = min(spacealist)
#spacemax = max(spacealist)
#timemin = min(timea)
#timemax = max(timea)

	intervals = []
	start = timea[0]
	end = timea[1]
	temp1 = start
	temp2 = start+agg
	while temp2 < end:
		intervals.append((temp1, temp2))
		temp1 = temp2
		temp2 += agg
	intervals.append((temp1, end))


	# ([time frame], [space])
	regions = [[([],[]) for j in intervals] for i in spacea]

	for id in meas:
		data = meas[id]
        #ronan modification  - prune data so we don't have to look over as many datapoints
#data = data[np.all([data[:,1] < timemax, data[:,1]>timemin], axis=0)]
#data = data[np.all([data[:,2] < spacemax, data[:,2]>spacemin], axis=0)]
        
		region_contained = []
		region_data = {}
		for i in data:
			t = i[1]
			s = i[2]
			t_id = -1
			s_id = -1
			for j in range(len(intervals)):
				if t<=intervals[j][1] and t>=intervals[j][0]: 
					t_id = j
			for j in range(len(spacea)):
				if s<=spacea[j][1] and s>=spacea[j][0]:
					s_id = j
			id_key = str(t_id)+" "+str(s_id)
			if id_key not in region_data:
				# in t, in s, out t, out s
				region_data[id_key] = [t,s,-1,-1]
			else:
				region_data[id_key][2] = t
				region_data[id_key][3] = s
		for i in region_data:
			t_id, s_id = i.split(" ")
			t_id = int(t_id)
			s_id = int(s_id)


			regions[s_id][t_id][0].append(region_data[i][2]-region_data[i][0])
			regions[s_id][t_id][1].append(region_data[i][3] - region_data[i][1])

	for i in range(len(regions)):
		for j in range(len(regions[0])):
			area = (spacea[i][1]-spacea[i][0]) * (intervals[i][1]-intervals[i][0])
			q[i].append(sum(regions[i][j][1])/area)
			k[i].append(sum(regions[i][j][0])/area)
	return q, k




def plotflows(meas, spacea, timea, agg, type = 'FD', FDagg= None):
	"""
	aggregates microscopic data into macroscopic quantities based on Edie's generalized ... definitions of traffic variables
	meas = measurements, in usual format (dictionary where keys are vehicle IDs, values ... are numpy arrays)
	spacea = reads as ``space A'' (where A is the region where the macroscopic quantities ... are being calculated). 
    list of lists, each nested list is a length 2 list which ... represents the starting and ending location on road. So if len(spacea) >1 there ... will be multiple regions on the road which we are tracking
	e.g. spacea = [[200,400],[800,1000]], calculate the flows in regions 200 to 400 and ... 800 to 1000 in meas.
	timea = reads as ``time A'', should be a list of the times (in the local time of the ... data). E.g. timea = [1000,3000] calculate times between 1000 and 3000.
	agg = aggregation length, float number which is the length of each aggregation ... interval. E.g. agg = 300 each measurement of the macroscopic quantities is over ... 300 time units in the data, so in NGSim where each time is a frameID with length ... .1s, we are aggregating every 30 seconds.
	type = `FD', if type is `FD', plot data in flow-density plane. Otherwise, plot in ... flow-time plane.
	FDagg = None - If FDagg is None and len(spacea) > 1, aggregate q and k measurements ... together. Otherwise if FDagg is an int, only show the q and k measurements for the ... corresponding spacea[int]
	`"""
	intervals = []
	start = timea[0]
	end = timea[1]
	temp1 = start
	temp2 = start + agg
	while temp2 < end:
		intervals.append((temp1, temp2))
		temp1 = temp2
		temp2 += agg
	intervals.append((temp1, end))

	q, k = calculateflows(meas, spacea, timea, agg)
	time_sequence = []
	time_sequence_for_line = []

	if len(q)>1 and FDagg != None:
		q = [q[FDagg]]
		k = [k[FDagg]]


	for i in range(len(q)):
		for j in range(len(intervals)):
			time_sequence.append(intervals[j][0])

	for i in range(len(intervals)):
		time_sequence_for_line.append(intervals[i][0])
	unzipped_q = []
	for i in q:
		unzipped_q += i
	unzipped_k = []
	for i in k:
		unzipped_k += i

	if type == 'FD':
		plt.scatter(unzipped_k, unzipped_q, c=time_sequence, cmap=cm.get_cmap('viridis'))
		plt.colorbar()
		plt.xlabel("density")
		plt.ylabel("flow")
		plt.show()

	elif type == 'line':
		for i in range(len(spacea)):
			plt.plot(time_sequence_for_line, q[i])
		plt.xlabel("time")
		plt.ylabel("flow")
		plt.show()

	return




def plotdist(meas1,sim1,platooninfo1, my_id, fulltraj = False, delay = 0 ,h=.1):
    #this is just for a single vehicle
    #this function plots the distance for measurement and simulation, given platooninfo
    #can do either the whole trajectory or just the simulated part
    #will use the current figure so if you want it in its own figure need to initialize a figure beforehand
    #can do either delay or no delay 
    meas = meas1[my_id]
    sim = sim1[my_id]
    platooninfo = platooninfo1[my_id]
    
    if not fulltraj: 
        t_nstar, t_n, T_nm1, T_n = platooninfo[0:4]
        if delay != 0:
            offset = math.ceil(delay/h)
            offsetend = math.floor(delay/h)
            if T_nm1 + offsetend >= T_n: 
                end = T_n 
            else: 
                end = T_nm1 + offsetend
            start = t_n+offset
        else: 
            start = t_n
            end = T_nm1 
        meas = meas[start-t_nstar:end-t_nstar+1,:]
        sim = sim[start-t_nstar:end-t_nstar+1,:]
    
#    plt.figure()
    plt.xlabel('time (.1 seconds)')
    plt.ylabel('space (feet)')
    plt.plot(meas[:,1],meas[:,2],'k',sim[:,1],sim[:,2],'r')
    plt.legend(['Measurements', 'Simulation after calibration'])
    plt.title('Space-Time plot for vehicle '+str(my_id))
    
    return 

def plotspeed(meas1,sim1,platooninfo1, my_id, fulltraj = False, delay = 0, h = .1,newfig = True):
    #this is just for a single vehicle
    #this function plots the distance for measurement and simulation, given platooninfo
    #can do either the whole trajectory or just the simulated part
    #can do either delay or no delay 
    meas = meas1[my_id]
    sim = sim1[my_id]
    platooninfo = platooninfo1[my_id]
    
    if not fulltraj: 
        t_nstar, t_n, T_nm1, T_n = platooninfo[0:4]
        if delay != 0:
            offset = math.ceil(delay/h)
            offsetend = math.floor(delay/h)
            if T_nm1 + offsetend >= T_n: 
                end = T_n 
            else: 
                end = T_nm1 + offsetend
            start = t_n+offset
        else: 
            start = t_n
            end = T_nm1 
        meas = meas[start-t_nstar:end-t_nstar+1,:]
        sim = sim[start-t_nstar:end-t_nstar+1,:]
    
    if newfig:
        plt.figure()
    plt.xlabel('time (.1 seconds)')
    plt.ylabel('speed (feet/second)')
    plt.plot(meas[:,1],meas[:,3],'k',sim[:,1],sim[:,3],'r')
    plt.legend(['Measurements', 'Simulation after calibration'])
    plt.title('Speed-Time plot for vehicle '+str(my_id))
    
    return 
    

#note: velocity headway plots need some tweaking where we pass in rinfo in addition to rp (because of the new strategy for dealing with the estimation of r constants if we recompute rinfo as is currently done
    #we are going to get something which is slightly different from what we should get 
    #this is only an issue if you have effective_headway = True
def plotvhd(meas,sim,platooninfo,my_id,show_sim = True,show_meas=True,effective_headway = False,rp=None,h=.1,datalen=9,end=None, delay=0, newfig = True):
    #plot in the velocity headway plane. 
    #would like to make this so that you can pass in rinfo and it will automatically not connect the lines between before/after the lane changes so you don't get the annoying horizontal lines 
    #in the plot (which occur because of lane changing)
    #note that in ``paperplots.py'' we have written some scripts that do this for the paper plots, in the first 3 sections of code
    if effective_headway:
        leadinfo,folinfo,rinfo = helper.makeleadfolinfo_r3([[],my_id],platooninfo,meas)
    else:
        leadinfo,folinfo,rinfo = helper.makeleadfolinfo([[],my_id],platooninfo,meas)
        
    t_nstar,t_n,T_nm1,T_n = platooninfo[my_id][0:4]
    
    if delay != 0:
        offset = math.ceil(delay/h)
        start = t_n+offset
    else: 
        start = t_n
    
    if end== None: 
        end = T_nm1
    frames = [t_n,T_nm1]
    lead = np.zeros((T_n+1-t_n,datalen)) #initialize the lead vehicle trajectory 
    for j in leadinfo[0]:
        curleadid = j[0] #current leader ID 
        leadt_nstar = int(sim[curleadid][0,1]) #t_nstar for the current lead, put into int 
        lead[j[1]-t_n:j[2]+1-t_n,:] = sim[curleadid][j[1]-leadt_nstar:j[2]+1-leadt_nstar,:] #get the lead trajectory from simulation 
            
    truelead = np.zeros((T_n+1-t_n,datalen)) #initialize the lead vehicle trajectory 
    for j in leadinfo[0]:
        curleadid = j[0] #current leader ID 
        leadt_nstar = int(sim[curleadid][0,1]) #t_nstar for the current lead, put into int 
        truelead[j[1]-t_n:j[2]+1-t_n,:] = meas[curleadid][j[1]-leadt_nstar:j[2]+1-leadt_nstar,:] #get the lead trajectory from simulation 
            
    relax,unused = r_constant(rinfo[0],frames,T_n,rp,False,h) #get the relaxation amounts for the current vehicle; these depend on the parameter curp[-1] only. 
    
    
    headway =  lead[start-t_n:,2]-sim[my_id][start-t_nstar:,2]-lead[start-t_n:,6]+relax[start-t_n:]
    trueheadway = truelead[start-t_n:,2]-meas[my_id][start-t_nstar:,2]-truelead[start-t_n:,6]+relax[start-t_n:]
    ####plotting
    if newfig:
        plt.figure()
    plt.xlabel('space headway (ft)')
    plt.ylabel('speed (ft/s)')
    plt.title('space-headway for vehicle '+str(my_id))
    
    if show_sim: 
        plt.plot(headway[:end+1-start],sim[my_id][start-t_nstar:end+1-t_nstar,3])
#    elif show_sim and not fulltraj: 
#        plt.plot(headway[:T_nm1+1-t_n],sim[my_id][t_n-t_nstar:T_nm1+1-t_nstar,3]) #bug here
    
    if show_meas: 
        plt.plot(trueheadway[:end+1-start],meas[my_id][start-t_nstar:end+1-t_nstar,3])
#    elif show_meas and not fulltraj:
#        plt.plot(trueheadway[:T_nm1+1-t_n],meas[my_id][t_n-t_nstar:T_nm1+1-t_nstar,3])

    if show_meas and show_sim:
        plt.legend(['Simulation','Measurements'])
    
    return

    

def animatevhd(meas,sim,platooninfo,my_id,lentail=20,show_sim = True,show_meas=True,effective_headway = False,rp=None,h=.1,datalen=9, end = None, delay=0):
    
    #my_id - id of the vehicle to plot 
    #lentail = 20 - number of observations to show in the past 
    #show_sim  = True - whether or not to show sim
    #show_meas = True - whether or not to show meas
    #effective_headway = False - if True, computes the relaxation amounts using rp, and then uses the headway + relaxation amount to plot instead of just the headway
    #rp = None - effective headway is true, rp is a float which is the parameter for the relaxation amount 
    #h = .1 - data discretization
    #datalen = 9
    #end = None - last time to show animation 
    #delay = 0 - gets starting time for newell model 
    
    
    
    t_nstar,t_n,T_nm1,T_n = platooninfo[my_id][0:4]
    if delay != 0:
        offset = math.ceil(delay/h)
        start = t_n+offset
    else: 
        start = t_n
        
    #animation in the velocity headway plane
    if effective_headway:
        leadinfo,folinfo,rinfo = helper.makeleadfolinfo_r3([[],my_id],platooninfo,meas,use_merge_constant = True)
    else:
        leadinfo,folinfo,rinfo = helper.makeleadfolinfo([[],my_id],platooninfo,meas)
        
    if end == None: 
        end = T_nm1
    frames = [t_n,T_nm1]
    relax,unused = r_constant(rinfo[0],frames,T_n,rp,False,h) #get the relaxation amounts for the current vehicle; these depend on the parameter curp[-1] only. 
    
    if sim is not None: 
        lead = np.zeros((T_n+1-t_n,datalen)) #initialize the lead vehicle trajectory 
        for j in leadinfo[0]:
            curleadid = j[0] #current leader ID 
            leadt_nstar = int(sim[curleadid][0,1]) #t_nstar for the current lead, put into int 
            lead[j[1]-t_n:j[2]+1-t_n,:] = sim[curleadid][j[1]-leadt_nstar:j[2]+1-leadt_nstar,:] #get the lead trajectory from simulation 
        headway =  lead[start-t_n:,2]-sim[my_id][start-t_nstar:,2]-lead[start-t_n:,6]+relax[start-t_n:]
            
    
    truelead = np.zeros((T_n+1-t_n,datalen)) #initialize the lead vehicle trajectory 
    for j in leadinfo[0]:
        curleadid = j[0] #current leader ID 
        leadt_nstar = int(meas[curleadid][0,1]) #t_nstar for the current lead, put into int 
        truelead[j[1]-t_n:j[2]+1-t_n,:] = meas[curleadid][j[1]-leadt_nstar:j[2]+1-leadt_nstar,:] #get the lead trajectory from simulation 
    trueheadway = truelead[start-t_n:,2]-meas[my_id][start-t_nstar:,2]-truelead[start-t_n:,6]+relax[start-t_n:]
    ####plotting
    fig = plt.figure()
    plt.xlabel('space headway (ft)')
    plt.ylabel('speed (ft/s)')
    plt.title('space-headway for vehicle '+str(my_id))
    
    ims = []
    
    
    
    if show_sim and show_meas:
        for i in range(len(headway)-lentail-(T_n-end)):
            t_n = start
            im = plt.plot(headway[i:i+lentail],sim[my_id][t_n-t_nstar+i:t_n-t_nstar+i+lentail,3],'C0',trueheadway[i:i+lentail],meas[my_id][t_n-t_nstar+i:t_n-t_nstar+i+lentail,3],'C1',headway[i+lentail],sim[my_id][t_n-t_nstar+i+lentail,3],'ko',trueheadway[i+lentail],meas[my_id][t_n-t_nstar+i+lentail,3],'ko')
            ims.append(im)
#        plt.legend(['Simulation','Measurements']) #for some reason this makes things very slow 
            
    elif show_sim: 
        for i in range(len(headway)-lentail-(T_n-end)):
            t_n = start
            im = plt.plot(headway[i:i+lentail],sim[my_id][t_n-t_nstar+i:t_n-t_nstar+i+lentail,3],'C0',headway[i+lentail],sim[my_id][t_n-t_nstar+i+lentail,3],'ko')
            ims.append(im)
    elif show_meas: 
        for i in range(len(trueheadway)-lentail-(T_n-end)):
            t_n = start
            im = plt.plot(trueheadway[i:i+lentail],meas[my_id][t_n-t_nstar+i:t_n-t_nstar+i+lentail,3],'C0',trueheadway[i+lentail],meas[my_id][t_n-t_nstar+i+lentail,3],'ko')
            ims.append(im)
        
    im_ani = animation.ArtistAnimation(fig,ims,interval=0)
    
    return im_ani

def animatevhd_list(meas, sim, platooninfo, my_id, lentail=20, show_sim=True, show_meas=True, effective_headway=False,
                    rp=None, h=.1, datalen=9, start=None, end=None, delay=0):
	# my_id - id of the vehicle to plot
	# lentail = 20 - number of observations to show in the past
	# show_sim  = True - whether or not to show sim
	# show_meas = True - whether or not to show meas
	# effective_headway = False - if True, computes the relaxation amounts using rp, and then uses the headway + relaxation amount to plot instead of just the headway
	# rp = None - effective headway is true, rp is a float which is the parameter for the relaxation amount
	# h = .1 - data discretization
	# datalen = 9
	# end = None - last time to show animation
	# delay = 0 - gets starting time for newell model
	fig = plt.figure()
	plt.xlabel('space headway (ft)')
	plt.ylabel('speed (ft/s)')
	plt.title('space-headway for vehicle ' + " ".join(list(map(str, (my_id)))))
	line_data = {}
	id2Line = {}
	# 0: tnstar, 1: tn, 2: t
	for id in my_id:

		t_nstar, t_n, T_nm1, T_n = platooninfo[id][0:4]
		if not start:
			if delay != 0:
				offset = math.ceil(delay / h)
				start = t_n + offset
			else:
				start = t_n

		# animation in the velocity headway plane
		if effective_headway:
			leadinfo, folinfo, rinfo = helper.makeleadfolinfo_r3([[], id], platooninfo, meas)
		else:
			leadinfo, folinfo, rinfo = helper.makeleadfolinfo([[], id], platooninfo, meas)

		if end == None:
			end = T_nm1
		frames = [t_n, T_nm1]
		relax, unused = r_constant(rinfo[0], frames, T_n, rp, False,
		                           h)  # get the relaxation amounts for the current vehicle; these depend on the parameter curp[-1] only.

		truelead = np.zeros((T_n + 1 - t_n, datalen))  # initialize the lead vehicle trajectory
		if sim is not None:
			lead = np.zeros((T_n + 1 - t_n, datalen))  # initialize the lead vehicle trajectory
			for j in leadinfo[0]:
				curleadid = j[0]  # current leader ID
				leadt_nstar = int(sim[curleadid][0, 1])  # t_nstar for the current lead, put into int
				lead[j[1] - t_n:j[2] + 1 - t_n, :] = sim[curleadid][j[1] - leadt_nstar:j[2] + 1 - leadt_nstar,
				                                     :]  # get the lead trajectory from simulation
			headway = lead[start - t_n:, 2] - sim[id][start - t_nstar:, 2] - lead[start - t_n:, 6] + relax[start - t_n:]
			for j in leadinfo[0]:
				curleadid = j[0]  # current leader ID
				leadt_nstar = int(meas[curleadid][0, 1])  # t_nstar for the current lead, put into int
				truelead[j[1] - t_n:j[2] + 1 - t_n, :] = meas[curleadid][j[1] - leadt_nstar:j[2] + 1 - leadt_nstar,
				                                         :]  # get the lead trajectory from simulation
			trueheadway = truelead[start - t_n:, 2] - meas[id][start - t_nstar:, 2] - truelead[start - t_n:, 6] + relax[
			                                                                                                      start - t_n:]
			index = start - t_n
			for i in range(len(headway) - lentail - (T_n - end)):
				line1 = (headway[i:i + lentail], sim[id][t_n - t_nstar + i:t_n - t_nstar + i + lentail, 3])
				line2 = (trueheadway[i:i + lentail], meas[id][t_n - t_nstar + i:t_n - t_nstar + i + lentail, 3])
				line3 = (headway[i + lentail], sim[id][t_n - t_nstar + i + lentail, 3])
				line4 = (trueheadway[i + lentail], meas[id][t_n - t_nstar + i + lentail, 3])
				if i + index in line_data.keys():
					line_data[i + index].append((line1, line2, line3, line4, id))
				else:
					line_data[i + index] = [(line1, line2, line3, line4, id)]
		else:
			lead = np.zeros((T_n + 1 - t_n, datalen))  # initialize the lead vehicle trajectory
			for j in leadinfo[0]:
				curleadid = j[0]  # current leader ID
				leadt_nstar = int(meas[curleadid][0, 1])  # t_nstar for the current lead, put into int
				truelead[j[1] - t_n:j[2] + 1 - t_n, :] = meas[curleadid][j[1] - leadt_nstar:j[2] + 1 - leadt_nstar,
				                                         :]  # get the lead trajectory from simulation
			trueheadway = truelead[start - t_n:, 2] - meas[id][start - t_nstar:, 2] - truelead[start - t_n:, 6] + relax[
			                                                                                                      start - t_n:]
			index = start - t_n
			for i in range(len(trueheadway) - lentail - (T_n - end)):
				# line1 = (headway[i:i + lentail], sim[id][t_n - t_nstar + i:t_n - t_nstar + i + lentail, 3])
				line1 = None
				line2 = (trueheadway[i:i + lentail], meas[id][t_n - t_nstar + i:t_n - t_nstar + i + lentail, 3])
				# line3 = (headway[i + lentail], sim[id][t_n - t_nstar + i + lentail, 3])
				line3 = None
				line4 = (trueheadway[i + lentail], meas[id][t_n - t_nstar + i + lentail, 3])
				if i + index in line_data.keys():
					line_data[i + index].append((line1, line2, line3, line4, id))
				else:
					line_data[i + index] = [(line1, line2, line3, line4, id)]


		truelead = np.zeros((T_n + 1 - t_n, datalen))  # initialize the lead vehicle trajectory
		for j in leadinfo[0]:
			curleadid = j[0]  # current leader ID
			leadt_nstar = int(meas[curleadid][0, 1])  # t_nstar for the current lead, put into int
			truelead[j[1] - t_n:j[2] + 1 - t_n, :] = meas[curleadid][j[1] - leadt_nstar:j[2] + 1 - leadt_nstar,
			                                         :]  # get the lead trajectory from simulation
		trueheadway = truelead[start - t_n:, 2] - meas[id][start - t_nstar:, 2] - truelead[start - t_n:, 6] + relax[
		                                                                                                      start - t_n:]
	####plotting

	ims = []

	# if show_sim and show_meas:
	#     for i in range(len(headway) - lentail - (T_n - end)):
	#         t_n = start
	#         im = plt.plot(headway[i:i + lentail], sim[id][t_n - t_nstar + i:t_n - t_nstar + i + lentail, 3], 'C0',
	#                       trueheadway[i:i + lentail], meas[id][t_n - t_nstar + i:t_n - t_nstar + i + lentail, 3],
	#                       'C1', headway[i + lentail], sim[id][t_n - t_nstar + i + lentail, 3], 'ko',
	#                       trueheadway[i + lentail], meas[id][t_n - t_nstar + i + lentail, 3], 'ko')
	#         ims.append(im)
	# #        plt.legend(['Simulation','Measurements']) #for some reason this makes things very slow
	#
	# elif show_sim:
	#     for i in range(len(headway) - lentail - (T_n - end)):
	#         t_n = start
	#         im = plt.plot(headway[i:i + lentail], sim[id][t_n - t_nstar + i:t_n - t_nstar + i + lentail, 3], 'C0',
	#                       headway[i + lentail], sim[id][t_n - t_nstar + i + lentail, 3], 'ko')
	#         ims.append(im)
	# elif show_meas:
	#     for i in range(len(trueheadway) - lentail - (T_n - end)):
	#         t_n = start
	#         im = plt.plot(trueheadway[i:i + lentail], meas[id][t_n - t_nstar + i:t_n - t_nstar + i + lentail, 3],
	#                       'C0', trueheadway[i + lentail], meas[id][t_n - t_nstar + i + lentail, 3], 'ko')
	#         ims.append(im)
	ax = plt.gca()
	i = 0
	# line1, = ax.plot(headway[i:i + lentail], sim[id][t_n - t_nstar + i:t_n - t_nstar + i + lentail, 3], 'C0')
	# line2, = ax.plot(trueheadway[i:i + lentail], meas[id][t_n - t_nstar + i:t_n - t_nstar + i + lentail, 3], 'C1')
	# line3, = ax.plot(headway[i + lentail],sim[id][t_n - t_nstar + i + lentail, 3], 'ko' )
	# line4, = ax.plot(trueheadway[i + lentail], meas[id][t_n - t_nstar + i + lentail, 3], 'ko')
	ax.set_xlim(0, 80)
	ax.set_ylim(0, 60)
	sortedKeys = list(sorted(line_data.keys()))
	curLines = []
	annotion_list = []

	def aniFunc(frame):

		lines = line_data[sortedKeys[frame]]
		ids = [j[4] for j in lines]

		for j in lines:
			id = j[4]

			if id in curLines:
				line1, line2, line3, line4, id = id2Line[id]

				if line1:

					line1.set_xdata(j[0][0])
					line1.set_ydata(j[0][1])

				line2.set_xdata(j[1][0])
				line2.set_ydata(j[1][1])

				# line3.set_xdata(j[2][0])
				# line3.set_ydata(j[2][1])

				# line4.set_xdata(j[3][0])
				# line4.set_ydata(j[3][1])



				if line3:
					line3.set_x(j[2][0])
					line3.set_y(j[2][1])

				line4.set_x(j[3][0])
				line4.set_y(j[3][1])
			else:

				line2, = ax.plot(j[1][0], j[1][1],
				                 'C1')
				# line3, = ax.plot(j[2][0], j[2][1], 'ko')
				# line4, = ax.plot(j[3][0], j[3][1], 'ko')

				if sim != None:
					line1, = ax.plot(j[0][0], j[0][1],
					                 'C0')
					line3 = ax.annotate(str(id), (j[2][0], j[2][1]), fontsize=7)

					annotion_list.append(line3)

				else:
					line1 = None
					line3 = None



				line4 = ax.annotate(str(id), (j[3][0], j[3][1]), fontsize=7)
				annotion_list.append(line4)
				id2Line[id] = (line1, line2, line3, line4, id)
				curLines.append(id)
		for line_id in curLines.copy():
			if line_id not in ids:
				line = id2Line[line_id]
				for plotted_line in line:
					if plotted_line in ax.lines:
						plotted_line.remove()
				curLines.remove(line_id)
				del id2Line[line_id]

	def init():
		for i in annotion_list:
			i.remove()
		annotion_list.clear()

		# line1.set_xdata(headway[i:i + lentail])
		# line1.set_ydata(sim[id][t_n - t_nstar + i:t_n - t_nstar + i + lentail, 3])
		#
		# line2.set_xdata(trueheadway[i:i + lentail])
		# line2.set_ydata(meas[id][t_n - t_nstar + i:t_n - t_nstar + i + lentail, 3])
		#
		# line3.set_xdata(headway[i + lentail])
		# line3.set_ydata(sim[id][t_n - t_nstar + i + lentail, 3])
		#
		# line4.set_xdata(trueheadway[i + lentail])
		# line4.set_ydata(meas[id][t_n - t_nstar + i + lentail, 3])

	# im_ani = animation.FuncAnimation(fig, aniFunc, frames=range(len(headway) - lentail - (T_n - end)), interval=3)
	im_ani = animation.FuncAnimation(fig, aniFunc, init_func=init, frames=len(sortedKeys), interval=0)
	plt.show()
	return im_ani

    

def animatetraj(meas, followerchain, platoon=[], usetime=[], presim=True, postsim=True, datalen=9):
	# platoon = [] - if given as a platoon, only plots those vehicles in the platoon (e.g. [[],1,2,3] )
	# usetime = [] - if given as a list, only plots those times in the list (e.g. list(range(1,100)) )
	# presim = True - presim and postsim control whether the entire trajectory is displayed or just the simulated parts (t_nstar - T_n versus T-n - T_nm1)
	# postsim = True

	# note that followerchain is essentially the same as platooninfo.
	# either plots everything in followerchain.keys() (if platoon is empty) or everything in platoon (if platoon is not empty)

	# currently this is pretty slow but as long as the platoon you want to look at is small it's fine. Also, everything is just a black dot so it can be a little tricky to see the 
	# vehicle IDs.
	from mpl_toolkits.axes_grid1 import make_axes_locatable
	if platoon != []:
		followerchain = helper.platoononly(followerchain, platoon)
	platoontraj, mytime = helper.arraytraj(meas, followerchain, presim, postsim, datalen)
	if not usetime:
		usetime = mytime

	fig = plt.figure(figsize=(10, 4))  # initialize figure and axis 
	ax = fig.add_axes([0, 0, 1, 1], frameon=False)
	ax.set_xlim(0, 1600), ax.set_xlabel('localY')
	ax.set_ylim(7.5, 0), ax.set_ylabel('laneID')

	# ims = []

	# scatter_pts = ax.scatter([], [], c='k')
	# im = ax.imshow([(100,100)], origin='lower')
	scatter_pts = ax.scatter([], [], c=[], cmap=cm.get_cmap('RdYlBu'), marker=">")

	# fig.colorbar(im,cmap=cm.get_cmap('RdYlBu'))
	# for i in usetime:
	#     curdata = platoontraj[i]
	#
	#     ims.append((plt.scatter(curdata[:,2], curdata[:,7],c='k'),))
	#        plt.show()

	# im_ani = animation.ArtistAnimation(fig,ims,interval=3)
	annotionList = []
	norm = plt.Normalize(0, 80)
	# divider = make_axes_locatable(ax)
	# cax = divider.append_axes("right", size="25%", pad=0.2)

	fig.colorbar(scatter_pts, cmap=cm.get_cmap('RdYlBu'), norm=norm, shrink=0.7)

	def aniFunc(frame):
		ax = plt.gca()
		for i in annotionList:
			i.remove()
		annotionList.clear()
		curdata = platoontraj[usetime[frame]]
		X = curdata[:, 2]
		Y = curdata[:, 7]
		speeds = curdata[:, 3]
		ids = curdata[:, 0]
		for i in range(len(ids)):
			annotionList.append(ax.annotate(str(int(ids[i])), (X[i], Y[i]), fontsize=7))
		# norm = plt.Normalize(speeds.min(), speeds.max())
		c = speeds
		pts = [[X[i], Y[i]] for i in range(len(X))]
		data = np.vstack(pts)
		scatter_pts.set_offsets(data)
		scatter_pts.set_array(c)

	def init():
		frame = 0
		ax = plt.gca()
		for i in annotionList:
			i.remove()
		annotionList.clear()
		curdata = platoontraj[usetime[frame]]
		X = curdata[:, 2]
		Y = curdata[:, 7]
		speeds = curdata[:, 3]
		ids = curdata[:, 0]
		for i in range(len(ids)):
			annotionList.append(ax.annotate(str(int(ids[i])), (X[i], Y[i]), fontsize=7))

		c = speeds
		pts = [[X[i], Y[i]] for i in range(len(X))]
		data = np.vstack(pts)
		scatter_pts.set(norm=norm)
		scatter_pts.set_offsets(data)
		scatter_pts.set_array(c)
		# fig.colorbar(scatter_pts, cmap=cm.get_cmap('RdYlBu'), norm=norm)
		# fig.colorbar(scatter_pts, cmap=cm.get_cmap('RdYlBu'))

	im_ani = animation.FuncAnimation(fig, aniFunc, init_func=init, frames=len(usetime), interval=3)

	plt.show()
	return im_ani 

def wtplot(meas, ID):
    
    testveh = ID
    
    test = meas[testveh][:,3]
    
#    X = np.arange(150,250)
    X = np.arange(15,100) 
    #X = np.array([64])
    out,out2 = pywt.cwt(test,X,'mexh',sampling_period=.1)
    Y = meas[testveh][:,1]
    X,Y, = np.meshgrid(X,Y); X =np.transpose(X); Y = np.transpose(Y)
    
    plt.close('all')
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X,Y,np.abs(out),cmap = cm.coolwarm)
    ax.view_init(azim=0, elev=90)
    
    #plt.figure()
    #plt.imshow(out,cmap = cm.coolwarm)
    plt.figure(figsize=(11,5))
    plt.subplot(1,2,1)
    plotspeed(meas,meas,platooninfo,testveh,fulltraj = True)
    
    energy = np.sum(np.abs(out),axis = 0)
    plt.subplot(1,2,2)
    plt.plot(meas[testveh][:,1],energy)
    plt.show()
    
    return

def wt(series,scale):
    out,out2 = pywt.cwt(series,scale,'mexh')
    energy = np.sum(np.abs(out),0)
    return energy
#example of executing above code 
#wtplot(meas,100) #example of output; 
    
def meanspeedplot(data, timeint, spacebins, lane = 1, use_avg = 'mean'):
    #data - raw data format (loaded in from csv and put into our vanilla format)
    #timeint - number of observations of data (in terms of data resolution) in each aggregated speed; 
         #e.g. 50 with timestep of .1 = 5 second aggregation
    #spacebins - number of intervals the position on road is divided into 
    #use_avg = 'mean' - controls averaging for speeds. if 'mean' then does arithmetic mean. if 'harm' then harmonic mean. 
    #lane = 1 - choose which lane of the data to plot. (does 1 lane at a time)
    
    #choose which lane. (note a possible feature would be the ability to choose multiple lanes at once)
    data = data[data[:,7]==lane]
    
    #can make this iterate once instead of 4 times 
    t0 = min(data[:,1])
    tend = max(data[:,1])
    x0 = min(data[:,2])
    xend = max(data[:,2])
    
    times = np.arange(t0,tend,timeint)
    if times[-1] != tend:
        times = np.append(times,tend)
    
    xint = (xend-x0)/spacebins
    x = np.arange(x0,xend,xint)
    x = np.append(x,xend)
    
    X, Y = np.meshgrid(times, x) #leave out the last point where making the grid to plot over
    
    speeds = []
    for i in range(len(times)-1): #minus 1 here because we have n+1 points but this defines only n bins. 
        speeds.append([])
        for j in range(len(x)-1 ):
            speeds[i].append([])
    
    for i in range(len(data)): #sort all of the speeds 
        curt = data[i,1]
        curx = data[i,2]
        curv = data[i,3]
        #this is too slow replace with modulus operations 
        
        curtimebin = math.floor((curt-t0)/timeint)
        curxbin = math.floor((curx-x0)/xint)
        
#        for j in range(len(times)-1):
#            if curt < times[j+1]: #if the datapoint fits into the current bin 
#                curtimebin = j
#                break #break loop
#        for k in range(len(x)-1):
#            if curx < x[k+1]: 
#                curxbin = k
#                break
#        speeds[curtimebin][curxbin].append(curv)
        
        try: 
            speeds[curtimebin][curxbin].append(curv)
        except IndexError:  #at most extreme point we want it to go into the last bin not into a new bin (which would contain only itself)
            if curxbin == len(x):
                curxbin += -1 
            if curtimebin == len(times):
                curtimebin += -1 
        
    meanspeeds = X.copy() #initialize output
    for i in range(len(times)-1):  #populate output
        for j in range(len(x)-1):
            cur = speeds[i][j]
            if 'mean' in use_avg:  #choose whether to do arithmetic or harmonic mean. if bin is empty, we set the value as np.nan (not the same as math.nan which is default)
                cur = np.mean(cur)
                if math.isnan(cur):
                    cur = np.nan #np.nan will get changed into different color 
            else: 
                if len(cur) ==0:
                    cur = np.nan
                else:
                    cur = harmonic_mean(cur)
    
            meanspeeds[j,i] = cur
            ######
    #surface plot #old#
#    fig = plt.figure(figsize = (12,5))
#    ax = fig.gca(projection='3d')
#    cmap = cm.coolwarm
#    cmap.set_bad('white',1.)
#    surf = ax.plot_surface(X,Y,meanspeeds,cmap = cmap)
#    
#    #change angle
#    ax.view_init( azim=270, elev=90)
#    #hide grid and z axis
#    ax.grid(False)
#    ax.set_zticks([])
#    ax.set_xlabel('Time (.1s)')
#    ax.set_ylabel('Space (ft)')
    ##########
    
    cmap = cm.RdYlBu #RdYlBu is probably the best colormap overall for this 
    cmap.set_bad('white',1.) #change np.nan into white color
    
    fig2 = plt.figure()
#    ax2 = fig2.add_subplot(111)
#    ax2.imshow(meanspeeds, cmap=cmap)
    plt.pcolormesh(X,Y,meanspeeds,cmap=cmap)
    plt.xlabel('Time (.1 s)')
    plt.ylabel('Space (ft)')
    cbar = plt.colorbar()
    cbar.set_label('Speed (ft/s)')
        
    return speeds, X, Y, meanspeeds

#below code will allow you to select rectangles but here we want to select polygons. 

def line_select_callback(eclick, erelease):
    'eclick and erelease are the press and release events'
    x1, y1 = eclick.xdata, eclick.ydata
    x2, y2 = erelease.xdata, erelease.ydata
    print("(%3.2f, %3.2f) --> (%3.2f, %3.2f)" % (x1, y1, x2, y2))
    print(" The button you used were: %s %s" % (eclick.button, erelease.button))
    


def toggle_selector(event):
    print(' Key pressed.')
    if event.key in ['Q', 'q'] and toggle_selector.RS.active:
        print(' RectangleSelector deactivated.')
        toggle_selector.RS.set_active(False)
    if event.key in ['A', 'a']: #and not toggle_selector.RS.active:
        print(' RectangleSelector activated.')
        toggle_selector.RS.set_active(True)
    if event.key == 'enter':
        print(toggle_selector.RS.extents)
        

def selectoscillation(data1, timeint, spacebins, lane = 1, use_avg = 'mean', region_shape='p'):
    """
    \\ TO DO \\
    selectvehID needs some more features added, refer to notes on paper for more details. 
    Basically we want to be able to sort vehicles on key press (note sortveh3 has been tested and is passing), remove vehicles before/after, 
    arrow keys to manually go through selected vehicles, 
    some ability to put up multiple speed/wavelet series at a time might be nice as well. 
    
    I think there is a bug with the initialization of selectvehID 
    \\ TO DO \\
    """
    
    #data1 - raw data format (loaded in from csv and put into our vanilla format)
    #timeint - number of observations of data (in terms of data resolution) in each aggregated speed; 
         #e.g. 50 with timestep of .1 = 5 second aggregation
    #spacebins - number of intervals the position on road is divided into 
    #use_avg = 'mean' - controls averaging for speeds. if 'mean' then does arithmetic mean. if 'harm' then harmonic mean. 
    #lane = 1 - choose which lane of the data to plot. (does 1 lane at a time)
    
    #choose which lane. (note a possible feature would be the ability to choose multiple lanes at once)
    data = data1[data1[:,7]==lane]
    
    #can make this iterate once instead of 4 times 
    t0 = min(data[:,1])
    tend = max(data[:,1])
    x0 = min(data[:,2])
    xend = max(data[:,2])
    
    times = np.arange(t0,tend,timeint)
    if times[-1] != tend:
        times = np.append(times,tend)
    
    xint = (xend-x0)/spacebins
    x = np.arange(x0,xend,xint)
    x = np.append(x,xend)
    
    X, Y = np.meshgrid(times, x) #leave out the last point where making the grid to plot over
    
    speeds = [] #store lists of speeds here
    for i in range(len(times)-1): #minus 1 here because we have n+1 points but this defines only n bins. 
        speeds.append([])
        for j in range(len(x)-1 ):
            speeds[i].append([])
            
    veh = [] #store which vehicles are giving measurements to each bin of the data 
    for i in range(len(times)-1):
        veh.append([])
        for j in range(len(x)-1):
            veh[i].append(set())
    
    for i in range(len(data)): #sort all of the speeds 
        curt = data[i,1]
        curx = data[i,2]
        curv = data[i,3]
        curveh = data[i,0]
        
        curtimebin = math.floor((curt-t0)/timeint)
        curxbin = math.floor((curx-x0)/xint)
        
        try: 
            speeds[curtimebin][curxbin].append(curv)
            veh[curtimebin][curxbin].add(curveh)
        except IndexError: #at most extreme point we want it to go into the last bin not into a new bin (which would contain only itself)
            if curxbin == len(x)-1:
                curxbin += -1 
            if curtimebin == len(times)-1:
                curtimebin += -1 
            speeds[curtimebin][curxbin].append(curv)
            veh[curtimebin][curxbin].add(curveh)
        
    meanspeeds = X.copy() #initialize output
    for i in range(len(times)-1):  #populate output
        for j in range(len(x)-1):
            cur = speeds[i][j]
            if 'mean' in use_avg: #choose whether to do arithmetic or harmonic mean. if bin is empty, we set the value as np.nan (not the same as math.nan which is default)
                cur = np.mean(cur)
                if math.isnan(cur):
                    cur = np.nan
            else: 
                if len(cur) ==0:
                    cur = np.nan
                else:
                    cur = harmonic_mean(cur)
    
            meanspeeds[j,i] = cur
            
    cmap = cm.RdYlBu #RdYlBu is probably the best colormap overall for this 
    cmap.set_bad('white',1.) #change np.nan into white color
    
    fig, current_ax = plt.subplots(figsize=(12,8))
    
    
            
    def my_callback(args): #this can be used to modify the current shape (for example, to make it into a perfect parralelogram, or to round the points to the nearest bin, etc.)
        #this will make the shape into a parralelogram with horizontal top and bottom. the first two points define the left side length, and the third point defines 
        #the length of the other side. So the fourth point is therefore unused. 
        
        
        if region_shape != 'p': #the callback function only does something when a parralelogram shape ('p') is requested 
            return 
        
        #note that you have to do a 4 sided shape 
        testx = mytoggle_selector.RS._xs #when shape is completed we get the coordinates of its vertices
        testy = mytoggle_selector.RS._ys

        #get x and y displacements of the first line
        lenx = testx[0]-testx[1]
        leny = testy[0]-testy[1]
        
        #set third point
        testy[2]=testy[1]
        #set last point to complete the parralogram
        testy[3] = testy[2]+leny
        testx[3] = testx[2]+lenx
        #redraw the shape 
        mytoggle_selector.RS._draw_polygon()
        
        return
    
    vehhelper = copy.deepcopy(veh) #vehhelper is going to keep track of what vehicles in each bin have not yet been plotted 
    vehlist = [] #list of vehicle trajectories weve plotted 
    lineobjects = [] #list of line objects weve plotted (note that we are including the option to remove all the things )
    
    vertlist = [] #list of the vertex corners 
    
    def mytoggle_selector(event):
        #these keep track of which vehicles we are showing in the plot
        nonlocal vehhelper
        nonlocal vehlist
        nonlocal lineobjects
        #keep track of the areas of oscillation identified
        nonlocal vertlist
        if event.key in ['A','a']: #this will select a vehicle trajectory in the bin to plot 
            curx, cury = event.xdata, event.ydata #get mouse position
            #translate the mouse position into the bin we want to use 
            curtimebin = math.floor((curx-t0)/timeint)
            curxbin = math.floor((cury-x0)/xint)
            
            if curxbin == len(x)-1: #prevent index error
                curxbin += -1 
            if curtimebin == len(times)-1: #prevent index error
                curtimebin += -1 
                
            newveh = None
            curlen = len(vehhelper[curtimebin][curxbin])
            while newveh is None and curlen > 0:
                newveh = vehhelper[curtimebin][curxbin].pop()
                if newveh in vehlist: 
                    continue
                else: 
                    curlen = curlen-1
                    vehlist.append(newveh)
                    temp = data[data[:,0]==newveh]
                    indjumps = sequential(temp) 
                    for i in range(len(indjumps)-1):
                        plotx, ploty = temp[indjumps[i]:indjumps[i+1],1],temp[indjumps[i]:indjumps[i+1],2]
                        newline = plt.plot(plotx,ploty,'C0',scalex=False,scaley=False)
                        lineobjects.append(newline)
                    plt.draw() #this will force the line to show immediately 
                    break #break so we won't go to else
                    
            else: 
#                print('no more vehicles to plot in this region, try another.')
                pass
                    
        if event.key in ['D','d']: #on this key press we remove all the trajectories 
            #reset all the plotting stuff 
            vehhelper = copy.deepcopy(veh) #vehhelper is going to keep track of what vehicles in each bin have not yet been plotted 
            vehlist = [] #list of vehicle trajectories weve plotted 
            for i in range(len(lineobjects)):
                lineobjects[i][0].remove() #remove all lines
            plt.draw()
            lineobjects = []
            
        if event.key in ['W','w']: #put top side of the shape to the top 
            testx = mytoggle_selector.RS._xs #when shape is completed we get the coordinates of its vertices
            testy = mytoggle_selector.RS._ys
            
            lenx = testx[0]-testx[1]
            leny = testy[0]-testy[1]

            dy = xend-testy[0]
            dx = dy*lenx/leny
            #update top side
            testx[0] += dx
            testy[0] += dy
            testx[-2] += dx
            testy[-2] += dy
            testx[-1] = testx[0]
            testy[-1] = testy[0]
            
            mytoggle_selector.RS._draw_polygon()
            
        if event.key in ['X','x']: #put bottom side of the shape to the bottom 
            testx = mytoggle_selector.RS._xs #when shape is completed we get the coordinates of its vertices
            testy = mytoggle_selector.RS._ys
            
            lenx = testx[0]-testx[1]
            leny = testy[0]-testy[1]
            theta = math.atan2(leny,lenx)-math.pi #want the angle going in the other direction so minus pi
            dy = -x0+testy[1]
            dx = dy*lenx/leny
            #update bottom side
            testx[1] += -dx
            testy[1] += -dy
            testx[2] += -dx
            testy[2] += -dy
            
            mytoggle_selector.RS._draw_polygon()
            
            
            
        if event.key == 'enter': #enter means we are happy with the current region selected and want to choose another. 
            print(mytoggle_selector.RS.verts)
            vertlist.append(mytoggle_selector.RS.verts) #keep track of the previous region 
#            plt.plot(mytoggle_selector.RS._xs, mytoggle_selector.RS._ys, 'k-', linewidth = 2, alpha=.6)
            
            
            #start new polygonselector
            mytoggle_selector.RS = PolygonSelector(current_ax, my_callback, 
                                           lineprops = dict(color='k', linestyle='-',linewidth=2, alpha=0.4), 
                                           markerprops = dict(marker='o',markersize=2, mec='k', mfc='k', alpha=0.4))
            plt.connect('key_press_event', mytoggle_selector)
            plt.show()
            
        if event.key in ['N', 'n']:
            if len(mytoggle_selector.RS.verts) == 4: 
                vertlist.append(mytoggle_selector.RS.verts)
            selectvehID(data1,times,x,lane,veh,vertlist)
            
    


    plt.pcolormesh(X,Y,meanspeeds,cmap=cmap) #pcolormesh is similar to imshow but is meant for plotting whereas imshow is for actual images 
    plt.xlabel('Time (.1 s)')
    plt.ylabel('Space (ft)')
    cbar = plt.colorbar() #colorbar
    cbar.set_label('Speed (ft/s)')   
    
    print('a to plot a vehicle trajectory, d to clear all vehicle trajectories')
    print('click with mouse to set a corner that encloses a region of the data')
    print('enter to select a new region to identify, click + shift to drag, esc to start over with current region')
    print('w to snap top side of shape to top of data, x to snap bottom side of shape to bottom of data')
    print('when all regions of interest are identified, press n to move to next stage of the process' )
    
    
    mytoggle_selector.RS = PolygonSelector(current_ax, my_callback, 
                                           lineprops = dict(color='k', linestyle='-',linewidth=2, alpha=0.4), 
                                           markerprops = dict(marker='o',markersize=2, mec='k', mfc='k', alpha=0.4))
    plt.connect('key_press_event', mytoggle_selector)
    plt.show()
    
    return times,x,lane,veh

def point2bin(xpoint,ypoint,x,y,x0,y0,xint,yint):
    #translates ypoint, xpoint into bin indices for bin edges y and x
    #ypoint - y coordinate of point 
    #xpoint - x coordinate of point 
    #y - list of edges for bin in y direction e.g. [0,1,2,3] = 3 bins, bin 0 = [0,1), bin1 = [1,2) etc. 
    #x - list of edges for bin in x direction 
    #y0 - lowest point in y bin
    #x0 - lowest point in x bin 
    #yint - length of bins in y axis
    #xint - length of bins in x axis 
    
    curybin = math.floor((ypoint-y0)/yint)
    curxbin = math.floor((xpoint-x0)/xint)
    
    if curxbin >= len(x)-1:
        curxbin = len(x)-2
    elif curxbin < 0: 
        curxbin = 0
    if curybin >= len(y)-1:
        curybin = len(y) -2
    elif curybin < 0:
        curybin = 0
        
    return curxbin, curybin

def selectvehID(data,times,x,lane,veh,vertlist,platoon=None, vert=0):
    """
    This function is very disorganized and there are a few bugs with how things are initialized 
    """
    #after selecting a region in selectoscillation, now we want to go through the inidividual trajectories so we can see 
    #exactly where the congestion starts (which vehicle, what cause) as well as examine the congestion propagation in detail. 
    
    #data - data in raw form 
    #times, x lane, veh - these are all outputs from selectoscillation. 
    #vertlist - list of verticies of region, each vertice is a tuple and there should usually be 4 corners. 
    #platoon = None - can either automatically get intial vehicles or you can manually specify which vehicles should be in the initialization
    #vert = 0 -You can choose which vertice to start centered on. 
    
    
    #outputs - 
    #plot with 4 subplots, shows the space-time, speed-time, std. dev of speed, wavelet series of vehicles. 
    #interactive plot can add vehicles before/after, select specific vehicles etc. 
    
    #nice feature to add is pass in platoon and it will plot it but this requires some rework of the initialization; just need to initialize pre, post and then call 
    #preposthelper, don't need to do other stuff during initialization. 
    
    #note that you can use a custom picker to make it so you can pick exactly one artist per mouse click by checking the mouse coordinates, 
    #then looking at the distance of the mouse to each artist, then only returning true for the closest artist. 
    #alternatively, you can use built in picker=float option and then the artist will only be selected if mouse is within float points of the artist. 
    #but this can mean you can select multiple artists per click if they are close together. 

    #can just pass in meas and platooninfo if desired - make kwarg 
    
    #need meas and only data was passed in 
    meas,platooninfo = makeplatoonlist(data,form_platoons=False)
    
    
    
    if platoon is not None: 
        newvehlist = platoon
        xvert = []; yvert = []
        
    else:
        t0 = times[0]
        x0 = x[0] 
        timeint = times[1]-times[0]
        xint = x[1]-x[0]
        
        curvert = vertlist[math.floor(vert/2)] #first area
        
        xvert = []; yvert = []
        for i in curvert: 
            xvert.append(i[0]); yvert.append(i[1])
        xvert.append(curvert[0][0]); yvert.append(curvert[0][1])
        
        curvertind = -2*(vert % 2)
        ypoint,xpoint = curvert[curvertind][1], curvert[curvertind][0] #top left corner of first area
        timebin, xbin = point2bin(xpoint,ypoint,times,x,t0,x0,timeint,xint)
        
        newvehlist = veh[timebin][xbin]; newvehlist2 = [] 
    
    
    
    ###########
#    #initialize 
#    prevehlist = [] #list for previous vehicles 
#    for i in curvehlist: #candidate vehicles to add
#        curleadlist = platooninfo[i][4]
#        prevehlist.extend(curleadlist) #add all leaders
#        for j in curleadlist: 
#            prevehlist.extend(platooninfo[j][-1][1]) #add as well followers of leaders
#        
#    #remove duplicates, vehicles already shown, vehicles not in target lane 
#    prevehlist = prehelper(prevehlist, curvehlist, lane, meas, platooninfo)
    
    #basically we want a list of vehicles which will be added when we add vehicles before, and we need a list of vehicles which will be added when we add after. 
    #also there should be some initial vehicles which populate the plots. 
    #initialize post and pre vehicle lists
    prevehlist, curvehlist,unused = prelisthelper([],[],newvehlist,[],lane,meas,platooninfo)
    postvehlist, curvehlist, unused = postlisthelper([],[],newvehlist,[],lane,meas,platooninfo)
    #initial quick sort of curvehlist just so we know which one is the earliest vehicle
    curvehlist = sorted(list(curvehlist), key = lambda veh: platooninfo[veh][0]) #sort list depending on when they enter
    
    
    if False: #So i'm not sure why this is here to begin with but it was messing things up in at least 1 instance so I have removed it. 
        #I believe the point of this first part is to make it so the initialization contains a good amount of vehicles, but really there is no point in that. 
        #as long as the initialization of curvehlist, prevehlist, postvehlist are correct, can make curvehlist bigger by adding entries in pre/post to it. 
        
        #this region had a messed up initialization if this part is allowed to execute
        #test2 = [[(5562.474476963611, 1476.8050669428), (6311.045414797408, 164.0527611552), (7203.064516129032, 164.0527611552), (6454.493578295235, 1476.8050669428)]] 
        #in lane 2 of the NGSim i-80 reconstructed data. 
        #problem had to do with followers of leaders being added, but one of the followers of leaders was very far away from other vehicles, and this 
        #messes up the distinction between pre, current, and post vehicles. Specifically with vehicle 1717 having leader 1757, 1748 is a follower of 1757. 
        for ind, i in enumerate(prevehlist):  #iterate over prevehlist; everything after the earliest vehicle will be automatically put in 
            if platooninfo[i][0] < platooninfo[curvehlist[0]][0]: #if entry time is earlier than first vehicle in curvehlist
                continue
            else: 
                newvehlist = prevehlist[ind:]
                prevehlist = prevehlist[:ind]
                
                prevehlist, curvehlist, postvehlist = prelisthelper(prevehlist,curvehlist,newvehlist,postvehlist,lane,meas,platooninfo)
                break
        
        postvehlist.reverse() #reverse here 
        for ind, i in enumerate(postvehlist): #iterate over the postvehlist in the same way; everything before latest vehicle will be automatically put in 
            #need to iterate over postvehlist in reverse order
            if platooninfo[i][0] > platooninfo[curvehlist[-1]][0]: #if entry time is later than first vehicle (note used to be curvehlist[-1] last vehicle)
                continue
            else: 
                newvehlist2 = postvehlist[ind:]
                postvehlist = postvehlist[:ind]
                postvehlist.reverse() #reverse back here
                
    #            #check for duplicates #deprecated now we recognize overlap in newvehlist and newvehlist2 as a special case
    #            temp = []
    #            for j in newvehlist: 
    #                if j in curvehlist: 
    #                    continue
    #                else:
    #                    temp.append(j)
    #            newvehlist = temp
                
                postvehlist, curvehlist, prevehlist = postlisthelper(postvehlist,curvehlist,newvehlist2,prevehlist, lane,meas,platooninfo) 
                break

    
    ##################this does not work 
#    overlaplist = []
#    for i in newvehlist.copy():
#        if i in newvehlist2.copy():
#            newvehlist.remove(i)
#            newvehlist2.remove(i)
#            overlaplist.append(i)  
#    #put in unique values
#    prevehlist, curvehlist = prelisthelper(prevehlist,curvehlist,newvehlist,lane,meas,platooninfo)
#    postvehlist, curvehlist = postlisthelper(postvehlist,curvehlist,newvehlist2,lane,meas,platooninfo)  
#    temp = curvehlist.copy() #now put in overlapping values
#    prevehlist, temp = prelisthelper(prevehlist,temp,overlaplist,lane,meas,platooninfo)
#    postvehlist, curvehlist = postlisthelper(postvehlist,curvehlist,overlaplist,lane,meas,platooninfo)  
    
    curvehlist = sorted(list(curvehlist), key = lambda veh: platooninfo[veh][0]) #quick sort list depending on when they enter
    curvehlist = sortveh3(curvehlist,lane,meas,platooninfo) #full sort takes into account lane changing
    
    prevehlist, postvehlist = preposthelper(prevehlist,postvehlist,curvehlist,platooninfo)
    
    fig = plt.figure()
    ax1 = plt.subplot(2,2,1)
    ax2 = plt.subplot(2,2,2)
    ax3 = plt.subplot(2,2,3)
    ax4 = plt.subplot(2,2,4)
#    axlist = [ax1,ax2,ax3,ax4] 
    axlist = [ax1,ax4]
    
    
    scale = np.arange(15,100) #for wavelet transform
    indcounter = np.asarray([],dtype = np.int64) #keeps track of which line2D artists correspond to which vehicles (for axis 1); 
    indcounter4 = np.asarray([],dtype = np.int64) #same thing for axis 4
    counter = 0
    #e.g. indcounter = [0,0,0,1,1,1] #first 3 artists of ax.lines correspond to curvehlist[0], next three correspond to curvehlist[1]
    

    
#    def plothelper2(veh,meas,platooninfo):
#        #old helper function for plotting; does the plotting for veh
#        nonlocal indcounter
#        x = meas[veh][:,1]
#        y1 = meas[veh][:,2]
#        y2 = meas[veh][:,3]
#        ax1.plot(x,y1,'C0-',picker=5)
#        ax2.plot(x,y2-indcounter*50,'C0-',picker=5)
#        energy = wt(y2,scale)
#        ax3.plot(x,energy-indcounter*1e4,'C0-',picker=5)
#        spdstd = np.std(y2)
#        ax4.plot(indcounter,spdstd,'C0.',picker=5)
#        
##        wtstd = np.std(energy)
##        ax42 = ax4.twinx()
##        ax42.plot(indcounter,wtstd,'k.')
#        indcounter += 1
#        return
    
    def plothelper(veh,meas,platooninfo,vehind):
        #new helper function will do plotting for 
        nonlocal indcounter
        nonlocal indcounter4
        nonlocal counter
        LCind = np.diff(meas[veh][:,7])
        LCind = np.nonzero(LCind)[0]+1; LCind = list(LCind) #indices where lane changes
        LCind.insert(0,0); LCind.append(len(meas[veh][:,7])) #append first and last index so we have all the intervals 
        
        x = meas[veh][:,1]
        y1 = meas[veh][:,2]
        y2 = meas[veh][:,3]
        spdstd = np.asarray([])
        
        for i in range(len(LCind)-1):
            kwargs = {}
            if meas[veh][LCind[i],7] != lane: 
                kwargs = {'linestyle': '--','alpha':.4}
            spdstd = np.append(spdstd,y2[LCind[i]:LCind[i+1]])
            ax1.plot(x[LCind[i]:LCind[i+1]], y1[LCind[i]:LCind[i+1]], 'C0',picker=5,**kwargs)
            indcounter = np.append(indcounter,vehind)
        
        
        
#        ax1.plot(x,y1,'C0-',picker=5)

        spdstd = np.std(spdstd)
        if vehind < len(ax4.lines): #something behind
            counter += -1
            ax4.plot(counter,spdstd,'C0.',picker=5)
        else:
            ax4.plot(vehind+counter,spdstd,'C0.',picker=5)
        indcounter4 = np.append(indcounter4,vehind)
            
        plot2helper(veh,meas,platooninfo)
        
#        wtstd = np.std(energy)
#        ax42 = ax4.twinx()
#        ax42.plot(indcounter,wtstd,'k.')
        return
    
    def plot2helper(veh, meas, platooninfo):
        ax2.cla(); ax3.cla()
        
        LCind = np.diff(meas[veh][:,7])
        LCind = np.nonzero(LCind)[0]+1; LCind = list(LCind) #indices where lane changes
        LCind.insert(0,0); LCind.append(len(meas[veh][:,7])) #append first and last index so we have all the intervals 
        x = meas[veh][:,1]
        y2 = meas[veh][:,3]
        energy = wt(y2,scale)
        
        for i in range(len(LCind)-1):
            kwargs = {}
            if meas[veh][LCind[i],7] != lane: 
                kwargs = {'linestyle': '--','alpha':.4}
            ax2.plot(x[LCind[i]:LCind[i+1]], y2[LCind[i]:LCind[i+1]], 'C0',picker=5,**kwargs)
            ax3.plot(x[LCind[i]:LCind[i+1]], energy[LCind[i]:LCind[i+1]], 'C0',picker=5,**kwargs)
        
        
        return
    
    for count,i in enumerate(curvehlist): 
        plothelper(i,meas,platooninfo,count)
        
    zoomind = None
    indlist = []
    indlist4 = None
#    ax2ylim, ax2xlim = ax2.get_ylim, ax2.get_xlim
#    ax3ylim, ax3xlim = ax3.get_ylim, ax3.get_xlim
    plt.suptitle('Left click on trajectory to select vehicle')

#    indkey = np.array(range(len(curvehlist))) #indkey is list of the order of vehicles in curvehlist  #deprecated
    
    ax1.plot(xvert,yvert,'k-', scalex=False, scaley=False, alpha=.4)  #draw box for trajectories  
#    autoscale_based_on(ax1,ax1.lines[:-1])
    
    def on_pick(event):
        nonlocal zoomind
        nonlocal indlist
        nonlocal indlist4
        
        ax = event.artist.axes; 
        curind = ax.lines.index(event.artist)
        
        
        if event.mouseevent.button == 3: #right click zooms in on wavelet and speed series
            pass 
        else: #left click selects the trajectory to begin/end with 
#            print(indcounter)
            if indlist4 != None:
                ax4.lines[indlist4].set_color('C0')
            if ax == ax1: 
                curvehind = indcounter[curind]
                for ind,j in enumerate(indcounter4):
                    if j ==curvehind: 
                        ax4.lines[ind].set_color('C1')
                        break
                indlist4 = ind
            if ax==ax4:
                curvehind = indcounter4[curind]
                event.artist.set_color('C1')
                indlist4 = curind
                
            for i in indlist: #reset old selection for ax4
                ax1.lines[i].set_color('C0')
            
                
                
            for ind,j in enumerate(indcounter): 
                if j ==curvehind:
                    indlist.append(ind)
                    ax1.lines[ind].set_color('C1')
                
#            for i in axlist:
#                if selectind != None: 
#                    i.lines[selectind].set_color('C0')
#                i.lines[curind].set_color('C1')
                
            veh = curvehlist[curvehind]
            plt.suptitle('Vehicle ID '+str(veh)+' selected')
            plot2helper(veh,meas,platooninfo)        
            
            fig.canvas.draw()
            return
        
    def key_press(event):
        nonlocal curvehlist
        nonlocal prevehlist
        nonlocal postvehlist
        nonlocal indcounter 
        nonlocal indcounter4
        #toggle opaque trajectories 
        if event.key in ['T','t']:
            first = True
            visbool = None
            ax = event.inaxes
            for i in ax.lines: 
                if i.get_alpha() != None: 
                    if first: 
                        visbool = i.get_visible()
                        visbool = not visbool
                        first = False
                    i.set_visible(visbool)
            fig.canvas.draw()
        
        
        if event.key in ['P','p']: #print current vehicle list 
#            print(prevehlist)
#            print(postvehlist)
            print('current vehicles shown are '+str(curvehlist))
        if event.key in ['A','a']: #add a vehicle before 

            count = -1
            while prevehlist[count] in curvehlist: #prevent a vehicle already in from being added 
                count += -1
            ax1.lines[-1].remove() #for study area to be shown
            ax1.relim()
            
            indcounter = indcounter + 1 
            indcounter4 = indcounter4 + 1
            plothelper(prevehlist[count],meas,platooninfo,0)
            prevehlist, curvehlist,postvehlist = prelisthelper(prevehlist[:count],curvehlist,[prevehlist[count]],postvehlist,lane,meas,platooninfo)
            
            ax1.plot(xvert,yvert,'k-', scalex=False, scaley=False, alpha=.4) 
            plt.draw()
            
#            print(indcounter)
            
            
        if event.key in ['Z','z']: #add a vehicle after

            
            
            count = 0
            while postvehlist[count] in curvehlist: #prevent a vehicle already in from being added 
                count += 1
            ax1.lines[-1].remove() #for study area to be shown
            ax1.relim()
            
            plothelper(postvehlist[count],meas,platooninfo,len(curvehlist))
            postvehlist, curvehlist,prevehlist = postlisthelper(postvehlist[count+1:],curvehlist,[postvehlist[count]],prevehlist,lane,meas,platooninfo)
            
            ax1.plot(xvert,yvert,'k-', scalex=False, scaley=False, alpha=.4)  #redraw study area
            plt.draw()
              
        if event.key in ['N','n']: #next study area 
            plt.close()
            selectvehID(data,times,x,lane,veh,vertlist,vert=vert+1)
            

        return
    
    fig.canvas.callbacks.connect('pick_event', on_pick)
    plt.connect('key_press_event',key_press)
        
    return 

def preposthelper(prevehlist, postvehlist, curvehlist, platooninfo):
    #takes curvehlist - current vehicles 
    #postvehlist - vehicles that are potentially to be added as next vehicles in platoon (in lane)
    #prevehlist - vehicles that are potentially be added as previous vehicles in platoon (in lane)
    #platooninfo
    
    #looks at pre and post lists, and put all vehicles in pre that occur after Tn in post, and all vehicles in post that occur before Tn in pre
    #this prevents vehicles which look like they should be in post being in pre and vice versa. Useful during initialization. 
    
    tn = platooninfo[curvehlist[0]][0]
    Tn = platooninfo[curvehlist[-1]][0]
    pretemp = reversed(prevehlist)
    posttemp = postvehlist.copy()
    prechange = False
    postchange = False
    prenew = []
    postnew = []
    
    for i in pretemp: #iterate over pre vehicles backwards
        if platooninfo[i][0] > Tn: #if they enter later than Tn we want to move them to post
            postchange = True #need to resort post
            postnew.append(i)
            prevehlist = prevehlist[:-1]
        else: 
            break
    
    for i in posttemp:
        if platooninfo[i][0] < Tn:
            prechange = True
            prenew.append(i)
            postvehlist = postvehlist[1:]
        else:
            break
    #update lists
    if prechange: 
        for i in prenew: 
            if i not in prevehlist: 
                prevehlist.append(i)
        prevehlist = sorted(prevehlist, key = lambda veh: platooninfo[veh][0])
    if postchange:
        for i in postnew:
            if i not in postvehlist: 
                postvehlist.append(i)
        postvehlist = sorted(postvehlist, key = lambda veh: platooninfo[veh][0])
        
        
    return prevehlist, postvehlist
            
    

def prelisthelper(prevehlist, curvehlist, newvehlist, postvehlist, lane, meas, platooninfo, check=True):
    #given a list of new vehicles added, get all of the candidate pre vehicles added, add new vehicles to the current veh list, and call prehelper on the prevehlist
    
#    print('new vehicles are '+str(newvehlist))
    for i in newvehlist:
        if i in postvehlist and check: #if new vehicle to be added is also in postvehlist we need to update that list as well as the prevehlist
            temp = curvehlist.copy()
            postvehlist, temp, temp = postlisthelper(postvehlist,temp,[i],prevehlist,lane,meas,platooninfo,check = False) #update postvehlist
        
        curvehlist.insert(0,i)
        curleadlist = platooninfo[i][4]
        prevehlist.extend(curleadlist)
        for j in curleadlist: 
            prevehlist.extend(platooninfo[j][-1][1])
            
#    print('all candidates are '+str(prevehlist))
    prevehlist = prehelper(prevehlist,curvehlist,lane,meas,platooninfo)
#    print('filtered candidates are '+str(prevehlist))
    
    return prevehlist,curvehlist, postvehlist

def postlisthelper(postvehlist, curvehlist, newvehlist, prevehlist, lane, meas, platooninfo, check = True):
    #given a list of new vehicles added, get all of the candidate pre vehicles added, add new vehicles to the current veh list, and call prehelper on the prevehlist
    
#    print('new vehicles are '+str(newvehlist))
    for i in newvehlist:
        if i in prevehlist and check: 
            temp = curvehlist.copy()
            prevehlist,temp,temp = prelisthelper(prevehlist,curvehlist,[i],postvehlist,lane,meas,platooninfo,check = False)
        curvehlist.append(i)
        curfollist = platooninfo[i][-1][1]
        postvehlist.extend(curfollist)
        for j in curfollist: 
            postvehlist.extend(platooninfo[j][4])
            
#    print('all candidates are '+str(prevehlist))
    postvehlist = prehelper(postvehlist,curvehlist,lane,meas,platooninfo)
#    print('filtered candidates are '+str(prevehlist))
    
    return postvehlist,curvehlist, prevehlist
    

def prehelper(prevehlist, curvehlist, lane, meas,platooninfo):
    #take prevehlist of candidate vehicles to previously add. 
    #return prevehlist which is a sorted list of order vehicles should be added in .
    #it's called prehelper but actually it works for the postvehlist as well. 
    
    prevehlist = list(set(prevehlist)) #remove duplicates
    temp = []
    for i in prevehlist: 
        if i in curvehlist: 
            continue
        if lane not in np.unique(meas[i][:,7]):
            continue
        temp.append(i)
    prevehlist = temp
    prevehlist = sorted(list(prevehlist), key = lambda veh: platooninfo[veh][0])
    
    return prevehlist 

def autoscale_based_on(ax, lines): #doesn't work right see selectvehID you can remove line reset window redraw line this is the solution we're using 
    ax.dataLim = mtransforms.Bbox.unit()
    for line in lines:
        xy = np.vstack(line.get_data()).T
        ax.dataLim.update_from_data_xy(xy, ignore=False)
    ax.autoscale_view()


def sortveh3(vehlist,lane,meas,platooninfo):
    #third attempt at a platoon ordering algorithm 
    #this one is more hueristic based but should work pretty much always 
    """
    main way to improve this functions 
    
    -come up with some clever way to sort vehicles. I've thought about it for a long time and this is pretty much the best I came up with. 
    
    -main way to improve speed is that when you are going over the leftover, you should check if you can add stuff to vehfollist, and potentially use the sortveh_disthelper again. 
    because the way we are adding vehicles in the leftover is very very slow (n^2) and the way sortveh_disthelper does it looks like (n), so if a bunch of stuff ends up in leftover
    it can potentially kill your performance. so if you need to sort stuff where alot of the vehicles end up in leftover you probably want to make that modification. 
    """
    
    #initialization 
    vehlist = sorted(list(vehlist), key = lambda veh: platooninfo[veh][0]) #initial guess of order
    out = [vehlist[0]] #initialize output 
    
    end = False #keep iterating while end is False
    curveh = out[0] #the first vehicle. 
    
    #get vehfollist, list of following vehicles which we will add.
    temp = meas[curveh]
    temp = temp[temp[:,7]==lane]
    vehfollist = np.unique(temp[:,5])
    temp = []
    for i in vehfollist: 
        if i == 0 : 
            continue
        if i in vehlist: 
            temp.append(i)
    vehfollist = temp
    
    #the way this works is that we keep adding things to vehfollist and they are sorted as they are added to the output. 
    #So we keep adding followers and followers of followers. This won't necessarily add all vehicles though, because we can also have leaders of followers, 
    #and also because we don't know the order in the beginning we don't necessarily begin with the first vehicle, so there may be leaders of the initial curveh. 
    
    count = 0 
    while not end: 
        count += 1 
        #
        if len(out)==len(vehlist): # if everything is added to out it means that everything is done
            end = True
            
        elif len(vehfollist) == 0: #if there are no followers to add but there are still vehicles we need to order then we need to do something about the rest of
            #the vehicles 
            #leftover has all vehicles which haven't been added/sorted yet. 
            leftover = vehlist.copy()
            for i in out: 
                leftover.remove(i)
                
            leftover = sortveh3(leftover,lane,meas,platooninfo) #order the leftover 
            leftover.reverse() #iterate over it in reverse 
            
            for i in leftover: #if we have leftover, we will go through and add each vehicle one at a time. 
                #first check if we can trivially add the vehicle to either the beginning or end. 
                platoont_nstar = platooninfo[out[0]][0] #first time in the platoon 
                platoonT_n = platooninfo[out[-1]][3] #last time in the platoon 
                if platooninfo[i][3] < platoont_nstar: 
#                    newout = out.copy()
#                    newout.insert(0,i)
                    out.insert(0,i)
                    continue
                if platooninfo[i][0] > platoonT_n:
#                    newout = out.copy()
#                    newout.append(i)
                    out.append(i)
                    continue 
                    
                #now we will go through each vehicle in the platoon and measure distance from the vehicle to the leftover. The very first vehicle with a positive value
                #must be the vehicle which is directly after the leftover. 
                leftovermeas = meas[i]
                leftovermeas = leftovermeas[leftovermeas[:,7]==lane]
#                times = leftovermeas[:,1] #times the vehicle is in the target lane 
                leftovert_nstar = platooninfo[i][0] #t_nstar for the leftover
                
                count2 = 0 #keep track of which vehicle we are currently on 
                case = False
                for j in out: #now we need to iterate over each vehicle in the platoon to get the distance for the leftover. 
                    curmeas = meas[j]
                    curmeas = curmeas[curmeas[:,7]==lane]
#                    curmeas = curmeas[curmeas[:,1]==times] #correct times #bug here 
                    curleftovermeas, curmeas = overlaphelp(leftovermeas,curmeas)
                    if len(curmeas) >0: #assuming we have any sort of overlap we can compute the distance
#                        timeinds = curmeas[:,1]-leftovert_nstar
#                        curleftovermeas = leftovermeas[timeinds]
                        curdist = np.mean(curleftovermeas[:,2]-curmeas[:,2]) #get dist for the current vehicle j 
                        if curdist > 0: #you are directly behind the current vehicle 
                            newout = copy.deepcopy(out) #need to create copy becuase for loop goes over out
                            newout.insert(count2,i)
                            break #exit loop
                        else: 
                            lastgood = count2
                            case = True
                    count2 += 1
                else: #if we can't put it in front of any of the vehicles then we can safely add it to the end, or we can put it after the last good vehicle we got 
                    if case: 
                        newout = copy.deepcopy(out)
                        newout.insert(lastgood+1,i)
                    else:
                        count2 = 0
                        lasttime = leftovermeas[-1,1]
                        for j in out: #can potentially get put in the middle in special case 
                            curmeas = meas[j]
                            curmeas = curmeas[curmeas[:,7]==lane]
                            t_n = curmeas[0,1]
                            if lasttime < t_n: 
                                newout = copy.deepcopy(out)
                                newout.insert(count2,i)
                        else: #put at end 
                            newout = copy.deepcopy(out)
                            newout.append(i)
                    #note that there is still one edge case we don't handle, where there is no overlap at all, but it doesn't actually
                    #go to the end, it's just kind of snuck in there
                    
                out = newout
                #improvement would be after you successfully add i, see if you can add to vehfollist, and if you can then you can use sortveh_disthelper again. 

            ################
            #deprecated  -  we always just want to call the sortveh_disthelper unless we don't have anything to add. 
#        elif len(vehfollist)==1: #just 1 follower can just add it don't need to get any special ordering. 
#            out.append(vehfollist[0])
            ###################
            
        else: #get order among followers, this should only be called once, and then we should either go to the if and immediately terminate or go to the elif
            #note that this is pretty slow to only call this once potentially, since we don't like adding leftovers. So if you wanted to optimize this you could 
            #make it so if you add a leftover which has followers, you can then use sortveh_disthelper again, add everything 
            if count > 1: #count should be 1 and it should never be any other value. 
                print('bug')
            out = sortveh_disthelper(out, curveh, vehfollist, vehlist, lane, meas, platooninfo)
            vehfollist = []
            #old notes 
            #possibly sortveh2 is the better approach. see notes we can have lists of vehicles you follow to try to get ordering, need to check for times we get multiple on same
            #level, and check for circular dependency. Can resolve these using average distance between follower and leader as discussed below.
            #for after having added multiple followers need a way to dealing with possibly getting multiple followers again. maybe just use last trajectory and do some thing
            #with the ranking. 
            
            #thinking its possible to order this by looking at distance between followers and their leader. then order by the average distance. 
            #edge cases of vehicle which does not follow anything else in platoon. these can be moved in front of their follower. if there are two such vehicles 
            #and they have same follower can do distance compared to the follower. 
            #after adding multiple followers, need to add all followers of followers, so may have multiple followers to add again. 
            #if this happens, compare everything to the leader which origianlly gave multiple followers. keep adding followers until we get to case where there is only 
            #1 vehicle to logically add next. Note that trajectories can be extended using np.diff or similar if needed. 
            pass
            
            
    return out

def overlaphelp(meas1, meas2):
    #for two vehicles meas1 and meas2, computes the overlap between them and returns the data which can be directly compared; this is useful for various things 
    #this is meant to be used when you have somethign like meas[meas[:,7] == lane] and meas2[meas2[:,7] == lane] and then you want all the times where there is overlap. 
    #uses functions sequential and indjumps from calibration 
    
    #get indices of the sequential data 
    ind1 = helper.sequential(meas1)
    ind2 = helper.sequential(meas2)
    
    #change the indices into times 
    times1 = helper.indtotimes(ind1,meas1)
    times2 = helper.indtotimes(ind2,meas2)
    #booking keeping, output will be the parts of meas we can use to compute the distance 
    outtimes = []
    outind1 = []
    outind2 = []
    #need to keep track of where we are 
    count1 = 0 
    count2 = 0
    while count1 < len(times1): #iterate over the first meas
        cur1 = times1[count1]
        while count2 < len(times2): #iterate over the second meas 
            cur2 = times2[count2]
            if cur2[0] < cur1[0] and cur2[1] < cur1[0]: #trivial case, check next 2 block 
                count2 += 1
                continue
            elif cur2[0] > cur1[1] and cur2[1] > cur1[1]: #other trivial case, done checking 2 blocks and check next 1 block 
                count2 = 0
                count1 += 1
                break 
            elif cur2[0] <= cur1[0] and cur2[1] >= cur1[0] and cur2[1] <= cur1[1]:
                curtimes = (cur1[0], cur2[1])
                #code to convert the times back into indices
                outtimes.append(curtimes)
                temp1 = (curtimes[0]-cur1[0]+cur1[2], curtimes[1]-cur1[0]+cur1[2])
                outind1.append(temp1)
                temp1 = (curtimes[0]-cur2[0]+cur2[2],curtimes[1]-cur2[0]+cur2[2])
                outind2.append(temp1)
                #update for next iteration 
                #check the next 2 block 
                count2 += 1 
                continue
            elif cur2[0] <= cur1[0] and cur2[1] >= cur1[1]:
                curtimes = (cur1[0],cur1[1])
                
                outtimes.append(curtimes)
                temp1 = (curtimes[0]-cur1[0]+cur1[2], curtimes[1]-cur1[0]+cur1[2])
                outind1.append(temp1)
                temp1 = (curtimes[0]-cur2[0]+cur2[2],curtimes[1]-cur2[0]+cur2[2])
                outind2.append(temp1)
                #check the next 1 block 
                count1 += 1 
                break
            elif cur1[0] <= cur2[0] and cur1[1] >= cur2[1]:
                curtimes = (cur2[0],cur2[1])
                
                outtimes.append(curtimes)
                temp1 = (curtimes[0]-cur1[0]+cur1[2], curtimes[1]-cur1[0]+cur1[2])
                outind1.append(temp1)
                temp1 = (curtimes[0]-cur2[0]+cur2[2],curtimes[1]-cur2[0]+cur2[2])
                outind2.append(temp1)
                #check the next 2 block 
                count2 += 1
                continue
            else: #cur1[0] < cur2[0] and cur1[1] < cur2[1]
                curtimes = (cur2[0], cur1[1])
                
                outtimes.append(curtimes)
                temp1 = (curtimes[0]-cur1[0]+cur1[2], curtimes[1]-cur1[0]+cur1[2])
                outind1.append(temp1)
                temp1 = (curtimes[0]-cur2[0]+cur2[2],curtimes[1]-cur2[0]+cur2[2])
                outind2.append(temp1)
                
                #check next 1 block
                count1 += 1
                break
        count1 += 1
            
    #now get the final data to return
    dim = np.shape(meas1)[1]
    out1 = np.zeros((0,dim))
    for i in outind1:
        out1 = np.append(out1,meas1[int(i[0]):int(i[1])],axis=0)
    out2 = np.zeros((0,dim))
    for i in outind2:
        out2 = np.append(out2,meas2[int(i[0]):int(i[1])],axis=0)
    
    return out1, out2
    

def sortveh_disthelper(out, curveh, vehfollist,vehlist, lane, meas, platooninfo):
    #note that this can still potentially fail in some special cases:
    #   say you have two followers, one is directly behind you, and then the other followers behind the person behind you. The first follower is only there for a short time, 
    # and then they change lanes. Then the person behind them gets very close to you. This algorithm will put the 2nd follower first, when they should be second, because we are 
    #computing the average distances to the followers. 
    
    #circular dependencies will always have the offending vehicle added in the beginning of the loop. 
    
    #this can fail to give a correct order in some weird cases, where we can't compare all followers, and we also can't compare all followers to a single leader. 
    
    #so some edge cases potentially wrong but overall I think this should be pretty robust. 
    
    #in this rewritten version of the original function, we don't assign distance scores, instead we get all the followers, sort them and add them, and then repeat. So you can never have a follower
    #which is ahead of you. Any circular dependencies will result in 
    
    #curveh - vehicles are sorted relative to this vehicle 
    
    #given an initial vehicle and list of vehicles to add, gets the distance of each trajectory from the initial vehicle, and then 
    #sorts all vehicles based on this distance. 
    distlist = {} #initialized as 0 because the curveh is 0 distance from itself. 
    curveht_nstar = platooninfo[curveh][0]
    out = [curveh]
    
    newfolveh = set()
    curmeas = meas[curveh]
    curmeas = curmeas[curmeas[:,7]==lane]
    for i in vehfollist: 
        #first need to get times when vehicle is following the curveh. 
        temp = meas[i]
        temp = temp[temp[:,7]==lane]
#        print(curveh)
#        print(i)
        lead,fol = overlaphelp(curmeas,temp) #something is wrong with overlaphelp it seems like its not working properly 
        
        curdist = np.mean(lead[:,2] - fol[:,2])
        distlist[i] = curdist #
        
        curfolveh = np.unique(temp[:,5])
#        curfolveh = platooninfo[i][-1][1] #here 
        for j in curfolveh: 
            if j == 0:
                continue 
            if j in vehlist: 
                if j not in vehfollist and j not in out:
                    newfolveh.add(j)
    
    out2 = list(distlist.keys())
    out2 = sorted(out2, key = lambda veh: distlist[veh])
    curlen = len(newfolveh)
    while curlen>0:
        #check if all new followers can be sorted according to a single follower 
        for i in newfolveh: 
            curmeas = meas[i]
            curmeas = curmeas[curmeas[:,7]==lane]
            distlist = {i:0}
            for j in newfolveh: 
                if j==i:
                    continue
                temp = meas[j]
                temp = temp[temp[:,7]==lane]
                
                lead,fol = overlaphelp(curmeas,temp)
                if len(lead) == 0: #failed for vehicle j, so try for other vehicles 
                    break
                else: 
                    curdist = np.mean(lead[:,2]-fol[:,2])
                    distlist[j] = curdist 
            else: 
                #no break in j loop means we succeeded for vehicle i 
                out2 = distlist.keys()
                out2 = sorted(out2,key = lambda veh: distlist[veh]) #sort the new vehicles 
                
#                out.extend(out2) #add all the new vehicles. now we need to get all the new vehicles 
                #to prevent any orders being off by 1, we always check if the order makes sense wehn we add a new vehicle 
                for k in out2: 
                    lastveh = out[-1]
                    lastmeas = meas[lastveh]
                    lastmeas = lastmeas[lastmeas[:,7]==lane]
                    temp = meas[k]
                    temp = temp[temp[:,7]==lane]
                    last, cur = overlaphelp(lastmeas,temp)
                    if len(last) == 0: #nbothing to check 
                        out.append(k)
                    elif np.mean(last[:,2]-cur[:,2]) > 0: #if this is positive it means the order is right 
                        out.append(k)
                    else: 
                        out.insert(-1,k) #it's before the last one in this case. 
                
                oldnewfolveh = newfolveh.copy() #copy of old we will iterate over this and make the new
                newfolveh = set()
                for k in oldnewfolveh: 
                    temp = meas[k]
                    temp = temp[temp[:,7]==lane]
                    curfolveh = np.unique(temp[:,5])
        #            curfolveh = platooninfo[i][-1][1] #and here 
                    for l in curfolveh: 
                        if l == 0:
                            continue 
                        if l in vehlist: #has to be a vehicle we are sorting 
                            if l not in out: #can't be a vehicle already sorted
                                newfolveh.add(l)
                curlen = len(newfolveh)
                break #this is beraking the first loop over i 
        else: #this else is with the first loop over i 
        #this means we weren't able to sort the new vehicles with respect to each other, so we'll try to sort them with respect to one of the leaders already added. 
            #first we have to figure out what leaders we can potentially use 
            for i in range(len(out)): #we will go backwards over everything in out 
                
                curveh = out[int(-(i+1))] #go backwards 
                curmeas = meas[curveh]
                curmeas = curmeas[curmeas[:,7]==lane]
                lasttime = curmeas[-1,1]
                distlist = {}
                
                for j in newfolveh:
                    temp = meas[j]
                    temp = temp[temp[:,7]==lane]
                    if temp[0,1] > lasttime:  #simple check if failed to potentially save a lot of time 
                        break
                    
                    lead,fol = overlaphelp(curmeas,temp)
                    if len(lead) == 0:  #failed 
                        break
                    else: 
                        curdist = np.mean(lead[:,2] - fol[:,2])
                        distlist[j] = curdist
                else: #succeeded
                    out2 = distlist.keys()
                    out2 = sorted(out2,key = lambda veh: distlist[veh]) #sort the new vehicles 
                    
#                    out.extend(out2) #add all the new vehicles. now we need to get all the new vehicles 
                    for k in out2: 
                        lastveh = out[-1]
                        lastmeas = meas[lastveh]
                        lastmeas = lastmeas[lastmeas[:,7]==lane]
                        temp = meas[k]
                        temp = temp[temp[:,7]==lane]
                        last, cur = overlaphelp(lastmeas,temp)
                        if len(last) == 0: #nbothing to check 
                            out.append(k)
                        elif np.mean(last[:,2]-cur[:,2]) > 0: #if this is positive it means the order is right 
                            out.append(k)
                        else: 
                            out.insert(-1,k) #it's before the last one in this case. 
                    
                    oldnewfolveh = newfolveh.copy() #copy of old we will iterate over this and make the new
                    newfolveh = set()
                    for k in oldnewfolveh: 
                        temp = meas[k]
                        temp = temp[temp[:,7]==lane]
                        curfolveh = np.unique(temp[:,5])
            #            curfolveh = platooninfo[i][-1][1] #and here 
                        for l in curfolveh: 
                            if l == 0:
                                continue 
                            if l in vehlist: #has to be a vehicle we are sorting 
                                if l not in out: #can't be a vehicle already sorted
                                    newfolveh.add(l)
                    curlen = len(newfolveh)
                    break #this is beraking the first loop over i 
                    
            else: #this is attached to the first loop over i, it means we have no idea how to sort the vehicles in newfolveh. 
                #in this case what we'll do is throw a warning, and not do anything with the vehicles, so they will get put into leftover. 
                print('we werent able to sort some vehicles, they will be handled as leftovers') #this should only happen rarely for a dataset with congestion in it. 
                oldnewfolveh = newfolveh.copy() #copy of old we will iterate over this and make the new
                newfolveh = set()
                for k in oldnewfolveh: 
                    temp = meas[k]
                    temp = temp[temp[:,7]==lane]
                    curfolveh = np.unique(temp[:,5])
        #            curfolveh = platooninfo[i][-1][1] #and here 
                    for l in curfolveh: 
                        if l == 0:
                            continue 
                        if l in vehlist: #has to be a vehicle we are sorting 
                            if l not in out: #can't be a vehicle already sorted
                                newfolveh.add(l)
                curlen = len(newfolveh)
                

    
    return out

#######these used to be in the old simulation.py file
###nothing special but the hd has some code to draw arrows which is actually somewhat nice so you can see what direction the loops are going in 
def rediff(x, dt, end = True):
    #first order forward difference on x
    #input - array of values x 
    
    #output - array of numerically differentiated dx which is 1 length shorter 
    #if end is true, we will return the last value twice so it's the same length. 
    
    out = []
    for i in range(len(x)-1):
        new = (x[i+1]-x[i])/dt
        out.append(new)
        
    if end: 
        out.append(out[-1])
    return out 

def hd(lead,fol, arrowinterval = .5 ):
    N = len(lead.x) #number measurements 
    hd = []
    for i in range(N):
        cur = lead.x[i]-fol.x[i]-lead.len
        hd.append(cur)
        
    plt.plot(hd,fol.dx)
    

    # \\ TO DO \\ ##
    #a possible improvement here would be to record the length of the trjaecotry and then choose the arrowinterval and 
    #arrow size based on that length so that you don't have to manually specify the arrowinterval
        
    counter = 0
    arrlen = .1
    arroffset = 13*math.pi/16
    for i in range(N-1):
        dy = fol.dx[i+1]- fol.dx[i]
        dx = hd[i+1] - hd[i]
        counter = counter + (dy**2+dx**2)**.5 #keep track of length traveled 
        if counter > arrowinterval: #if its time to draw another arrow 
            counter = 0 #reset counter
            theta = math.atan2(dy,dx) #angle at which arrow will point 
            arr1dx = arrlen*math.cos(theta-arroffset)
            arr2dx = arrlen*math.cos(theta+arroffset)
            arr1dy = arrlen*math.sin(theta-arroffset)
            arr2dy = arrlen*math.sin(theta+arroffset)
            plt.plot([hd[i],hd[i]+arr1dx],[fol.dx[i],fol.dx[i]+arr1dy],'k-')
            plt.plot([hd[i],hd[i]+arr2dx],[fol.dx[i],fol.dx[i]+arr2dy],'k-')
    
    return hd 

def vehplot(universe, interval = 1):
    N = len(universe)
    plt.subplot(1,2,1)
    for i in range(0,N,interval):
        plt.plot(universe[i].x)
    plt.ylabel('space')
    plt.xlabel('time')
    plt.yticks([])
    plt.xticks([])
    plt.subplot(1,2,2)
    for i in range(0,N,interval):
        plt.plot(np.asarray(universe[i].dx)- 2*i)
    plt.ylabel('speed')
    plt.xlabel('time')
    plt.yticks([])
    plt.xticks([])
        
    return

def stdplot(universe,customx = None):
    N = len(universe)
    y = []
    for i in range(N):
        y.append(np.std(universe[i].dx))
    if customx == None: 
        plt.plot(y,'k.')
    else: 
        plt.plot(customx,y,'k.')
        
    return 
