"""
where all the plotting functions go

@author: rlk268@cornell.edu
"""

import numpy as np
import copy
import math
import bisect

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.animation as animation
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import pywt  # package for wavelet transforms in python
from matplotlib import cm
from statistics import harmonic_mean  # harmonic mean function
from matplotlib.widgets import RectangleSelector
from matplotlib.widgets import PolygonSelector
import matplotlib.transforms as mtransforms

import palettable

from .calibration import helper
from .calibration.helper import sequential, indtotimes
from havsim.calibration.algs import makeplatoonlist, sortveh3
from havsim.calibration.opt import r_constant


def optimalvelocity(s, p):
    V = p[0] * (np.tanh(p[1] * (s) - p[2] - p[4]) - np.tanh(-p[2]))
    return V


def optimalvelocityplot(p):
    s = np.linspace(0, 200, 2000)
    v = np.zeros(len(s))

    for i in range(len(v)):
        v[i] = optimalvelocity(s[i], p)

    plt.figure()
    plt.plot(s, v)
    plt.xlabel('headway (ft)')
    plt.ylabel('speed (ft/s)')
    # plt.savefig('ovm3.png')
    return


def checksort(vehlist, meas, lane):
    # very simple function plots trajectory in a line, allows you to debug the platoon order
    fig = plt.figure()
    for i in range(len(vehlist)):
        cur = meas[vehlist[i]]
        cur = cur[cur[:, 7] == lane]
        plt.plot(cur[:, 1], cur[:, 2], picker=5)

    def on_pick(event):
        ax = event.artist.axes
        curind = ax.lines.index(event.artist)  # artist index
        plt.title('selected vehicle ' + str(vehlist[curind]) + ' which has order ' + str(curind))
        plt.draw()

    fig.canvas.callbacks.connect('pick_event', on_pick)
    pass


def optplot(out, meas, sim, followerchain, platoonlist, model, modeladj, modeladjsys, makeleadfolfun, platoonobjfn,
            args,
            speed=False, newfig=True, clr=['C0', 'C1'], fulltraj=True, lane=None,opacity = .4):  # plot platoon in space-time
    # CURRENT DOCUMENTATION
    # out - results from optimization algorithm
    # meas - measurements in np array, rows are observations
    # sim - simulation in same format as meas. can pass in None and we will get the simulation from out. otherwise we will assume sim is loaded in correctly
    #
    # in different colors.
    # followerchain (platooninfo) - dictionary containing information on each vehicle ID
    # platoonlist - input the list of platoons used in the optimization
    # model
    # modeladj
    # modeladjsys
    # makeleadfolfun
    # platoonobjfn
    # args

    # speed = False - if True will plot the speed instead of position
    # newfig = True - if True will create a new figure, otherwise it will use the current figure
    # clr = 'C0', assuming Colors = False, clr will control what colors will be used. Default is ['C0','C1'] which are the default matplotlib colors
    # fulltraj = True - assuming Colors = False, if True, will plot the entire trajectory, if False, will plot only the simulation
    # lane = None - If passed in as a laneID, the parts of trajectories not in the lane ID given will be made opaque

    # this is meant to examine the output from using an optimization algorithm - visualize results and see which vehicles

    ind = 2

    if sim is None:
        sim = copy.deepcopy(meas)
        sim = helper.obj_helper(out[0], model, modeladj, modeladjsys, meas, sim, followerchain, platoonlist,
                                makeleadfolfun, platoonobjfn, args)

    artist2veh = []  # artist2veh takes artist index and converts it to vehicle ID index (for followerlist)
    veh2platoon = []  # takes veh ID index and converts to platoon index (for platoonlist)
    platoonrmse = out[2]  # convert platoon index to platoon rmse
    vehrmse = []  # converts veh ID index to rmse for vehicle

    # vehrmse
    for i in platoonlist:
        obj = helper.SEobj_pervehicle(meas, sim, followerchain,
                                      i)  # list of (individual) objective functions. This is needed for next step
        for j in range(len(i)):  # convert each individual objective to individual rmse
            temp = helper.convert_to_rmse(obj[j], followerchain, [i[j]])
            vehrmse.append(temp)

    #    indcounter = np.asarray([],dtype = int64) #keeps track of which artists correspond to which vehicle
    if speed:
        ind = 3
    if platoonlist != []:
        followerchain = helper.platoononly(followerchain, platoonlist)
    followerlist = list(followerchain.keys())  # list of vehicle ID

    for count, i in enumerate(platoonlist):
        for j in i:
            veh2platoon.append(count)

    if newfig:
        fig = plt.figure()

    counter = 0
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
                plt.plot(x[LCind[j]:LCind[j + 1]], y[LCind[j]:LCind[j + 1]], clr[0], picker=5, **kwargs)
                artist2veh.append(
                    counter)  # artist2veh takes artist index and converts it to vehicle ID index (for followerlist)



        else:
            plt.plot(x, y, clr[0])
            artist2veh.append(counter)
        counter += 1

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
                        kwargs = {'linestyle': '--', 'alpha': opacity}  # dashed line .4 opacity (60% see through)
                    plt.plot(x[LCind[j]:LCind[j + 1]], y[LCind[j]:LCind[j + 1]], clr[1], picker=5, **kwargs)
                    artist2veh.append(counter)
            else:
                plt.plot(x, y, clr[1])
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
    halfn = int(nartists / 2)

    def on_pick(event):
        nonlocal find_artists
        ax = event.artist.axes
        curind = ax.lines.index(event.artist)  # artist index

        if event.mouseevent.button == 1:  # left click selects vehicle
            # deselect old vehicle
            for j in find_artists:
                if j < nartists / 2:
                    ax.lines[j].set_color('C0')
                else:
                    ax.lines[j].set_color('C1')

            # select new vehicle
            vehind = artist2veh[curind]  # convert from artist to vehicle index
            platoonind = veh2platoon[vehind]
            find_artists = np.asarray(artist2veh)
            find_artists = np.nonzero(find_artists == vehind)[
                0]  # all artist indices which are associated with vehicle
            #            nartists = len(ax.lines)

            for j in find_artists:
                if j < nartists / 2:  # part of measurements because in first half
                    ax.lines[j].set_color('C2')
                else:
                    ax.lines[j].set_color('C3')
            plt.title('Vehicle ID ' + str(followerlist[vehind]) + ' has RMSE ' + str(
                round(vehrmse[vehind], 2)) + ' in platoon ' + str(platoonlist[platoonind]) + ' with RMSE ' + str(
                round(platoonrmse[platoonind], 2)))
            plt.draw()

        if event.mouseevent.button == 3:  # right click selects platoon
            #            print(find_artists)
            # deselect old vehicle
            for j in find_artists:
                if j < nartists / 2:
                    ax.lines[j].set_color('C0')
                else:
                    ax.lines[j].set_color('C1')

            # select new vehicle
            vehind = artist2veh[curind]  # convert from artist to vehicle index
            platoonind = veh2platoon[vehind]  # get platoonindex
            #            print(platoonind)

            platoonindlist = []  # we will get a list of the vehicle indices for all vehicles in the platoon
            countvehs = 0  # we first find how many vehicles come before the platoon
            for i in range(platoonind):
                countvehs += len(platoonlist[i][1:])
            for i in range(len(platoonlist[platoonind][1:])):
                platoonindlist.append(countvehs + i)

            find_artists = []  # now find all artists which are associated with platoon
            for count, i in enumerate(ax.lines):  # get artist index
                if artist2veh[
                    count] in platoonindlist:  # convert artist index to vehicle index, check if vehicle index is included in platoon, if it is we keep track of it
                    find_artists.append(count)

            #            print(find_artists)
            # select new vehicles
            for j in find_artists:
                if j < nartists / 2:  # part of measurements because in first half
                    ax.lines[j].set_color('C2')
                else:
                    ax.lines[j].set_color('C3')
            plt.title('Vehicle ID ' + str(followerlist[vehind]) + ' has RMSE ' + str(
                round(vehrmse[vehind], 2)) + ' in platoon ' + str(platoonlist[platoonind]) +
                      ' with RMSE ' + str(round(platoonrmse[platoonind], 2)))
            plt.draw()

        return

    toggle1 = 1

    def key_press(event):
        nonlocal toggle1
        # toggle out of lane trajectories  for all vehicles
        if event.key in ['T', 't']:
            first = True
            visbool = None
            ax = event.inaxes  # get axes
            #            print(ax)
            #            print(ax.lines)
            for i in ax.lines:
                if i.get_alpha() != None:  # line has alpha then we do something
                    if first:  # for first line, we need to check whether it is currently turned on or off
                        visbool = i.get_visible()  # get visibility
                        visbool = not visbool  # set visbility to opposite
                        first = False  # after doing this check we don't need to do it again
                    i.set_visible(visbool)  # set desired visbility
            plt.draw()

        # toggle out of lane trajectories for current selected vehicle/platoon
        if event.key in ['U', 'u']:
            first = True
            visbool = None
            ax = event.inaxes  # get axes
            #            print(ax)
            #            print(ax.lines)
            for z in find_artists:
                i = ax.lines[z]
                if i.get_alpha() != None:  # line has alpha then we do something
                    if first:  # for first line, we need to check whether it is currently turned on or off
                        visbool = i.get_visible()  # get visibility
                        visbool = not visbool  # set visbility to opposite
                        first = False  # after doing this check we don't need to do it again
                    i.set_visible(visbool)  # set desired visbility
            plt.draw()

        # toggle between showing sim, meas, and both
        if event.key in ['Y', 'y']:
            ax = event.inaxes
            curtoggle = toggle1 % 3
            if curtoggle == 1:
                for j in range(0, halfn):
                    ax.lines[j].set_visible(0)
                for j in range(halfn, nartists):
                    ax.lines[j].set_visible(1)
            if curtoggle == 2:
                for j in range(0, halfn):
                    ax.lines[j].set_visible(1)
                for j in range(halfn, nartists):
                    ax.lines[j].set_visible(0)
            if curtoggle == 0:
                for j in range(nartists):
                    ax.lines[j].set_visible(1)
            toggle1 += 1
            plt.draw()

        return

    fig.canvas.callbacks.connect('pick_event', on_pick)
    fig.canvas.callbacks.connect('key_press_event', key_press)
    return


def plotColorLines(X, Y, SPEED, speed_limit, colormap = 'speeds', ind = 0):
    
    #helper for platoonplot
    axs = plt.gca()
    c = SPEED
    points = np.array([X, Y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    # if speed_limit:
    # 	norm = plt.Normalize(speed_limit[0], speed_limit[1])
    # else:
    # 	norm = plt.Normalize(c.min(), c.max())
    norm = plt.Normalize(speed_limit[0], speed_limit[1])
    if colormap =='speeds':
        lc = LineCollection(segments, cmap=palettable.colorbrewer.diverging.RdYlGn_4.mpl_colormap, norm=norm)
    elif colormap =='times': 
        cmap_list = [palettable.colorbrewer.sequential.Blues_9.mpl_colormap, palettable.colorbrewer.sequential.Oranges_9.mpl_colormap, 
                     palettable.colorbrewer.sequential.Greens_9.mpl_colormap, palettable.colorbrewer.sequential.Greys_9.mpl_colormap]
        
#        lc = LineCollection(segments, cmap=plt.get_cmap('viridis'), norm=norm)
        if ind > len(cmap_list)-1:
            ind = len(cmap_list)-1
        lc = LineCollection(segments, cmap=cmap_list[ind], norm=norm)
        
#    lc = LineCollection(segments, cmap=cm.get_cmap('RdYlBu'), norm=norm)
    lc.set_array(c)
    lc.set_linewidth(1)
    line = axs.add_collection(lc)
    return line

def plotformat(sim, auxinfo, roadinfo, starttimeind = 0, endtimeind = 3000, density = 2, indlist = [], specialind = 21):
    #get output from simulation into a format we can plot using plotting functions
    #output format is pretty inefficient in terms of memory usage - all plotting functions use this format though 
    
    #starttimeind = first time to be plotted 
    #endtimeind = last time to be plotted 
    
    #density = k plots every kth vehicle, indlist = [keys] plots keys only. 
    #specialind does not do anything
    #it handles the circular road by creating the wrap around as new vehicles. 
    
    L = roadinfo[0]
    platooninfo = {} #need platooninfo 0 - 4 : observation times
    meas = {} #need columns 1, 2,3, 7
    idcount = 0
    speciallist = []
    
    if indlist == []:
        uselist = list(sim.keys())[::density]
    else:
        uselist = indlist
    for i in uselist:
        cur = sim[i]
        
#        if i == specialind:
#            speciallist.append(idcount)
        
        #initialize output for current vehicle
        curtime = 0
        prevx = -1
        tlist = []
        xlist = []
        vlist = []
        platooninfo[idcount] = [curtime, curtime, None, None]
        for counter, j in enumerate(cur): #for each vehicle
            if counter < starttimeind:
                continue 
            
            if j[0] < prevx: #if distance decreases its because we wrapped around - store measurements in new vehicle 
                
                endtime = counter #get end time and update platooninfo
                platooninfo[idcount][2:] = [endtime-1,endtime-1]
                #update meas
                meas[idcount] = np.zeros((endtime-curtime,8))
                meas[idcount][:,1] = tlist
                meas[idcount][:,2] = xlist
                meas[idcount][:,3] = vlist
                #lane just set always to 1
                meas[idcount][:,7] = 1
            
                #reset iteration
                idcount += 1
                curtime = endtime
                prevx = j[0]
                tlist = [curtime]
                xlist = [j[0]]
                vlist = [j[1]]
                platooninfo[idcount] = [curtime, curtime, None, None]
                continue
                
            tlist.append(counter)
            xlist.append(j[0])
            vlist.append(j[1])
            prevx = j[0]
            
            if counter >= endtimeind:
                break
            
        #also need to finish current once for loop ends 
        endtime = counter #get end time and update platooninfo
        platooninfo[idcount][2:] = [endtime,endtime]
        #update meas
        meas[idcount] = np.zeros((endtime-curtime+1,8))
        meas[idcount][:,1] = tlist
        meas[idcount][:,2] = xlist
        meas[idcount][:,3] = vlist
        #lane just set always to 1
        meas[idcount][:,7] = 1
        #update iteration
        idcount += 1
            

    return meas, platooninfo

def platoonplot(meas, sim, platooninfo, platoon=[], newfig=True, clr=['C0', 'C1'],
                fulltraj=True, lane=None, opacity=.4, colorcode=True, speed_limit=[], timerange=[None, None]):  # plot platoon in space-time
    # meas - measurements in np array, rows are observations
    # sim - simulation in same format as meas. can pass in None and only meas will be shown, or can pass in the data and they will be plotted together
    # in different colors.
    # platooninfo (platooninfo) - dictionary containing information on each vehicle ID
    # platoon - default is [], in which case all keys of platooninfo are plotted. If passed in as a platoon (list of vehicle ID as [1:] so first entry not included)
    # only those vehicles will be plotted.

    # newfig = True - if True will create a new figure, otherwise it will use the current figure
    # clr = 'C0', assuming Colors = False, clr will control what colors will be used. Default is ['C0','C1'] which are the default matplotlib colors
    # this is used is sim is not None and colorcode = False
    # fulltraj = True controls how much of each trajectory to plot

    # lane = None - If passed in as a laneID, the parts of trajectories not in the lane ID given will be made opaque
    # colorcode = True - if colorcode is True, sim must be None, and we will plot the trajectories
    # colorcoded based on their speeds. It looks nice!
    # speed_limit = [] - only used when colorcode is True, if empty we will find the minimum and maximum speeds
    # and colorcode based on those speeds. Otherwise you can specify the min/max, and anything below/above
    # those limits will be colorcoded according to the limits
    # timerange = [None, None] - If fulltraj is True, this parameter is ingored
    # Otherwise, if values are passed in, only plot the trajectories in the provided time range

    # plots a platoon of vehicles in space-time plot.
    # features - can click on vehicles to display their IDs. Can compare meas and sim when colorcode is False.
    # can specify a lane, and make trajectories outside of that lane opaque.
    # can colorcode trajectories based on their speeds to easily see shockwaves and other structures.

    if sim is not None:
        colorcode = False
    
    ind = 2
    artist2veh = []

    if platoon != []:
        platooninfo = helper.platoononly(platooninfo, platoon)
    followerlist = list(platooninfo.keys())  # list of vehicle ID
    if lane != None: 
        for i in followerlist.copy(): 
            if lane not in np.unique(meas[i][:,7]):
                followerlist.remove(i)
    if newfig:
        fig = plt.figure()


    counter = 0
    mymin = 1e10
    mymax = 0
    for i in followerlist:
        curmin = min(meas[i][:, 3])
        curmax = max(meas[i][:, 3])
        if mymin > curmin:
            mymin = curmin
        if mymax < curmax:
            mymax = curmax

    if not speed_limit:
        speed_limit = [mymin, mymax]

    for i in followerlist:  # iterate over each vehicle
        veh = meas[i]        
        veh = extract_relevant_data(veh, i, platooninfo, fulltraj, timerange)
        
        # If current vehicle's data is irrelavant to given time range, move on to the next one
        if veh is None:
            continue
        
        x = veh[:, 1]
        y = veh[:, ind]
        speed_list = veh[:, 3]

        LCind = generate_LCind(veh, lane)

        for j in range(len(LCind) - 1):
            kwargs = {}
            if veh[LCind[j], 7] != lane and lane != None:
                kwargs = {'linestyle': '--', 'alpha': opacity}  # dashed line .4 opacity (60% see through)
                plt.plot(x[LCind[j]:LCind[j + 1]], y[LCind[j]:LCind[j + 1]], clr[0], **kwargs)
                artist2veh.append(counter)
            else:

                X = x[LCind[j]:LCind[j + 1]]
                Y = y[LCind[j]:LCind[j + 1]]
                SPEED = speed_list[LCind[j]:LCind[j + 1]]
                if colorcode:
                    line = plotColorLines(X, Y, SPEED, speed_limit=speed_limit)

                else:
                    plt.plot(x[LCind[j]:LCind[j + 1]], y[LCind[j]:LCind[j + 1]], clr[0], picker=5, **kwargs)
                    artist2veh.append(counter)

        counter += 1


    if sim != None:
        counter = 0
        for i in followerlist:  # iterate over each vehicle
            veh = sim[i]
            veh = extract_relevant_data(veh, i, platooninfo, fulltraj, timerange)
            
            if len(veh) == 0:
                continue
            
            x = veh[:, 1]
            y = veh[:, ind]

            LCind = generate_LCind(veh, lane)

            for j in range(len(LCind) - 1):
                kwargs = {}
                if veh[LCind[j], 7] != lane and lane != None:
                    kwargs = {'linestyle': '--', 'alpha': .4}  # dashed line .4 opacity (60% see through)
                plt.plot(x[LCind[j]:LCind[j + 1]], y[LCind[j]:LCind[j + 1]], clr[1], **kwargs)

            counter += 1

    find_artists = []
    nartists = len(artist2veh)
    
    def on_pick(event):
        nonlocal find_artists
        ax = event.artist.axes
        curind = ax.lines.index(event.artist)  # artist index

        if event.mouseevent.button == 1:  # left click selects vehicle
            # deselect old vehicle
            for j in find_artists:
                ax.lines[j].set_color('C0')
                if sim != None: 
                    ax.lines[j+nartists].set_color('C1')

            # select new vehicle
            vehind = artist2veh[curind]  # convert from artist to vehicle index
            find_artists = np.asarray(artist2veh)
            find_artists = np.nonzero(find_artists == vehind)[0]  # all artist indices which are associated with vehicle

            for j in find_artists:
                ax.lines[j].set_color('C3')
                if sim != None:
                    ax.lines[j+nartists].set_color('C3')
            plt.title('Vehicle ID ' + str(list(followerlist)[vehind]))
            plt.draw()
        plt.draw()

    # recolor the selected artist and all other artists associated with the vehicle ID so you can see what line you clicked on
    # change the title to say what vehicle you selected.

    fig.canvas.callbacks.connect('pick_event', on_pick)
    axs = plt.gca()

    plt.xlabel('time (frameID )')
    plt.ylabel('space (ft)')

    if colorcode:
        fig.colorbar(line, ax=axs)

    axs.autoscale(axis='x')
    axs.autoscale(axis='y')

    return

def extract_relevant_data(veh, i, platooninfo, fulltraj, timerange):
    #if fulltraj is True, plot between t_nstar - T_n; plot between t_n and T_nm1 otherwise
    #trajectories additionally must be between timerange if possible
    t_nstar, t_n, T_nm1, T_n = platooninfo[i][0:4]
        
    if fulltraj:  # entire trajectory including pre simulation and shifted end
        start = t_nstar
        end = T_n
    else:  # only show trajectory which is simulated
        start = t_n
        end = T_nm1
        
    if timerange[0] != None:
        if timerange[0] <= end:
            if start > timerange[0]:
                pass
            else:
                start = timerange[0]
        else:
            start = None
    
    if timerange[1] != None: 
        if end < timerange[1]:
            pass
        else: 
            end = timerange[1]
    
    if start == None: 
        return np.zeros((0,8))
    return veh[start-t_nstar:end-t_nstar+1, :]


def generate_LCind(veh, lane):
    if lane != None:
        # LCind is a list of indices where the lane the vehicle is in changes. Note that it includes the first and last index.
        LCind = np.diff(veh[:, 7])
        LCind = np.nonzero(LCind)[0] + 1
        LCind = list(LCind)
        LCind.insert(0, 0)
        LCind.append(len(veh[:, 7]))
            
    else: 
        LCind = [0, len(veh[:,1])]
        
    return LCind

def overlap(interval1, interval2):
    #given two tuples of start - end times, computes overlap between them
    #can pass None as either of values in interval2 to get better data 
    outint = interval1.copy()
    if interval2[0] != None: 
        if interval2[0] <= interval1[1]:
            if interval2[0] > interval1[0]: 
                outint[0] = interval2[0]
        else:
            return None
    if interval2[1] is not None: 
        if interval2[1] < interval1[1]: 
            outint[1] = interval2[1]
    
    return outint

def generate_changetimes(veh, col_index):
    #returns list of indices [ind] (from 0 index of whatever is passed in) where
    #veh[ind, col_index] is different from veh[ind-1, col_index]. Then to slice the different blocks, 
    #you can use veh[ind[0]:ind[1], col_index] where blocks have the same value repeated. 
    
    #this is a generalization of generate_LCind
    ind = np.diff(veh[:, col_index])
    ind = np.nonzero(ind)[0] + 1
    ind = list(ind)
    ind.insert(0, 0)
    ind.append(len(veh[:, col_index]))
    
    return ind
#########################################



# def calculateflows(meas, spacea, timea, agg):
# 	q = [[] for i in spacea]
# 	k = [[] for i in spacea]
#     #ronan modification - calculate minimum and maximum space, minimum and maximum time
# #spacealist = []
# #for i in spacea:
# #    spacealist.extend(i)
# #spacemin = min(spacealist)
# #spacemax = max(spacealist)
# #timemin = min(timea)
# #timemax = max(timea)
#
# 	intervals = []
# 	start = timea[0]
# 	end = timea[1]
# 	temp1 = start
# 	temp2 = start+agg
# 	while temp2 < end:
# 		intervals.append((temp1, temp2))
# 		temp1 = temp2
# 		temp2 += agg
# 	intervals.append((temp1, end))
#
#
# 	# ([time frame], [space])
# 	regions = [[([],[]) for j in intervals] for i in spacea]
#
# 	for id in meas:
# 		data = meas[id]
#         #ronan modification  - prune data so we don't have to look over as many datapoints
# #data = data[np.all([data[:,1] < timemax, data[:,1]>timemin], axis=0)]
# #data = data[np.all([data[:,2] < spacemax, data[:,2]>spacemin], axis=0)]
#
# 		region_contained = []
# 		region_data = {}  # key: tid, sid
# 		for i in data:
# 			t = i[1]
# 			s = i[2]
# 			t_id = -1
# 			s_id = -1
#
# 			for j in range(len(intervals)):
# 				if t<=intervals[j][1] and t>=intervals[j][0]:
# 					t_id = j
# 					break
# 			for j in range(len(spacea)):
# 				if s<=spacea[j][1] and s>=spacea[j][0]:
# 					s_id = j
# 					break
# 			if t_id == -1 or s_id == -1:
# 				continue
# 			id_key = str(t_id)+" "+str(s_id)
# 			if id_key not in region_data:
# 				# in t, in s, out t, out s
# 				region_data[id_key] = [t,s,-1,-1]
# 			else:
# 				region_data[id_key][2] = t
# 				region_data[id_key][3] = s
# 		for i in region_data:
# 			t_id, s_id = i.split(" ")
# 			t_id = int(t_id)
# 			s_id = int(s_id)
#
#
# 			regions[s_id][t_id][0].append(region_data[i][2]-region_data[i][0])
# 			regions[s_id][t_id][1].append(region_data[i][3] - region_data[i][1])
#
# 	for i in range(len(regions)):
# 		for j in range(len(regions[0])):
# 			area = (spacea[i][1]-spacea[i][0]) * (intervals[j][1]-intervals[j][0])
# 			q[i].append(sum(regions[i][j][1])/area)
# 			k[i].append(sum(regions[i][j][0])/area)
# 	return q, k


def calculateflows(meas, spacea, timea, agg, lane = None):
    q = [[] for i in spacea]
    k = [[] for i in spacea]

    spacealist = []
    for i in spacea:
        spacealist.extend(i)
    spacemin = min(spacealist)
    spacemax = max(spacealist)
    timemin = min(timea)
    timemax = max(timea)

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

    # ([time frame], [space])
    regions = [[([], []) for j in intervals] for i in spacea]

    for id in meas:
        #heuristic for selecting shorter region of data; many trajectories we can potentially ignore
        alldata = meas[id]
        alldata = alldata[np.all([alldata[:,1] < timemax, alldata[:,1]>timemin], axis=0)]
        alldata = alldata[np.all([alldata[:,2] < spacemax, alldata[:,2]>spacemin], axis=0)]
        if len(alldata) == 0:
           continue
       
        #if lane is given we need to find the segments of data inside the lane
        if lane is not None: 
            alldata = alldata[alldata[:,7]==lane] #boolean mask selects data inside lane
            inds = helper.sequential(alldata) #returns indexes where there are jumps
            indlist = []
            for i in range(len(inds)-1):
                indlist.append([inds[i], inds[i+1]])
        else: #otherwise can just use everything
            indlist = [[0,len(alldata)]]
        
        for i in indlist:
            data = alldata[i[0]:i[1]] #select only current region of data 
            if len(data) == 0:
                continue
            region_contained = []
            region_data = {}  # key: tid, sid
    
            for i in range(len(intervals)):
                try:
                    start = max(0, intervals[i][0] - data[0][1])
                    end = max(0, intervals[i][1] - data[0][1])
                except:
                    print("Empty data")
                start = int(start)
                end = int(end)
    
                if start == end:
                    continue
    
                dataInterval = data[start:end]
                spaceInterval = [j[2] for j in dataInterval]
    
                for j in range(len(spacea)):
                    try:
                        start = min(bisect.bisect_left(spaceInterval, spacea[j][0]), len(dataInterval)-1)
                        end = min(bisect.bisect_left(spaceInterval, spacea[j][1]), len(dataInterval)-1)
                        if start == end:
                            continue
                        regions[j][i][0].append(dataInterval[end][1] - dataInterval[start][1])
                        regions[j][i][1].append(dataInterval[end][2] - dataInterval[start][2])
    
                    except:
                        print("out index")

    for i in range(len(regions)):
        for j in range(len(regions[0])):
            area = (spacea[i][1] - spacea[i][0]) * (intervals[j][1] - intervals[j][0])
            q[i].append(sum(regions[i][j][1]) / area)
            k[i].append(sum(regions[i][j][0]) / area)
    return q, k


def plotflows(meas, spacea, timea, agg, type='FD', FDagg=None, lane = None):
    """
	aggregates microscopic data into macroscopic quantities based on Edie's generalized definitions of traffic variables
    
	meas = measurements, in usual format (dictionary where keys are vehicle IDs, values are numpy arrays)
    
	spacea = reads as ``space A'' (where A is the region where the macroscopic quantities are being calculated). 
    list of lists, each nested list is a length 2 list which ... represents the starting and ending location on road. 
    So if len(spacea) >1 there will be multiple regions on the road which we are tracking e.g. spacea = [[200,400],[800,1000]], 
    calculate the flows in regions 200 to 400 and 800 to 1000 in meas.
    
	timea = reads as ``time A'', should be a list of the times (in the local time of thedata). 
    E.g. timea = [1000,3000] calculate times between 1000 and 3000.
    
	agg = aggregation length, float number which is the length of each aggregation interval. 
    E.g. agg = 300 each measurement of the macroscopic quantities is over 300 time units in the data, 
    so in NGSim where each time is a frameID with length .1s, we are aggregating every 30 seconds.
    
	type = `FD', if type is `FD', plot data in flow-density plane. Otherwise, plot in flow-time plane.
    
	FDagg = None - If FDagg is None and len(spacea) > 1, aggregate q and k measurements together. 
    Otherwise if FDagg is an int, only show the q and k measurements for the corresponding spacea[int]
    
    lane = None - If lane is given, it only uses measurement in that lane. 
    
    Note that if the aggregation intervals are too small the plots won't really make sense 
    because a lot of the variation is just due to the aggregation. Increase either agg
    or spacea regions to prevent this problem. 
	"""
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

    q, k = calculateflows(meas, spacea, timea, agg, lane = lane)
    time_sequence = []
    time_sequence_for_line = []

    if len(q) > 1 and FDagg != None:
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
        print(q)
        plt.xlabel("time")
        plt.ylabel("flow")
        plt.show()

    return


def plotdist(meas1, sim1, platooninfo1, my_id, fulltraj=False, delay=0, h=.1):
    # this is just for a single vehicle
    # this function plots the distance for measurement and simulation, given platooninfo
    # can do either the whole trajectory or just the simulated part
    # will use the current figure so if you want it in its own figure need to initialize a figure beforehand
    # can do either delay or no delay
    meas = meas1[my_id]
    sim = sim1[my_id]
    platooninfo = platooninfo1[my_id]

    if not fulltraj:
        t_nstar, t_n, T_nm1, T_n = platooninfo[0:4]
        if delay != 0:
            offset = math.ceil(delay / h)
            offsetend = math.floor(delay / h)
            if T_nm1 + offsetend >= T_n:
                end = T_n
            else:
                end = T_nm1 + offsetend
            start = t_n + offset
        else:
            start = t_n
            end = T_nm1
        meas = meas[start - t_nstar:end - t_nstar + 1, :]
        sim = sim[start - t_nstar:end - t_nstar + 1, :]

    #    plt.figure()
    plt.xlabel('time (.1 seconds)')
    plt.ylabel('space (feet)')
    plt.plot(meas[:, 1], meas[:, 2], 'k', sim[:, 1], sim[:, 2], 'r')
    plt.legend(['Measurements', 'Simulation after calibration'])
    plt.title('Space-Time plot for vehicle ' + str(my_id))

    return


def plotspeed(meas1, sim1, platooninfo1, my_id, fulltraj=False, delay=0, h=.1, newfig=True):
    # this is just for a single vehicle
    # this function plots the distance for measurement and simulation, given platooninfo
    # can do either the whole trajectory or just the simulated part
    # can do either delay or no delay
    meas = meas1[my_id]
    sim = sim1[my_id]
    platooninfo = platooninfo1[my_id]

    if not fulltraj:
        t_nstar, t_n, T_nm1, T_n = platooninfo[0:4]
        if delay != 0:
            offset = math.ceil(delay / h)
            offsetend = math.floor(delay / h)
            if T_nm1 + offsetend >= T_n:
                end = T_n
            else:
                end = T_nm1 + offsetend
            start = t_n + offset
        else:
            start = t_n
            end = T_nm1
        meas = meas[start - t_nstar:end - t_nstar + 1, :]
        sim = sim[start - t_nstar:end - t_nstar + 1, :]

    if newfig:
        plt.figure()
    plt.xlabel('time (.1 seconds)')
    plt.ylabel('speed (feet/second)')
    plt.plot(meas[:, 1], meas[:, 3], 'k', sim[:, 1], sim[:, 3], 'r')
    plt.legend(['Measurements', 'Simulation after calibration'])
    plt.title('Speed-Time plot for vehicle ' + str(my_id))

    return

def plotvhd(meas, sim, platooninfo, vehicle_id, draw_arrow=False, arrow_interval=20, effective_headway=False, rp=None, h=.1,
            datalen=9, timerange=[None, None], lane=None, delay=0, newfig=True, plot_color_line=False):
    # draw_arrow = True: draw arrows (indicating direction) along with trajectories; False: plot the trajectories only
    # effective_headway = False - if True, computes the relaxation amounts using rp, and then uses the headway + relaxation amount to plot instead of just the headway
    # rp = None - effective headway is true, rp is a float which is the parameter for the relaxation amount
    # h = .1 - data discretization
    # datalen = 9
    # timerange = [None, None] indicates the start and end timestamps that the plot limits
    # lane = None, the lane number that need highlighted: Trajectories in all other lanes would be plotted with opacity
    # delay = 0 - gets starting time for newell model
    # newfig = True - if True will create a new figure, otherwise it will use the current figure
    # plot_color_line = False; If set to true, plot all trajectories using colored lines based on timestamp

    ####plotting
    if newfig:
        plt.figure()
    plt.xlabel('space headway (ft)')
    plt.ylabel('speed (ft/s)')
    title_text = 'space-headway for vehicle ' + " ".join(list(map(str, (vehicle_id))))
    if lane is not None:
        title_text = title_text + ' on lane ' + str(lane)
    plt.title(title_text)
    ax = plt.gca()
    artist_list = []

    if sim is None:
        # If sim is None, plot meas for all vehicles in vehicle_id
        for count, my_id in enumerate(vehicle_id):
            ret_list = process_one_vehicle(ax, meas, sim, platooninfo, my_id, timerange, lane, plot_color_line, effective_headway, rp, h, datalen, delay, count = count )
            artist_list.extend(ret_list)
    else:
        # If both meas and sim are provided,
        # will plot both simulation and measurement data for the first vehicle in vehicle_id
        if len(vehicle_id) > 1: 
            print('plotting first vehicle '+str(vehicle_id[0])+' only')
        ret_list = process_one_vehicle(ax, meas, sim, platooninfo, vehicle_id[0], timerange, lane, plot_color_line, effective_headway, rp, h, datalen, delay)
        artist_list.extend(ret_list)

    if plot_color_line:
        ax.autoscale(axis = 'x')
        ax.autoscale(axis = 'y')
    else:
        organize_legends()

    if draw_arrow:
        for art in artist_list:
            add_arrow(art[0], arrow_interval)

    return

# This function will process and prepare xy-coordinates, color, labels, etc. 
# necessary to plot trajectories for a given vehicle and then invoke plot_one_vehicle() function to do the plotting
def process_one_vehicle(ax, meas, sim, platooninfo, my_id, timerange, lane, plot_color_line, effective_headway=False, rp=None, h=.1, datalen=9, delay=0, count = 0):
    artist_list = []
    if effective_headway:
        leadinfo, folinfo, rinfo = helper.makeleadfolinfo([my_id], platooninfo, meas)
    else:
        leadinfo, folinfo, rinfo = helper.makeleadfolinfo([my_id], platooninfo, meas, relaxtype = 'none')
    
    t_nstar, t_n, T_nm1, T_n = platooninfo[my_id][0:4]
    
    # Compute and validate start and end time
    start, end = compute_validate_time(timerange, t_n, T_nm1, h, delay)
            
    frames = [t_n, T_nm1]
    relax, unused = r_constant(rinfo[0], frames, T_n, rp, False, h)  # get the relaxation amounts for the current vehicle; these depend on the parameter curp[-1] only.
    meas_label = str(my_id)

    headway = None
    if sim is not None:
        headway = compute_headway(t_nstar, t_n, T_n, datalen, leadinfo, start, sim, my_id, relax)
        sim_color = next(ax._get_lines.prop_cycler)['color']
        meas_label = 'Measurements'
        ret_list = plot_one_vehicle(headway[:end + 1 - start], sim[my_id][start - t_nstar:end + 1 - t_nstar, 3],
                                            sim[my_id][start - t_nstar:end + 1 - t_nstar, 1],
                                            sim[my_id][start - t_nstar:end + 1 - t_nstar, 7],
                                            lane, plot_color_line, leadinfo, start, end, 'Simulation', sim_color, count = count)
        artist_list.extend(ret_list)
        
    trueheadway = compute_headway(t_nstar, t_n, T_n, datalen, leadinfo, start, meas, my_id, relax)
    meas_color = next(ax._get_lines.prop_cycler)['color']
    ret_list = plot_one_vehicle(trueheadway[:end + 1 - start], meas[my_id][start - t_nstar:end + 1 - t_nstar, 3],
                                            meas[my_id][start - t_nstar:end + 1 - t_nstar, 1], 
                                            meas[my_id][start - t_nstar:end + 1 - t_nstar, 7],
                                            lane, plot_color_line, leadinfo, start, end, meas_label, meas_color, count = count)
    artist_list.extend(ret_list)
    return artist_list

def plot_one_vehicle(x_coordinates, y_coordinates, timestamps, lane_numbers, target_lane, plot_color_line, leadinfo, start, end, label, color, opacity=.4, count = 0):
    # If there is at least a leader change,
    # we want to separate data into multiple sets otherwise there will be horizontal lines that have no meanings
    # x_coordinates and y_coordinates will have the same length,
    # and x_coordinates[0] and y_coordinates[0] have the same time frame == start
    # lane_numbers list has corresponding lane number for every single y_coordinates (speed)
    temp_start = 0
    leader_id = leadinfo[0][0][0]
    artist_list = []

    ##############################
#    if plot_color_line:
#        lines = plotColorLines(x_coordinates, y_coordinates, timestamps, [timestamps[0], timestamps[-1]])
#        return artist_list
    ##############################

    for index in range(0, len(x_coordinates)):
        current_leader_id = find_current_leader(start + index, leadinfo[0])
        if current_leader_id != leader_id:
            # Detected a leader change, plot the previous set
            leader_id = current_leader_id

            # Check if should do color line plotting
            if plot_color_line:
                lines = plotColorLines(x_coordinates[temp_start:index], y_coordinates[temp_start:index], timestamps[temp_start:index], [start- 100, end+10], colormap = 'times', ind = count)
            else:
                kwargs = {}
                # Check if lane changed as well, if yes, plot opaque lines instead
                if lane_numbers[temp_start] != target_lane and target_lane is not None:
                    kwargs = {'alpha': opacity}  # .4 opacity (60% see through)
                art = plt.plot(x_coordinates[temp_start:index], y_coordinates[temp_start:index], label=label, color=color, **kwargs)
                artist_list.append(art)

            temp_start = index

    # Plot the very last set, if there is one
    if plot_color_line:
        lines = plotColorLines(x_coordinates[temp_start:], y_coordinates[temp_start:], timestamps[temp_start:], [start-100, end+10], colormap = 'times', ind = count)
    else:
        kwargs = {}
        if lane_numbers[temp_start] != target_lane and target_lane is not None:
            kwargs = {'alpha': opacity}  # .4 opacity (60% see through)
        art = plt.plot(x_coordinates[temp_start:], y_coordinates[temp_start:], label=label, color=color, **kwargs)
        artist_list.append(art)
    return artist_list

# This function is used to merge legends (when necessary) especially the same vehicle has multiple trajectories sections
# due to leader or lane changes
def organize_legends():
    handles, labels = plt.gca().get_legend_handles_labels()
    newLabels, newHandles = [], []
    for handle, label in zip(handles, labels):
        if label not in newLabels:
            newLabels.append(label)
            newHandles.append(handle)
    plt.legend(newHandles, newLabels)

def add_arrow(line, arrow_interval=20, direction='right', size=15, color=None):
    """
    add an arrow to a line.

    line:           Line2D object
    arrow_interval: the min length on x-axis between two arrows, given a list of xdata,
                    this can determine the number of arrows to be drawn
    direction:      'left' or 'right'
    size:           size of the arrow in fontsize points
    color:          if None, line color is taken.
    """
    if color is None:
        color = line.get_color()

    xdata = line.get_xdata()
    ydata = line.get_ydata()
    curdist = 0
    for i in range(len(xdata)-1):
        curdist += ((xdata[i+1] - xdata[i])**2 + (ydata[i+1] - ydata[i])**2 )**.5
        if curdist > arrow_interval:
            curdist += - arrow_interval
            start_ind = i
            
            if start_ind == 0 or start_ind == len(xdata) - 1:
                continue
    
#            if direction == 'right':
            end_ind = start_ind + 1
#            else:
#                end_ind = start_ind - 1
    
            line.axes.annotate('',
                xytext=(xdata[start_ind], ydata[start_ind]),
                xy=(xdata[end_ind], ydata[end_ind]),
                arrowprops=dict(arrowstyle="->", color=color),
                size=size
            )
            

#    x_max = max(xdata)
#    x_min = min(xdata)
#    num_arrows = math.floor((x_max - x_min) / arrow_interval) # // TO DO could compute distance traveled and put arrows based on that  low priority 
#
#    for i in range(0, num_arrows):
#        position = x_min + i * arrow_interval
#
#        # find closest index
#        start_ind = np.argmin(np.absolute(xdata - position))
#
#        # To avoid index out of bounds, skip if
#        # start_ind is either index 0, or xdata.length - 1
#        if start_ind == 0 or start_ind == len(xdata) - 1:
#            continue
#
#        if direction == 'right':
#            end_ind = start_ind + 1
#        else:
#            end_ind = start_ind - 1
#
#        line.axes.annotate('',
#            xytext=(xdata[start_ind], ydata[start_ind]),
#            xy=(xdata[end_ind], ydata[end_ind]),
#            arrowprops=dict(arrowstyle="->", color=color),
#            size=size
#        )

def animatevhd(meas, sim, platooninfo, platoon, lentail=20, timerange=[None, None], 
               lane = None, opacity = .2, interval = 10, rp = None, h=.1, delay=0):
    # plot multiple vehicles in phase space (speed v headway)
    #meas, sim - data in key = ID, value = numpy array format, pass sim = None to plot one set of data 
    #platooninfo
    # platoon - list of vehicles to plot
    # lentail = 20 - number of observations to show in the past
    # timerange = [usestart, useend]
    # rp = None - can add relaxation to the headway, if you pass a number this is used as relaxation amount 
    # h = .1 - data discretization, deprecated 
    # delay = 0 - gets starting time for newell model, deprecated
    #lane = None - can specify a lane to make trajectories opaque if not in desired lane 
    #opacity = .2 - controls opacity (set = 0 to not show, if 1 its equivalent to lane = None)
    
    #I think this function has good general design for how a animation for traffic simulation should be structured in python
    #each vehicle has a dictionary, which contains relevant plotting data and any extra information (keyword args, start/end times, etc)
    #create a sorted list with tuples of the (times, dictionary, 'add' or 'remove') which represent when artists (vehicles)
    #will enter or leave animation. THen in animation, in each frame check if there are any artists to add or remove; 
    #if you add a vehicle, create an artist (or potentially multiple artists) and add its reference to the dictionary
    #keep a list of all active dictionaries (vehicles) during animation - update artists so you can use blitting and 
    #dont have to keep redrawing - its faster and animation is smoother this way. 
    fig = plt.figure()
    plt.xlabel('space headway (ft)')
    plt.ylabel('speed (ft/s)')
    plt.title('space-headway for vehicle ' + " ".join(list(map(str, (platoon)))))
    plotsim = False if sim is None else True
    xmin, xmax, ymin, ymax = math.inf, -math.inf, math.inf, -math.inf
    
    startendtimelist = []
    
    for veh in platoon: 
        t_nstar, t_n, T_nm1, T_n = platooninfo[veh][:4]
        #heuristic will speed up plotting if a large dataset is passed in
        if timerange[0] is not None: 
            if T_nm1 < timerange[0]: 
                continue
        if timerange[1] is not None: 
            if t_n > timerange[1]:
                continue
            
        #compute headway, speed between t_n and T_nm1
        headway = compute_headway2(veh, meas, platooninfo, rp, h)
        speed = meas[veh][t_n-t_nstar:T_nm1-t_nstar+1,3]
        if plotsim: 
            simheadway = compute_headway2(veh, sim, platooninfo, rp, h)
            simspeed = sim[veh][t_n-t_nstar:T_nm1-t_nstar,3]
        
        curxmin, curxmax, curymin, curymax = min(headway), max(headway), min(speed), max(speed)
        xmin, xmax, ymin, ymax = min([xmin, curxmin]), max([xmax, curxmax]), min([ymin, curymin]), max([ymax, curymax])
    
        #split up headway/speed into sections based on having a continuous leader
        #assume that sim and measurements have same leaders in this code 
        ind = generate_changetimes(meas[veh][t_n-t_nstar:T_nm1-t_nstar+1,:], 4) 
        for i in range(len(ind)-1):
            #each section has the relevant speed, headway, start and end times, and opaque. 
            newsection = {}
            
            #start and end times are in real time (not slices indexing).
            start = ind[i] + t_n
            end = ind[i+1]-1 + t_n
            curlane = meas[veh][start - t_nstar, 7]
            times = overlap([start, end], timerange) #times of section to use, in real time 
            if times == None: 
                continue
            newsection['hd'] = headway[times[0] - t_n:times[1]+1-t_n ]
            newsection['spd'] = speed[times[0] - t_n:times[1]+1-t_n ]
            newsection['start'] = times[0]
            newsection['end'] = times[1]
            kwargs = {'color': 'C0'}
            if lane != None and curlane != lane: 
                kwargs['alpha'] = opacity
            newsection['kwargs'] = kwargs
            newsection['veh'] = str(int(veh))
            
            if plotsim: 
                #literally the same thing repeated 
                newsimsection = {}
                newsimsection['hd'] = simheadway[times[0] - t_n:times[1]+1-t_n ]
                newsimsection['spd'] = simspeed[times[0] - t_n:times[1]+1-t_n ]
                newsimsection['start'] = times[0]
                newsimsection['end'] = times[1]
                kwargs = {'color': 'C1'}
                if lane != None and curlane != lane: 
                    kwargs['alpha'] = opacity
                newsimsection['kwargs'] = kwargs
                newsimsection['veh'] = str(int(veh))
                
            startendtimelist.append((times[0], newsection, 'add'))
            startendtimelist.append((times[1]+lentail+1, newsection, 'remove'))
            if plotsim:
                startendtimelist.append((times[0], newsimsection, 'add'))
                startendtimelist.append((times[1]+lentail+1, newsimsection, 'remove'))
           
    #sort timelist
    startendtimelist.sort(key = lambda x: x[0]) #sort according to times
    ax = plt.gca()
    ax.set_xlim(xmin - 5, xmax + 5)
    ax.set_ylim(ymin - 5, ymax + 5)
    seclist = []
    times = [startendtimelist[0][0], startendtimelist[-1][0]]
    frames = list(range(times[0], times[1]+1))
    usetimelist = None
    
    def init():
        nonlocal usetimelist
        nonlocal seclist
        artists = []
        for sec in seclist: 
            sec['traj'].remove()
            sec['label'].remove()
            artists.append(sec['traj'])
            artists.append(sec['label'])
        seclist = []
        usetimelist = startendtimelist.copy()
        return artists
    def anifunc(frame):
        nonlocal seclist
        nonlocal usetimelist
        artists = []
        #add or remove vehicles as needed
        while len(usetimelist) > 0: 
            nexttime = usetimelist[0][0]
            if nexttime == frame: 
                time, sec, task = usetimelist.pop(0)
                if task == 'add':
                    #create artists and keep reference to it in the dictionary - keep dictionary in seclist - all active trajectories
                    traj = ax.plot([xmin,xmax],[ymin,ymax], **sec['kwargs'])[0]
                    label = ax.annotate(sec['veh'], (xmin,ymin), fontsize = 7)
                    sec['traj']  = traj
                    sec['label'] = label
                    seclist.append(sec)
                elif task == 'remove': 
                    #remove artists
                    seclist.remove(sec)
                    sec['traj'].remove()
                    sec['label'].remove()
                    
                    artists.append(sec['traj'])
                    artists.append(sec['label'])
            else:
                break
        
        for sec in seclist: 
            #do updating here
            animatevhdhelper(sec, frame, lentail)
            artists.append(sec['traj'])
            artists.append(sec['label'])
            
            
        return artists
    
    ani = animation.FuncAnimation(fig, anifunc, init_func = init, frames = frames, blit = True, interval = interval)
    
    return ani
            

def animatevhdhelper(sec, time, lentail):      
    starttime = sec['start']
    endtime = sec['end']
    if time > endtime:
        end = endtime
    else: 
        end = time
    
    if time < starttime + lentail+1:
        start = starttime
    else: 
        start = time - lentail
    
    sec['traj'].set_data(sec['hd'][start - starttime: end - starttime+1], sec['spd'][start - starttime:end - starttime+1])
    sec['label'].set_position((sec['hd'][end - starttime], sec['spd'][end - starttime]))
    return


def find_current_leader(current_frame, leadinfo):
    # leadinfo is already only about one vehicle id
    # leadinfo is of the form [[leader, start_frame, end_frame], [new_leader, end_frame+1, another_end_frame]]
    leader_id = leadinfo[0][0]
    for k in range(len(leadinfo)):
        if leadinfo[k][1] <= current_frame and current_frame <= leadinfo[k][2]:
            leader_id = leadinfo[k][0]
    # After validation of start and end frame, this function is guaranteed to return a valid result
    return leader_id

def compute_validate_time(timerange, t_n, T_nm1, h=.1, delay=0):
    # start time validation
    # If passed in as None, or any value outside [t_n, T_nm1], defaults to t_n
    if timerange[0] is None or timerange[0] < t_n or timerange[0] >= T_nm1:
        start = t_n
        if delay != 0:
            offset = math.ceil(delay / h)
            start = t_n + offset
    else:
        start = timerange[0]
        
    # end time validation
    # If passed in as None, or any value outside [t_n, T_nm1], or smaller than timerange[0], default to T_nm1
    if timerange[1] is None or timerange[1] < timerange[0] or timerange[1] > T_nm1:
        end = T_nm1
    else:
        end = timerange[1]
    return start, end

def compute_headway(t_nstar, t_n, T_n, datalen, leadinfo, start, dataset, veh_id, relax):
    lead = np.zeros((T_n + 1 - t_n, datalen))  # initialize the lead vehicle trajectory
    for j in leadinfo[0]:
        curleadid = j[0]  # current leader ID
        leadt_nstar = int(dataset[curleadid][0, 1])  # t_nstar for the current lead, put into int
        lead[j[1] - t_n:j[2] + 1 - t_n, :] = dataset[curleadid][j[1] - leadt_nstar:j[2] + 1 - leadt_nstar,:]  # get the lead trajectory from simulation
    headway = lead[start - t_n:, 2] - dataset[veh_id][start - t_nstar:, 2] - lead[start - t_n:, 6] + relax[start - t_n:]
    return headway

def compute_headway2(veh, data, platooninfo, rp, h =.1):
    #compute headways from data and platooninfo, possibly adding relaxation if desired
    #different format than compute_headway 
    
    relaxtype = 'both' if rp is not None else 'none'
    leadinfo, unused, rinfo = helper.makeleadfolinfo([veh], platooninfo, data, relaxtype = relaxtype)
    t_nstar, t_n, T_nm1, T_n = platooninfo[veh][:4]
    relax, unused = r_constant(rinfo[0], [t_n, T_nm1], T_n, rp, False, h)
    
    lead = np.zeros((T_nm1 + 1 - t_n, 9))  # initialize the lead vehicle trajectory
    for j in leadinfo[0]:
        curleadid = j[0]  # current leader ID
        leadt_nstar = int(data[curleadid][0, 1])  # t_nstar for the current lead, put into int
        lead[j[1] - t_n:j[2] + 1 - t_n, :] = data[curleadid][j[1] - leadt_nstar:j[2] + 1 - leadt_nstar,:]  # get the lead trajectory from simulation
    headway = lead[:, 2] - data[veh][t_n - t_nstar:T_nm1 - t_nstar+1, 2] - lead[:, 6] + relax[:T_nm1+1-t_n]
    
    return headway 
    
    
    
    

def compute_line_data(headway, i, lentail, dataset, veh_id, time):
    trajectory = (headway[i:i + lentail], dataset[veh_id][time + i:time + i + lentail, 3])
    label = (headway[i + lentail], dataset[veh_id][time + i + lentail, 3])
    
    # Compute x_min, y_min, x_max and y_max for the given data and return
    if lentail == 0:
        x_min = 0
        y_min = 0
        x_max = 0
        y_max = 0
    else:
        x_min = min(headway[i:i + lentail])
        x_max = max(headway[i:i + lentail])
        y_min = min(dataset[veh_id][time + i:time + i + lentail, 3])
        y_max = max(dataset[veh_id][time + i:time + i + lentail, 3])
    
    return trajectory, label, x_min, y_min, x_max, y_max


def animatetraj(meas, followerchain, platoon=[], usetime=[], presim=True, postsim=True, datalen=9, speed_limit = [], 
                   show_ID = True, interval = 10):
    #plots vehicles platoon using data meas.

    # platoon = [] - if given as a platoon, only plots those vehicles in the platoon (e.g. [[],1,2,3] )
    # usetime = [] - if given as a list, only plots those times in the list (e.g. list(range(1,100)) )
    # presim = True - presim and postsim control whether the entire trajectory is displayed or just the simulated parts (t_nstar - T_n versus T-n - T_nm1)
    # postsim = True

    if platoon != []:
        followerchain = helper.platoononly(followerchain, platoon)
    platoontraj, mytime = helper.arraytraj(meas, followerchain, presim, postsim, datalen)
    if not usetime:
        usetime = mytime

    fig = plt.figure(figsize=(10, 4))  # initialize figure and axis
    ax = fig.add_axes([0, 0, 1, 1], frameon=False)
    ax.set_xlim(0, 1600), ax.set_xlabel('localY')
    ax.set_ylim(7.5, 0), ax.set_ylabel('laneID')

    scatter_pts = ax.scatter([], [], c=[], cmap=palettable.colorbrewer.diverging.RdYlGn_4.mpl_colormap, marker=">") #cm.get_cmap('RdYlBu')

    if speed_limit == []:
        maxspeed = 0
        minspeed = math.inf
        for i in followerchain.keys():
            curmax = max(meas[i][:,3])
            curmin = min(meas[i][:,3])
            if curmin < minspeed:
                minspeed = curmin
            if curmax > maxspeed:
                maxspeed = curmax
        norm = plt.Normalize(minspeed,maxspeed)
    else:
        norm = plt.Normalize(speed_limit[0], speed_limit[1])

    fig.colorbar(scatter_pts, cmap=cm.get_cmap('RdYlBu'), norm=norm, shrink=0.7)
    current_annotation_dict = {}

    def aniFunc(frame):
        artists = [scatter_pts]
        ax = plt.gca()
        curdata = platoontraj[usetime[frame]]
        X = curdata[:, 2]
        Y = curdata[:, 7]
        speeds = curdata[:, 3]
        ids = curdata[:, 0]
        existing_vids = list(current_annotation_dict.keys()).copy()

        # Go through ids list
        # If the annotation already exists, modify it via set_position
        # If the annotation doesn't exist before, introduce it via ax.annotate
        if show_ID:
            for i in range(len(ids)):
                vid = ids[i]
                if vid in current_annotation_dict.keys():
                    current_annotation_dict[vid].set_position((X[i], Y[i]))
                    existing_vids.remove(vid)
                else:
                    current_annotation_dict[vid] = ax.annotate(str(int(vid)), (X[i], Y[i]), fontsize=7)
                artists.append(current_annotation_dict[vid])
    
            # Afterwards, check if existing annotations need to be removed, process it accordingly
            if len(existing_vids) > 0:
                for vid in existing_vids:
                    artists.append(current_annotation_dict[vid])
                    current_annotation_dict[vid].remove()
                    del current_annotation_dict[vid]
                

        c = speeds
        pts = [[X[i], Y[i]] for i in range(len(X))]
        data = np.vstack(pts)
        scatter_pts.set_offsets(data)
        scatter_pts.set_array(c)
        return artists

    def init():
        artists = [scatter_pts]
        ax = plt.gca()
        if show_ID:
            for vid, annotation in list(current_annotation_dict.items()).copy():
                artists.append(annotation)
                annotation.remove()
                del current_annotation_dict[vid]
        curdata = platoontraj[usetime[0]]
        X = curdata[:, 2]
        Y = curdata[:, 7]
        speeds = curdata[:, 3]
        ids = curdata[:, 0]
        for i in range(len(ids)):
            current_annotation_dict[ids[i]] = ax.annotate(str(int(ids[i])), (X[i], Y[i]), fontsize=7)
            artists.append(current_annotation_dict[ids[i]])
        c = speeds
        pts = [[X[i], Y[i]] for i in range(len(X))]
        data = np.vstack(pts)
        scatter_pts.set(norm=norm)
        scatter_pts.set_offsets(data)
        scatter_pts.set_array(c)
        return artists

    out = animation.FuncAnimation(fig, aniFunc, init_func=init, frames=len(usetime), interval=interval, blit = True)

    return out

####################################


def wtplot(meas, ID):
    testveh = ID

    test = meas[testveh][:, 3]

    #    X = np.arange(150,250)
    X = np.arange(15, 100)
    # X = np.array([64])
    out, out2 = pywt.cwt(test, X, 'mexh', sampling_period=.1)
    Y = meas[testveh][:, 1]
    X, Y, = np.meshgrid(X, Y);
    X = np.transpose(X);
    Y = np.transpose(Y)

    plt.close('all')
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, np.abs(out), cmap=cm.coolwarm)
    ax.view_init(azim=0, elev=90)

    # plt.figure()
    # plt.imshow(out,cmap = cm.coolwarm)
    plt.figure(figsize=(11, 5))
    plt.subplot(1, 2, 1)
    plotspeed(meas, meas, platooninfo, testveh, fulltraj=True)

    energy = np.sum(np.abs(out), axis=0)
    plt.subplot(1, 2, 2)
    plt.plot(meas[testveh][:, 1], energy)
    plt.show()

    return


def wt(series, scale):
    out, out2 = pywt.cwt(series, scale, 'mexh')
    energy = np.sum(np.abs(out), 0)
    return energy


# example of executing above code
# wtplot(meas,100) #example of output;

def meanspeedplot(data, timeint, spacebins, lane=1, use_avg='mean'):
    # data - raw data format (loaded in from csv and put into our vanilla format)
    # timeint - number of observations of data (in terms of data resolution) in each aggregated speed;
    # e.g. 50 with timestep of .1 = 5 second aggregation
    # spacebins - number of intervals the position on road is divided into
    # use_avg = 'mean' - controls averaging for speeds. if 'mean' then does arithmetic mean. if 'harm' then harmonic mean.
    # lane = 1 - choose which lane of the data to plot. (does 1 lane at a time)

    # choose which lane. (note a possible feature would be the ability to choose multiple lanes at once)
    if type(data) == dict: 
        data = np.concatenate(list(data.values()))
    
    data = data[data[:, 7] == lane]

    # can make this iterate once instead of 4 times
    t0 = min(data[:, 1])
    tend = max(data[:, 1])
    x0 = min(data[:, 2])
    xend = max(data[:, 2])

    times = np.arange(t0, tend, timeint)
    if times[-1] != tend:
        times = np.append(times, tend)

    xint = (xend - x0) / spacebins
    x = np.arange(x0, xend, xint)
    x = np.append(x, xend)

    X, Y = np.meshgrid(times, x)  # leave out the last point where making the grid to plot over

    speeds = []
    for i in range(len(times) - 1):  # minus 1 here because we have n+1 points but this defines only n bins.
        speeds.append([])
        for j in range(len(x) - 1):
            speeds[i].append([])

    for i in range(len(data)):  # sort all of the speeds
        curt = data[i, 1]
        curx = data[i, 2]
        curv = data[i, 3]
        # this is too slow replace with modulus operations

        curtimebin = math.floor((curt - t0) / timeint)
        curxbin = math.floor((curx - x0) / xint)

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
        except IndexError:  # at most extreme point we want it to go into the last bin not into a new bin (which would contain only itself)
            if curxbin == len(x):
                curxbin += -1
            if curtimebin == len(times):
                curtimebin += -1

    meanspeeds = X.copy()  # initialize output
    for i in range(len(times) - 1):  # populate output
        for j in range(len(x) - 1):
            cur = speeds[i][j]
            if 'mean' in use_avg:  # choose whether to do arithmetic or harmonic mean. if bin is empty, we set the value as np.nan (not the same as math.nan which is default)
                cur = np.mean(cur)
                if math.isnan(cur):
                    cur = np.nan  # np.nan will get changed into different color
            else:
                if len(cur) == 0:
                    cur = np.nan
                else:
                    cur = harmonic_mean(cur)

            meanspeeds[j, i] = cur
            ######
    # surface plot #old#
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

    cmap = cm.RdYlBu  # RdYlBu is probably the best colormap overall for this
    cmap.set_bad('white', 1.)  # change np.nan into white color

    fig2 = plt.figure()
    #    ax2 = fig2.add_subplot(111)
    #    ax2.imshow(meanspeeds, cmap=cmap)
    plt.pcolormesh(X, Y, meanspeeds, cmap=cmap)
    plt.xlabel('Time (.1 s)')
    plt.ylabel('Space (ft)')
    cbar = plt.colorbar()
    cbar.set_label('Speed (ft/s)')

    return speeds, X, Y, meanspeeds


# below code will allow you to select rectangles but here we want to select polygons.

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
    if event.key in ['A', 'a']:  # and not toggle_selector.RS.active:
        print(' RectangleSelector activated.')
        toggle_selector.RS.set_active(True)
    if event.key == 'enter':
        print(toggle_selector.RS.extents)


def selectoscillation(data1, timeint, spacebins, lane=1, use_avg='mean', region_shape='p'):
    """
    \\ TO DO \\
    selectvehID needs some more features added, refer to notes on paper for more details. 
    Basically we want to be able to sort vehicles on key press (note sortveh3 has been tested and is passing), remove vehicles before/after, 
    arrow keys to manually go through selected vehicles, 
    some ability to put up multiple speed/wavelet series at a time might be nice as well. 
    
    I think there is a bug with the initialization of selectvehID 
    \\ TO DO \\
    """

    # data1 - raw data format (loaded in from csv and put into our vanilla format)
    # timeint - number of observations of data (in terms of data resolution) in each aggregated speed;
    # e.g. 50 with timestep of .1 = 5 second aggregation
    # spacebins - number of intervals the position on road is divided into
    # use_avg = 'mean' - controls averaging for speeds. if 'mean' then does arithmetic mean. if 'harm' then harmonic mean.
    # lane = 1 - choose which lane of the data to plot. (does 1 lane at a time)

    # choose which lane. (note a possible feature would be the ability to choose multiple lanes at once)
    data = data1[data1[:, 7] == lane]

    # can make this iterate once instead of 4 times
    t0 = min(data[:, 1])
    tend = max(data[:, 1])
    x0 = min(data[:, 2])
    xend = max(data[:, 2])

    times = np.arange(t0, tend, timeint)
    if times[-1] != tend:
        times = np.append(times, tend)

    xint = (xend - x0) / spacebins
    x = np.arange(x0, xend, xint)
    x = np.append(x, xend)

    X, Y = np.meshgrid(times, x)  # leave out the last point where making the grid to plot over

    speeds = []  # store lists of speeds here
    for i in range(len(times) - 1):  # minus 1 here because we have n+1 points but this defines only n bins.
        speeds.append([])
        for j in range(len(x) - 1):
            speeds[i].append([])

    veh = []  # store which vehicles are giving measurements to each bin of the data
    for i in range(len(times) - 1):
        veh.append([])
        for j in range(len(x) - 1):
            veh[i].append(set())

    for i in range(len(data)):  # sort all of the speeds
        curt = data[i, 1]
        curx = data[i, 2]
        curv = data[i, 3]
        curveh = data[i, 0]

        curtimebin = math.floor((curt - t0) / timeint)
        curxbin = math.floor((curx - x0) / xint)

        try:
            speeds[curtimebin][curxbin].append(curv)
            veh[curtimebin][curxbin].add(curveh)
        except IndexError:  # at most extreme point we want it to go into the last bin not into a new bin (which would contain only itself)
            if curxbin == len(x) - 1:
                curxbin += -1
            if curtimebin == len(times) - 1:
                curtimebin += -1
            speeds[curtimebin][curxbin].append(curv)
            veh[curtimebin][curxbin].add(curveh)

    meanspeeds = X.copy()  # initialize output
    for i in range(len(times) - 1):  # populate output
        for j in range(len(x) - 1):
            cur = speeds[i][j]
            if 'mean' in use_avg:  # choose whether to do arithmetic or harmonic mean. if bin is empty, we set the value as np.nan (not the same as math.nan which is default)
                cur = np.mean(cur)
                if math.isnan(cur):
                    cur = np.nan
            else:
                if len(cur) == 0:
                    cur = np.nan
                else:
                    cur = harmonic_mean(cur)

            meanspeeds[j, i] = cur

    cmap = cm.RdYlBu  # RdYlBu is probably the best colormap overall for this
    cmap.set_bad('white', 1.)  # change np.nan into white color

    fig, current_ax = plt.subplots(figsize=(12, 8))

    def my_callback(
            args):  # this can be used to modify the current shape (for example, to make it into a perfect parralelogram, or to round the points to the nearest bin, etc.)
        # this will make the shape into a parralelogram with horizontal top and bottom. the first two points define the left side length, and the third point defines
        # the length of the other side. So the fourth point is therefore unused.

        if region_shape != 'p':  # the callback function only does something when a parralelogram shape ('p') is requested
            return

            # note that you have to do a 4 sided shape
        testx = mytoggle_selector.RS._xs  # when shape is completed we get the coordinates of its vertices
        testy = mytoggle_selector.RS._ys

        # get x and y displacements of the first line
        lenx = testx[0] - testx[1]
        leny = testy[0] - testy[1]

        # set third point
        testy[2] = testy[1]
        # set last point to complete the parralogram
        testy[3] = testy[2] + leny
        testx[3] = testx[2] + lenx
        # redraw the shape
        mytoggle_selector.RS._draw_polygon()

        return

    vehhelper = copy.deepcopy(
        veh)  # vehhelper is going to keep track of what vehicles in each bin have not yet been plotted
    vehlist = []  # list of vehicle trajectories weve plotted
    lineobjects = []  # list of line objects weve plotted (note that we are including the option to remove all the things )

    vertlist = []  # list of the vertex corners

    def mytoggle_selector(event):
        # these keep track of which vehicles we are showing in the plot
        nonlocal vehhelper
        nonlocal vehlist
        nonlocal lineobjects
        # keep track of the areas of oscillation identified
        nonlocal vertlist
        if event.key in ['A', 'a']:  # this will select a vehicle trajectory in the bin to plot
            curx, cury = event.xdata, event.ydata  # get mouse position
            # translate the mouse position into the bin we want to use
            curtimebin = math.floor((curx - t0) / timeint)
            curxbin = math.floor((cury - x0) / xint)

            if curxbin == len(x) - 1:  # prevent index error
                curxbin += -1
            if curtimebin == len(times) - 1:  # prevent index error
                curtimebin += -1

            newveh = None
            curlen = len(vehhelper[curtimebin][curxbin])
            while newveh is None and curlen > 0:
                newveh = vehhelper[curtimebin][curxbin].pop()
                if newveh in vehlist:
                    continue
                else:
                    curlen = curlen - 1
                    vehlist.append(newveh)
                    temp = data[data[:, 0] == newveh]
                    indjumps = sequential(temp)
                    for i in range(len(indjumps) - 1):
                        plotx, ploty = temp[indjumps[i]:indjumps[i + 1], 1], temp[indjumps[i]:indjumps[i + 1], 2]
                        newline = plt.plot(plotx, ploty, 'C0', scalex=False, scaley=False)
                        lineobjects.append(newline)
                    plt.draw()  # this will force the line to show immediately
                    break  # break so we won't go to else

            else:
                #                print('no more vehicles to plot in this region, try another.')
                pass

        if event.key in ['D', 'd']:  # on this key press we remove all the trajectories
            # reset all the plotting stuff
            vehhelper = copy.deepcopy(
                veh)  # vehhelper is going to keep track of what vehicles in each bin have not yet been plotted
            vehlist = []  # list of vehicle trajectories weve plotted
            for i in range(len(lineobjects)):
                lineobjects[i][0].remove()  # remove all lines
            plt.draw()
            lineobjects = []

        if event.key in ['W', 'w']:  # put top side of the shape to the top
            testx = mytoggle_selector.RS._xs  # when shape is completed we get the coordinates of its vertices
            testy = mytoggle_selector.RS._ys

            lenx = testx[0] - testx[1]
            leny = testy[0] - testy[1]

            dy = xend - testy[0]
            dx = dy * lenx / leny
            # update top side
            testx[0] += dx
            testy[0] += dy
            testx[-2] += dx
            testy[-2] += dy
            testx[-1] = testx[0]
            testy[-1] = testy[0]

            mytoggle_selector.RS._draw_polygon()

        if event.key in ['X', 'x']:  # put bottom side of the shape to the bottom
            testx = mytoggle_selector.RS._xs  # when shape is completed we get the coordinates of its vertices
            testy = mytoggle_selector.RS._ys

            lenx = testx[0] - testx[1]
            leny = testy[0] - testy[1]
            theta = math.atan2(leny, lenx) - math.pi  # want the angle going in the other direction so minus pi
            dy = -x0 + testy[1]
            dx = dy * lenx / leny
            # update bottom side
            testx[1] += -dx
            testy[1] += -dy
            testx[2] += -dx
            testy[2] += -dy

            mytoggle_selector.RS._draw_polygon()

        if event.key == 'enter':  # enter means we are happy with the current region selected and want to choose another.
            print(mytoggle_selector.RS.verts)
            vertlist.append(mytoggle_selector.RS.verts)  # keep track of the previous region
            #            plt.plot(mytoggle_selector.RS._xs, mytoggle_selector.RS._ys, 'k-', linewidth = 2, alpha=.6)

            # start new polygonselector
            mytoggle_selector.RS = PolygonSelector(current_ax, my_callback,
                                                   lineprops=dict(color='k', linestyle='-', linewidth=2, alpha=0.4),
                                                   markerprops=dict(marker='o', markersize=2, mec='k', mfc='k',
                                                                    alpha=0.4))
            plt.connect('key_press_event', mytoggle_selector)
            plt.show()

        if event.key in ['N', 'n']:
            if len(mytoggle_selector.RS.verts) == 4:
                vertlist.append(mytoggle_selector.RS.verts)
#            selectvehID(data1, times, x, lane, veh, vertlist)
            selectvehID_v2(data1, times, x, lane, veh, vertlist)

    plt.pcolormesh(X, Y, meanspeeds,
                   cmap=cmap)  # pcolormesh is similar to imshow but is meant for plotting whereas imshow is for actual images
    plt.xlabel('Time (.1 s)')
    plt.ylabel('Space (ft)')
    cbar = plt.colorbar()  # colorbar
    cbar.set_label('Speed (ft/s)')

    print('a to plot a vehicle trajectory, d to clear all vehicle trajectories')
    print('click with mouse to set a corner that encloses a region of the data')
    print('enter to select a new region to identify, click + shift to drag, esc to start over with current region')
    print('w to snap top side of shape to top of data, x to snap bottom side of shape to bottom of data')
    print('when all regions of interest are identified, press n to move to next stage of the process')

    mytoggle_selector.RS = PolygonSelector(current_ax, my_callback,
                                           lineprops=dict(color='k', linestyle='-', linewidth=2, alpha=0.4),
                                           markerprops=dict(marker='o', markersize=2, mec='k', mfc='k', alpha=0.4))
    plt.connect('key_press_event', mytoggle_selector)
    plt.show()

    return times, x, lane, veh


def point2bin(xpoint, ypoint, x, y, x0, y0, xint, yint):
    # translates ypoint, xpoint into bin indices for bin edges y and x
    # ypoint - y coordinate of point
    # xpoint - x coordinate of point
    # y - list of edges for bin in y direction e.g. [0,1,2,3] = 3 bins, bin 0 = [0,1), bin1 = [1,2) etc.
    # x - list of edges for bin in x direction
    # y0 - lowest point in y bin
    # x0 - lowest point in x bin
    # yint - length of bins in y axis
    # xint - length of bins in x axis

    curybin = math.floor((ypoint - y0) / yint)
    curxbin = math.floor((xpoint - x0) / xint)

    if curxbin >= len(x) - 1:
        curxbin = len(x) - 2
    elif curxbin < 0:
        curxbin = 0
    if curybin >= len(y) - 1:
        curybin = len(y) - 2
    elif curybin < 0:
        curybin = 0

    return curxbin, curybin

#########################################

def get_all_vehicles_for_lane(meas, platooninfo, lane, start, end):
    veh_list = []
    for veh_id in meas:
        if lane in np.unique(meas[veh_id][:, 7]):
            if platooninfo[veh_id][2] < start or platooninfo[veh_id][1] > end:
                continue
            veh_list.append(veh_id)

    return veh_list


def selectvehID_v2(data, times, x, lane, veh, vertlist, platoon=None, vert=0):
    # data - data in raw form
    # times, x, lane, veh - these are all outputs from selectoscillation.
    # vertlist - list of verticies of region, each vertice is a tuple and there should usually be 4 corners.
    # platoon = None - can either automatically get intial vehicles or you can manually specify which vehicles should be in the initialization
    # vert = 0 -You can choose which vertice to start centered on.

    # outputs -
    # plot with 4 subplots, shows the space-time, speed-time, std. dev of speed, wavelet series of vehicles.
    # interactive plot can add vehicles before/after, select specific vehicles etc.

    # Retrieve meas and platooninfo from data
    meas, platooninfo = makeplatoonlist(data, form_platoons=False)

    if platoon is not None:
        veh_list = platoon
        xvert = []
        yvert = []

    else:
        # Get time limit based on vertlist
        timestamps = []
        for i in range(4):
            timestamps.append(vertlist[0][i][0])

        # Sort the list of vehicles based on the given lane
        all_veh_list = get_all_vehicles_for_lane(meas, platooninfo, lane, min(timestamps), max(timestamps))
        all_veh_list = sortveh3(all_veh_list, lane, meas, platooninfo)
        initial_index = len(all_veh_list) // 2
        veh_list = []
        veh_list.append(all_veh_list[initial_index])

        xvert = []
        yvert = []

    left_window_index = initial_index
    right_window_index = initial_index

    # Initiate plotting
    fig = plt.figure()
    ax1 = plt.subplot(2, 2, 1)
    ax2 = plt.subplot(2, 2, 2)
    ax3 = plt.subplot(2, 2, 3)
    ax4 = plt.subplot(2, 2, 4)

    scale = np.arange(15, 100)  # for wavelet transform
    indcounter = np.asarray([], dtype=np.int64)  # keeps track of which line2D artists correspond to which vehicles (for axis 1);
    indcounter4 = np.asarray([], dtype=np.int64)  # same thing for axis 4
    counter4 = 0

    # The centralized dictionary that stores {veh_id -> list of all 4 artists in ax1, ax2, ax3 and ax4}
    artist_dict = {}

    def plot_ax1(veh, meas, platooninfo, vehind):
        nonlocal artist_dict
        nonlocal indcounter

        LCind = generate_LCind(meas[veh], lane)

        x = meas[veh][:, 1]
        y1 = meas[veh][:, 2]
        y2 = meas[veh][:, 3]
        spdstd = np.asarray([])
        artist_dict[veh] = [None] * 4

        ax1_artist_list = []

        for i in range(len(LCind) - 1):
            kwargs = {}
            if meas[veh][LCind[i], 7] != lane:
                kwargs = {'linestyle': '--', 'alpha': .4}
            spdstd = np.append(spdstd, y2[LCind[i]:LCind[i + 1]])
            ax1_artist = ax1.plot(x[LCind[i]:LCind[i + 1]], y1[LCind[i]:LCind[i + 1]], 'C0', picker=5, **kwargs)
            ax1_artist_list.append(ax1_artist)
            indcounter = np.append(indcounter, vehind)

        artist_dict[veh][0] = ax1_artist_list
        spdstd = np.std(spdstd)

        return spdstd

    def plot_ax2(veh, meas, platooninfo):
        nonlocal artist_dict
        ax2.cla()
        LCind = generate_LCind(meas[veh], lane)

        x = meas[veh][:, 1]
        y2 = meas[veh][:, 3]

        ax2_artist_list = []

        for i in range(len(LCind) - 1):
            kwargs = {}
            if meas[veh][LCind[i], 7] != lane:
                kwargs = {'linestyle': '--', 'alpha': .4}
            ax2_artist = ax2.plot(x[LCind[i]:LCind[i + 1]], y2[LCind[i]:LCind[i + 1]], 'C0', picker=5, **kwargs)
            ax2_artist_list.append(ax2_artist)

        artist_dict[veh][1] = ax2_artist_list
        return

    def plot_ax3(veh, meas, platooninfo):
        nonlocal artist_dict
        ax3.cla()

        LCind = generate_LCind(meas[veh], lane)

        x = meas[veh][:, 1]
        y2 = meas[veh][:, 3]
        energy = wt(y2, scale)

        ax3_artist_list = []

        for i in range(len(LCind) - 1):
            kwargs = {}
            if meas[veh][LCind[i], 7] != lane:
                kwargs = {'linestyle': '--', 'alpha': .4}
            ax3_artist = ax3.plot(x[LCind[i]:LCind[i + 1]], energy[LCind[i]:LCind[i + 1]], 'C0', picker=5, **kwargs)
            ax3_artist_list.append(ax3_artist)

        artist_dict[veh][2] = ax3_artist_list
        return

    def plot_ax4(veh, meas, platooninfo, vehind, spdstd):
        nonlocal artist_dict
        nonlocal indcounter4
        nonlocal counter4

        spdstd = np.std(spdstd)
        if vehind < len(ax4.lines):  # something behind
            counter4 += -1
            ax4_artist = ax4.plot(counter4, spdstd, 'C0.', picker=5)
        else:
            ax4_artist = ax4.plot(vehind + counter4, spdstd, 'C0.', picker=5)
        artist_dict[veh][3] = ax4_artist
        indcounter4 = np.append(indcounter4, vehind)

        return

    # Plot the 4 subplots
    for veh_index, veh in enumerate(veh_list):
        spdstd = plot_ax1(veh, meas, platooninfo, veh_index)
        plot_ax2(veh, meas, platooninfo)
        plot_ax3(veh, meas, platooninfo)
        plot_ax4(veh, meas, platooninfo, veh_index, spdstd)

    zoomind = None
    indlist = []
    indlist4 = None
    plt.suptitle('Left click on trajectory to select vehicle')

    ax1.plot(xvert, yvert, 'k-', scalex=False, scaley=False, alpha=.4)  # draw box for trajectories

    def on_pick(event):
        # there is a bug: when there is only one line left, clicking would result in an error
        nonlocal artist_dict
        nonlocal zoomind
        nonlocal indlist
        nonlocal indlist4

        ax = event.artist.axes;
        curind = ax.lines.index(event.artist)

        if event.mouseevent.button == 3:  # right click zooms in on wavelet and speed series
            pass
        else:  # left click selects the trajectory to begin/end with
            if indlist4 != None:
                ax4.lines[indlist4].set_color('C0')
            if ax == ax1:
                curvehind = indcounter[curind]
                for ind, j in enumerate(indcounter4):
                    if j == curvehind:
                        ax4.lines[ind].set_color('C1')
                        break
                indlist4 = ind
            if ax == ax4:
                curvehind = indcounter4[curind]
                event.artist.set_color('C1')
                indlist4 = curind

            for i in indlist:  # reset old selection for ax4
                if i < len(ax1.lines):
                    ax1.lines[i].set_color('C0')

            for ind, j in enumerate(indcounter):
                if j == curvehind:
                    indlist.append(ind)
                    ax1.lines[ind].set_color('C1')

            veh = veh_list[curvehind]
            plt.suptitle('Vehicle ID ' + str(veh) + ' selected')
            plot_ax2(veh, meas, platooninfo)
            plot_ax3(veh, meas, platooninfo)

            fig.canvas.draw()
            return

    def key_press(event):
        nonlocal artist_dict
        nonlocal veh_list
        nonlocal indcounter
        nonlocal indcounter4
        nonlocal left_window_index
        nonlocal right_window_index

        if event.key in ['T', 't']:  # whether to show the trajectory box
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

        if event.key in ['P', 'p']:  # print current vehicle list
            print('current vehicles shown are ' + str(veh_list))
        if event.key in ['A', 'a']:  # add a vehicle before
            ax1.lines[-1].remove()  # for study area to be shown
            ax1.relim()

            if left_window_index > 0:
                indcounter += 1
                indcounter4 += 1

                left_window_index += -1
                new_veh = all_veh_list[left_window_index]
                veh_list.insert(0, new_veh)
                spdstd = plot_ax1(new_veh, meas, platooninfo, 0)
                plot_ax2(new_veh, meas, platooninfo)
                plot_ax3(new_veh, meas, platooninfo)
                plot_ax4(new_veh, meas, platooninfo, 0, spdstd)

                ax1.plot(xvert, yvert, 'k-', scalex=False, scaley=False, alpha=.4)
                plt.draw()

        if event.key in ['Z', 'z']:  # add a vehicle after
            ax1.lines[-1].remove()  # for study area to be shown
            ax1.relim()

            if right_window_index < len(all_veh_list) - 1:
                right_window_index += 1
                new_veh = all_veh_list[right_window_index]
                veh_list.append(new_veh)
                spdstd = plot_ax1(new_veh, meas, platooninfo, len(veh_list) - 1)
                plot_ax2(new_veh, meas, platooninfo)
                plot_ax3(new_veh, meas, platooninfo)
                plot_ax4(new_veh, meas, platooninfo, len(veh_list) - 1, spdstd)

                ax1.plot(xvert, yvert, 'k-', scalex=False, scaley=False, alpha=.4)  # redraw study area
                plt.draw()

        if event.key in ['D', 'd']:  # remove a vehicle before
            ax1.lines[-1].remove()  # for study area to be shown
            ax1.relim()

            if right_window_index > left_window_index:
                indcounter = np.delete(indcounter, np.where(indcounter == 0))
                indcounter += -1
                indcounter4 = np.delete(indcounter4, np.where(indcounter4 == 0))
                indcounter4 += -1
                veh_tbr = all_veh_list[left_window_index]
                left_window_index += 1
                veh_list.pop(0)
                ax1_artist_list = artist_dict[veh_tbr][0]
                for ax1_artist in ax1_artist_list:
                    for art in ax1_artist:
                        art.remove()
                        del art

                ax1.plot(xvert, yvert, 'k-', scalex=False, scaley=False, alpha=.4)  # redraw study area
                plt.draw()

        if event.key in ['C', 'c']:  # remove a vehicle after
            ax1.lines[-1].remove()  # for study area to be shown
            ax1.relim()

            if right_window_index > left_window_index:
                max_index_ax1 = np.max(indcounter)
                indcounter = np.delete(indcounter, np.where(indcounter == max_index_ax1))
                max_index_ax4 = np.max(indcounter4)
                indcounter4 = np.delete(indcounter4, np.where(indcounter4 == max_index_ax4))
                veh_tbr = all_veh_list[right_window_index]
                right_window_index += -1
                veh_list.pop()
                ax1_artist_list = artist_dict[veh_tbr][0]
                for ax1_artist in ax1_artist_list:
                    for art in ax1_artist:
                        art.remove()
                        del art

                ax1.plot(xvert, yvert, 'k-', scalex=False, scaley=False, alpha=.4)  # redraw study area
                plt.draw()

        if event.key in ['N', 'n']:  # next study area
            plt.close()
            selectvehID(data, times, x, lane, veh, vertlist, vert=vert + 1)

        return

    fig.canvas.callbacks.connect('pick_event', on_pick)
    plt.connect('key_press_event', key_press)
    return




#########################################



def selectvehID(data, times, x, lane, veh, vertlist, platoon=None, vert=0):
    """
    This function is very disorganized and there are a few bugs with how things are initialized 
    """
    # after selecting a region in selectoscillation, now we want to go through the inidividual trajectories so we can see
    # exactly where the congestion starts (which vehicle, what cause) as well as examine the congestion propagation in detail.

    # data - data in raw form
    # times, x lane, veh - these are all outputs from selectoscillation.
    # vertlist - list of verticies of region, each vertice is a tuple and there should usually be 4 corners.
    # platoon = None - can either automatically get intial vehicles or you can manually specify which vehicles should be in the initialization
    # vert = 0 -You can choose which vertice to start centered on.

    # outputs -
    # plot with 4 subplots, shows the space-time, speed-time, std. dev of speed, wavelet series of vehicles.
    # interactive plot can add vehicles before/after, select specific vehicles etc.

    # nice feature to add is pass in platoon and it will plot it but this requires some rework of the initialization; just need to initialize pre, post and then call
    # preposthelper, don't need to do other stuff during initialization.

    # note that you can use a custom picker to make it so you can pick exactly one artist per mouse click by checking the mouse coordinates,
    # then looking at the distance of the mouse to each artist, then only returning true for the closest artist.
    # alternatively, you can use built in picker=float option and then the artist will only be selected if mouse is within float points of the artist.
    # but this can mean you can select multiple artists per click if they are close together.

    # can just pass in meas and platooninfo if desired - make kwarg

    # need meas and only data was passed in
    meas, platooninfo = makeplatoonlist(data, form_platoons=False)

    if platoon is not None:
        newvehlist = platoon
        xvert = [];
        yvert = []

    else:
        t0 = times[0]
        x0 = x[0]
        timeint = times[1] - times[0]
        xint = x[1] - x[0]

        curvert = vertlist[math.floor(vert / 2)]  # first area

        xvert = [];
        yvert = []
        for i in curvert:
            xvert.append(i[0]);
            yvert.append(i[1])
        xvert.append(curvert[0][0]);
        yvert.append(curvert[0][1])

        curvertind = -2 * (vert % 2)
        ypoint, xpoint = curvert[curvertind][1], curvert[curvertind][0]  # top left corner of first area
        timebin, xbin = point2bin(xpoint, ypoint, times, x, t0, x0, timeint, xint)

        newvehlist = veh[timebin][xbin];
        newvehlist2 = []

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

    # basically we want a list of vehicles which will be added when we add vehicles before, and we need a list of vehicles which will be added when we add after.
    # also there should be some initial vehicles which populate the plots.
    # initialize post and pre vehicle lists

    prevehlist, curvehlist, unused = prelisthelper([], [], newvehlist, [], lane, meas, platooninfo)
    postvehlist, curvehlist, unused = postlisthelper([], [], newvehlist, [], lane, meas, platooninfo)
    # initial quick sort of curvehlist just so we know which one is the earliest vehicle
    curvehlist = sorted(list(curvehlist), key=lambda veh: platooninfo[veh][0])  # sort list depending on when they enter

    if False:  # So i'm not sure why this is here to begin with but it was messing things up in at least 1 instance so I have removed it.
        # I believe the point of this first part is to make it so the initialization contains a good amount of vehicles, but really there is no point in that.
        # as long as the initialization of curvehlist, prevehlist, postvehlist are correct, can make curvehlist bigger by adding entries in pre/post to it.

        # this region had a messed up initialization if this part is allowed to execute
        # test2 = [[(5562.474476963611, 1476.8050669428), (6311.045414797408, 164.0527611552), (7203.064516129032, 164.0527611552), (6454.493578295235, 1476.8050669428)]]
        # in lane 2 of the NGSim i-80 reconstructed data.
        # problem had to do with followers of leaders being added, but one of the followers of leaders was very far away from other vehicles, and this
        # messes up the distinction between pre, current, and post vehicles. Specifically with vehicle 1717 having leader 1757, 1748 is a follower of 1757.
        for ind, i in enumerate(
                prevehlist):  # iterate over prevehlist; everything after the earliest vehicle will be automatically put in
            if platooninfo[i][0] < platooninfo[curvehlist[0]][
                0]:  # if entry time is earlier than first vehicle in curvehlist
                continue
            else:
                newvehlist = prevehlist[ind:]
                prevehlist = prevehlist[:ind]

                prevehlist, curvehlist, postvehlist = prelisthelper(prevehlist, curvehlist, newvehlist, postvehlist,
                                                                    lane, meas, platooninfo)
                break

        postvehlist.reverse()  # reverse here
        for ind, i in enumerate(
                postvehlist):  # iterate over the postvehlist in the same way; everything before latest vehicle will be automatically put in
            # need to iterate over postvehlist in reverse order
            if platooninfo[i][0] > platooninfo[curvehlist[-1]][
                0]:  # if entry time is later than first vehicle (note used to be curvehlist[-1] last vehicle)
                continue
            else:
                newvehlist2 = postvehlist[ind:]
                postvehlist = postvehlist[:ind]
                postvehlist.reverse()  # reverse back here

                #            #check for duplicates #deprecated now we recognize overlap in newvehlist and newvehlist2 as a special case
                #            temp = []
                #            for j in newvehlist:
                #                if j in curvehlist:
                #                    continue
                #                else:
                #                    temp.append(j)
                #            newvehlist = temp

                postvehlist, curvehlist, prevehlist = postlisthelper(postvehlist, curvehlist, newvehlist2, prevehlist,
                                                                     lane, meas, platooninfo)
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

    curvehlist = sorted(list(curvehlist),
                        key=lambda veh: platooninfo[veh][0])  # quick sort list depending on when they enter
    curvehlist = sortveh3(curvehlist, lane, meas, platooninfo)  # full sort takes into account lane changing

    prevehlist, postvehlist = preposthelper(prevehlist, postvehlist, curvehlist, platooninfo)

    fig = plt.figure()
    ax1 = plt.subplot(2, 2, 1)
    ax2 = plt.subplot(2, 2, 2)
    ax3 = plt.subplot(2, 2, 3)
    ax4 = plt.subplot(2, 2, 4)
    #    axlist = [ax1,ax2,ax3,ax4]
    axlist = [ax1, ax4]

    scale = np.arange(15, 100)  # for wavelet transform
    indcounter = np.asarray([],
                            dtype=np.int64)  # keeps track of which line2D artists correspond to which vehicles (for axis 1);
    indcounter4 = np.asarray([], dtype=np.int64)  # same thing for axis 4
    counter = 0

    # e.g. indcounter = [0,0,0,1,1,1] #first 3 artists of ax.lines correspond to curvehlist[0], next three correspond to curvehlist[1]

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

    def plothelper(veh, meas, platooninfo, vehind):
        # new helper function will do plotting for
        nonlocal indcounter
        nonlocal indcounter4
        nonlocal counter
        LCind = np.diff(meas[veh][:, 7])
        LCind = np.nonzero(LCind)[0] + 1;
        LCind = list(LCind)  # indices where lane changes
        LCind.insert(0, 0);
        LCind.append(len(meas[veh][:, 7]))  # append first and last index so we have all the intervals

        x = meas[veh][:, 1]
        y1 = meas[veh][:, 2]
        y2 = meas[veh][:, 3]
        spdstd = np.asarray([])

        for i in range(len(LCind) - 1):
            kwargs = {}
            if meas[veh][LCind[i], 7] != lane:
                kwargs = {'linestyle': '--', 'alpha': .4}
            spdstd = np.append(spdstd, y2[LCind[i]:LCind[i + 1]])
            ax1.plot(x[LCind[i]:LCind[i + 1]], y1[LCind[i]:LCind[i + 1]], 'C0', picker=5, **kwargs)
            indcounter = np.append(indcounter, vehind)

        #        ax1.plot(x,y1,'C0-',picker=5)

        spdstd = np.std(spdstd)
        if vehind < len(ax4.lines):  # something behind
            counter += -1
            ax4.plot(counter, spdstd, 'C0.', picker=5)
        else:
            ax4.plot(vehind + counter, spdstd, 'C0.', picker=5)
        indcounter4 = np.append(indcounter4, vehind)

        plot2helper(veh, meas, platooninfo)

        #        wtstd = np.std(energy)
        #        ax42 = ax4.twinx()
        #        ax42.plot(indcounter,wtstd,'k.')
        return

    def plot2helper(veh, meas, platooninfo):
        ax2.cla();
        ax3.cla()

        LCind = np.diff(meas[veh][:, 7])
        LCind = np.nonzero(LCind)[0] + 1;
        LCind = list(LCind)  # indices where lane changes
        LCind.insert(0, 0);
        LCind.append(len(meas[veh][:, 7]))  # append first and last index so we have all the intervals
        x = meas[veh][:, 1]
        y2 = meas[veh][:, 3]
        energy = wt(y2, scale)

        for i in range(len(LCind) - 1):
            kwargs = {}
            if meas[veh][LCind[i], 7] != lane:
                kwargs = {'linestyle': '--', 'alpha': .4}
            ax2.plot(x[LCind[i]:LCind[i + 1]], y2[LCind[i]:LCind[i + 1]], 'C0', picker=5, **kwargs)
            ax3.plot(x[LCind[i]:LCind[i + 1]], energy[LCind[i]:LCind[i + 1]], 'C0', picker=5, **kwargs)

        return

    for count, i in enumerate(curvehlist):
        plothelper(i, meas, platooninfo, count)

    zoomind = None
    indlist = []
    indlist4 = None
    #    ax2ylim, ax2xlim = ax2.get_ylim, ax2.get_xlim
    #    ax3ylim, ax3xlim = ax3.get_ylim, ax3.get_xlim
    plt.suptitle('Left click on trajectory to select vehicle')

    #    indkey = np.array(range(len(curvehlist))) #indkey is list of the order of vehicles in curvehlist  #deprecated

    ax1.plot(xvert, yvert, 'k-', scalex=False, scaley=False, alpha=.4)  # draw box for trajectories

    #    autoscale_based_on(ax1,ax1.lines[:-1])

    def on_pick(event):
        nonlocal zoomind
        nonlocal indlist
        nonlocal indlist4

        ax = event.artist.axes;
        curind = ax.lines.index(event.artist)

        if event.mouseevent.button == 3:  # right click zooms in on wavelet and speed series
            pass
        else:  # left click selects the trajectory to begin/end with
            #            print(indcounter)
            if indlist4 != None:
                ax4.lines[indlist4].set_color('C0')
            if ax == ax1:
                curvehind = indcounter[curind]
                for ind, j in enumerate(indcounter4):
                    if j == curvehind:
                        ax4.lines[ind].set_color('C1')
                        break
                indlist4 = ind
            if ax == ax4:
                curvehind = indcounter4[curind]
                event.artist.set_color('C1')
                indlist4 = curind

            for i in indlist:  # reset old selection for ax4
                ax1.lines[i].set_color('C0')

            for ind, j in enumerate(indcounter):
                if j == curvehind:
                    indlist.append(ind)
                    ax1.lines[ind].set_color('C1')

            #            for i in axlist:
            #                if selectind != None:
            #                    i.lines[selectind].set_color('C0')
            #                i.lines[curind].set_color('C1')

            veh = curvehlist[curvehind]
            plt.suptitle('Vehicle ID ' + str(veh) + ' selected')
            plot2helper(veh, meas, platooninfo)

            fig.canvas.draw()
            return

    def key_press(event):
        nonlocal curvehlist
        nonlocal prevehlist
        nonlocal postvehlist
        nonlocal indcounter
        nonlocal indcounter4
        # toggle opaque trajectories
        if event.key in ['T', 't']:
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

        if event.key in ['P', 'p']:  # print current vehicle list
            #            print(prevehlist)
            #            print(postvehlist)
            print('current vehicles shown are ' + str(curvehlist))
        if event.key in ['A', 'a']:  # add a vehicle before

            count = -1
            while prevehlist[count] in curvehlist:  # prevent a vehicle already in from being added
                count += -1
            ax1.lines[-1].remove()  # for study area to be shown
            ax1.relim()

            indcounter = indcounter + 1
            indcounter4 = indcounter4 + 1
            plothelper(prevehlist[count], meas, platooninfo, 0)
            prevehlist, curvehlist, postvehlist = prelisthelper(prevehlist[:count], curvehlist, [prevehlist[count]],
                                                                postvehlist, lane, meas, platooninfo)

            ax1.plot(xvert, yvert, 'k-', scalex=False, scaley=False, alpha=.4)
            plt.draw()

        #            print(indcounter)

        if event.key in ['Z', 'z']:  # add a vehicle after

            count = 0
            while postvehlist[count] in curvehlist:  # prevent a vehicle already in from being added
                count += 1
            ax1.lines[-1].remove()  # for study area to be shown
            ax1.relim()

            plothelper(postvehlist[count], meas, platooninfo, len(curvehlist))
            postvehlist, curvehlist, prevehlist = postlisthelper(postvehlist[count + 1:], curvehlist,
                                                                 [postvehlist[count]], prevehlist, lane, meas,
                                                                 platooninfo)

            ax1.plot(xvert, yvert, 'k-', scalex=False, scaley=False, alpha=.4)  # redraw study area
            plt.draw()

        if event.key in ['N', 'n']:  # next study area
            plt.close()
            selectvehID(data, times, x, lane, veh, vertlist, vert=vert + 1)

        return

    fig.canvas.callbacks.connect('pick_event', on_pick)
    plt.connect('key_press_event', key_press)

    return


def preposthelper(prevehlist, postvehlist, curvehlist, platooninfo):
    # takes curvehlist - current vehicles
    # postvehlist - vehicles that are potentially to be added as next vehicles in platoon (in lane)
    # prevehlist - vehicles that are potentially be added as previous vehicles in platoon (in lane)
    # platooninfo

    # looks at pre and post lists, and put all vehicles in pre that occur after Tn in post, and all vehicles in post that occur before Tn in pre
    # this prevents vehicles which look like they should be in post being in pre and vice versa. Useful during initialization.

    tn = platooninfo[curvehlist[0]][0]
    Tn = platooninfo[curvehlist[-1]][0]
    pretemp = reversed(prevehlist)
    posttemp = postvehlist.copy()
    prechange = False
    postchange = False
    prenew = []
    postnew = []

    for i in pretemp:  # iterate over pre vehicles backwards
        if platooninfo[i][0] > Tn:  # if they enter later than Tn we want to move them to post
            postchange = True  # need to resort post
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
    # update lists
    if prechange:
        for i in prenew:
            if i not in prevehlist:
                prevehlist.append(i)
        prevehlist = sorted(prevehlist, key=lambda veh: platooninfo[veh][0])
    if postchange:
        for i in postnew:
            if i not in postvehlist:
                postvehlist.append(i)
        postvehlist = sorted(postvehlist, key=lambda veh: platooninfo[veh][0])

    return prevehlist, postvehlist


def prelisthelper(prevehlist, curvehlist, newvehlist, postvehlist, lane, meas, platooninfo, check=True):
    # given a list of new vehicles added, get all of the candidate pre vehicles added, add new vehicles to the current veh list, and call prehelper on the prevehlist

    #    print('new vehicles are '+str(newvehlist))
    for i in newvehlist:
        if i in postvehlist and check:  # if new vehicle to be added is also in postvehlist we need to update that list as well as the prevehlist
            temp = curvehlist.copy()
            postvehlist, temp, temp = postlisthelper(postvehlist, temp, [i], prevehlist, lane, meas, platooninfo,
                                                     check=False)  # update postvehlist

        curvehlist.insert(0, i)
        curleadlist = platooninfo[i][4]
        prevehlist.extend(curleadlist)
        for j in curleadlist:
            prevehlist.extend(platooninfo[j][-1])

    #    print('all candidates are '+str(prevehlist))
    prevehlist = prehelper(prevehlist, curvehlist, lane, meas, platooninfo)
    #    print('filtered candidates are '+str(prevehlist))

    return prevehlist, curvehlist, postvehlist


def postlisthelper(postvehlist, curvehlist, newvehlist, prevehlist, lane, meas, platooninfo, check=True):
    # given a list of new vehicles added, get all of the candidate pre vehicles added, add new vehicles to the current veh list, and call prehelper on the prevehlist

    #    print('new vehicles are '+str(newvehlist))
    for i in newvehlist:
        if i in prevehlist and check:
            temp = curvehlist.copy()
            prevehlist, temp, temp = prelisthelper(prevehlist, curvehlist, [i], postvehlist, lane, meas, platooninfo,
                                                   check=False)
        curvehlist.append(i)
        curfollist = platooninfo[i][-1]
        postvehlist.extend(curfollist)
        for j in curfollist:
            postvehlist.extend(platooninfo[j][4])

    #    print('all candidates are '+str(prevehlist))
    postvehlist = prehelper(postvehlist, curvehlist, lane, meas, platooninfo)
    #    print('filtered candidates are '+str(prevehlist))

    return postvehlist, curvehlist, prevehlist


def prehelper(prevehlist, curvehlist, lane, meas, platooninfo):
    # take prevehlist of candidate vehicles to previously add.
    # return prevehlist which is a sorted list of order vehicles should be added in .
    # it's called prehelper but actually it works for the postvehlist as well.

    prevehlist = list(set(prevehlist))  # remove duplicates
    temp = []
    for i in prevehlist:
        if i in curvehlist:
            continue
        if lane not in np.unique(meas[i][:, 7]):
            continue
        temp.append(i)
    prevehlist = temp
    prevehlist = sorted(list(prevehlist), key=lambda veh: platooninfo[veh][0])

    return prevehlist


def autoscale_based_on(ax,
                       lines):  # doesn't work right see selectvehID you can remove line reset window redraw line this is the solution we're using
    ax.dataLim = mtransforms.Bbox.unit()
    for line in lines:
        xy = np.vstack(line.get_data()).T
        ax.dataLim.update_from_data_xy(xy, ignore=False)
    ax.autoscale_view()


#######these used to be in the old simulation.py file
###nothing special but the hd has some code to draw arrows which is actually somewhat nice so you can see what direction the loops are going in 
def rediff(x, dt, end=True):
    # first order forward difference on x
    # input - array of values x

    # output - array of numerically differentiated dx which is 1 length shorter
    # if end is true, we will return the last value twice so it's the same length.

    out = []
    for i in range(len(x) - 1):
        new = (x[i + 1] - x[i]) / dt
        out.append(new)

    if end:
        out.append(out[-1])
    return out


def hd(lead, fol, arrowinterval=.5):
    N = len(lead.x)  # number measurements
    hd = []
    for i in range(N):
        cur = lead.x[i] - fol.x[i] - lead.len
        hd.append(cur)

    plt.plot(hd, fol.dx)

    # \\ TO DO \\ ##
    # a possible improvement here would be to record the length of the trjaecotry and then choose the arrowinterval and
    # arrow size based on that length so that you don't have to manually specify the arrowinterval

    counter = 0
    arrlen = .1
    arroffset = 13 * math.pi / 16
    for i in range(N - 1):
        dy = fol.dx[i + 1] - fol.dx[i]
        dx = hd[i + 1] - hd[i]
        counter = counter + (dy ** 2 + dx ** 2) ** .5  # keep track of length traveled
        if counter > arrowinterval:  # if its time to draw another arrow
            counter = 0  # reset counter
            theta = math.atan2(dy, dx)  # angle at which arrow will point
            arr1dx = arrlen * math.cos(theta - arroffset)
            arr2dx = arrlen * math.cos(theta + arroffset)
            arr1dy = arrlen * math.sin(theta - arroffset)
            arr2dy = arrlen * math.sin(theta + arroffset)
            plt.plot([hd[i], hd[i] + arr1dx], [fol.dx[i], fol.dx[i] + arr1dy], 'k-')
            plt.plot([hd[i], hd[i] + arr2dx], [fol.dx[i], fol.dx[i] + arr2dy], 'k-')

    return hd


def vehplot(universe, interval=1, option = False):
    N = len(universe)
    plt.subplot(1, 2, 1)
    for i in range(0, N, interval):
        if option:
            if i in np.arange(1, N,10):
                plt.plot(universe[i].x, 'C1')
            else: 
                plt.plot(universe[i].x, 'C0')
        else: 
            plt.plot(universe[i].x, 'C0')
    plt.ylabel('space')
    plt.xlabel('time')
    plt.yticks([])
    plt.xticks([])
    plt.subplot(1, 2, 2)
    for i in range(0, N, interval):
        if option:
            if i in np.arange(1, N,10):
                plt.plot(np.asarray(universe[i].dx) - 2 * i, 'C1')
            else:
                plt.plot(np.asarray(universe[i].dx) - 2 * i, 'C0')
        else:
            plt.plot(np.asarray(universe[i].dx) - 2 * i, 'C0')
    plt.ylabel('speed')
    plt.xlabel('time')
    plt.yticks([])
    plt.xticks([])

    return


def stdplot(universe, customx=None):
    N = len(universe)
    y = []
    for i in range(N):
        y.append(np.std(universe[i].dx))
    if customx == None:
        plt.plot(y, 'k.')
    else:
        plt.plot(customx, y, 'k.')

    return


#old version of animatevhd
#def animatevhd_list(meas, sim, platooninfo, my_id, lentail=20, h=.1, datalen=9, timerange=[None, None], delay=0):
#    # plot multiple vehicles in phase space (speed v headway)
#    # my_id - id of the vehicle to plot
#    # lentail = 20 number of observations to show in the past
#    # h = .1 - data discretization
#    # datalen = 9
#    # timerange = [usestart, useend]
#    # delay = 0 - gets starting time for newell model
#    fig = plt.figure()
#    plt.xlabel('space headway (ft)')
#    plt.ylabel('speed (ft/s)')
#    plt.title('space-headway for vehicle ' + " ".join(list(map(str, (my_id)))))
#    line_data = {}
#    id2Line = {}
#    
#    # If sim is None, only plot one set of data
#    plotOne = False
#    if sim is None:
#        plotOne = True
#        
#    x_min_lim = 1e10
#    y_min_lim = 1e10
#    x_max_lim = 0
#    y_max_lim = 0
#    
#    # 0: tnstar, 1: tn, 2: t
#    for veh_id in my_id:
#
#        t_nstar, t_n, T_nm1, T_n = platooninfo[veh_id][0:4]
#        
#        # Compute and validate start and end time
#        start, end = compute_validate_time(timerange, t_n, T_nm1, h=.1, delay=0)
#        
#        # animation in the velocity headway plane
#        leadinfo, folinfo, rinfo = helper.makeleadfolinfo([veh_id], platooninfo, meas, relaxtype = 'none')
#                
#        frames = [t_n, T_nm1]
#        relax, unused = r_constant(rinfo[0], frames, T_n, None, False, h)  # get the relaxation amounts for the current vehicle; these depend on the parameter curp[-1] only.
#
#        trueheadway = compute_headway(t_nstar, t_n, T_n, datalen, leadinfo, start, meas, veh_id, relax)
#        if not plotOne: 
#            headway = compute_headway(t_nstar, t_n, T_n, datalen, leadinfo, start, sim, veh_id, relax)
#        
#        for i in range(len(trueheadway) - lentail - (T_n - end)):
#            if plotOne:
#                sim_line = None
#                sim_label = None
#            else:
#                sim_line, sim_label, sim_x_min, sim_y_min, sim_x_max, sim_y_max = compute_line_data(headway, i, lentail, sim, veh_id, t_n - t_nstar)
#                x_min_lim = min(x_min_lim, sim_x_min)
#                y_min_lim = min(y_min_lim, sim_y_min)
#                x_max_lim = max(x_max_lim, sim_x_max)
#                y_max_lim = max(y_max_lim, sim_y_max)
#            
#            meas_line, meas_label, meas_x_min, meas_y_min, meas_x_max, meas_y_max = compute_line_data(trueheadway, i, lentail, meas, veh_id, t_n - t_nstar)
#            x_min_lim = min(x_min_lim, meas_x_min)
#            y_min_lim = min(y_min_lim, meas_y_min)
#            x_max_lim = max(x_max_lim, meas_x_max)
#            y_max_lim = max(y_max_lim, meas_y_max)
#
#            if i + start in line_data.keys():
#                line_data[i + start].append((sim_line, meas_line, sim_label, meas_label, veh_id))
#            else:
#                line_data[i + start] = [(sim_line, meas_line, sim_label, meas_label, veh_id)]
#    
#    ####plotting
#
#    ax = plt.gca()    
#    ax.set_xlim(x_min_lim - 10, x_max_lim + 10)
#    ax.set_ylim(y_min_lim - 10, y_max_lim + 10)
#    sortedKeys = list(sorted(line_data.keys()))
#    curLines = []
#
#    def init():
#        # Clean up, takes in effect when the animation starts to repeat
#        for veh_id in id2Line:
#            sim_line, meas_line, sim_annotation, meas_annotation, vehicle_id = id2Line[veh_id]
#            if not plotOne:
#                sim_line.set_data([],[])
#                sim_annotation.set_text("")
#            meas_line.set_data([],[])
#            meas_annotation.set_text("")
#            curLines.remove(vehicle_id)
#            del id2Line[vehicle_id]
#        return
#    
#    def aniFunc(frame):
#        allLines = line_data[sortedKeys[frame]]
#
#        for line in allLines:
#            veh_id = line[4]
#            # Check if veh_id has already been plotted in the last frame
#            if veh_id in curLines:
#                # If yes, fetch existing lines and annotations and modify
#                sim_line, meas_line, sim_annotation, meas_annotation, vehicle_id = id2Line[veh_id]
#
#                # In order to remove horizontal lines when the leader changes,
#                # need to detect leader change here and separate data into two groups
#                # Need to create new line and annotation for the new group
#                # We'll call a function that processes (line[0][0], line[0][1]) and (line[1][0], line[1][1])
#                # which returns a list of xy-coordinates
#                # The size of the list determines how many groups of data it got separated
#                # Essentially, if the list size > 1, there is a leader change
#
#                if not plotOne:
#                    sim_line.set_data(line[0][0], line[0][1])
#                    sim_annotation.set_position((line[2][0], line[2][1]))
#
#                meas_line.set_data(line[1][0], line[1][1])
#                meas_annotation.set_position((line[3][0], line[3][1]))
#            else:
#                # If no, plot new lines and annotations
#                if plotOne:
#                    sim_line = None
#                    sim_annotation = None
#                else:
#                    sim_line, = ax.plot(line[0][0], line[1][1], 'C1')
#                    sim_annotation = ax.annotate(str(math.floor(veh_id)), (line[2][0], line[2][1]), fontsize=7)
#                
#                meas_line, = ax.plot(line[1][0], line[1][1], 'C0')
#                meas_annotation = ax.annotate(str(math.floor(veh_id)), (line[3][0], line[3][1]), fontsize=7)
#                
#                # Save lines and annotations
#                id2Line[veh_id] = (sim_line, meas_line, sim_annotation, meas_annotation, veh_id)
#                curLines.append(veh_id)
#        return
#
#    im_ani = animation.FuncAnimation(fig, aniFunc, init_func=init, frames=len(sortedKeys), interval=10)
#    plt.show()
#    return im_ani
