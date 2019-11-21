# -*- coding: utf-8 -*-
"""
Created on Wed May 22 17:39:02 2019
Need to do more testing, but I have developed the following theory - 
1. the size of the perturbation is completely irrelevant 
2. what is important is the modulus of the transfer function
3. linear analysis is only quantiatively correct when the transfer function is very close to unity 
4. thus for some arbitrary, realistic perturbation, which has a whole spectrum of fourier modes, 
being applied to a realistic model, 
it is almost certain that virtually all the modes will be far from unity and the growth of congestion is not quantitative
5. also this implies that multispecieis analysis is not correct. 


-note that there are two different ways to think about the linear analysis: in terms of transfer function or in terms 
of eigenvalues. 
-energy gets spread around the modes so linear stability analysis is not terribly useful 
-need to examine criteria for stability of multiclass traffic and see what existing work for quantitatively correct
oscillation growth has been given 
-some funny business going on as well with same magnitude perturbations but different frequencies giving different results. 
-recall that different qualitative (i.e. long run) behavior is only supposed to happen from the size of perturbations. 

@author: rlk268
"""
from havsim.simulation.simulationold import * 
from havsim.simulation.modelsold import * 
from havsim.plotting import vehplot, hd

#simulation results from large simulation 
#with open('C:/Users/rlk268/OneDrive - Cornell University/important misc/pickle files/simeg33.33-1.1-2-.9-1.5IDMb3.pkl','rb') as f:
#    data = pickle.load(f)
    
#with open('C:/Users/rlk268/OneDrive - Cornell University/important misc/pickle files/simeg33.33-1.5-2-1.1-1.5IDM_b3.pkl','rb') as f:
#    data = pickle.load(f)
#%%


#create speed profile for the leader
#CHOOSE A DISTURBANCE TYPE FROM THE COMMENTS BELOW. DEFAULT EQUILIBRIUM SPEED IS 30 FT/S. CHANGE BY CHANGING SPEEDOFFSET. 
#CHOOSE SIMLEN AND N TO MAKE BIGGER SIMULATIONS WHICH TAKE LONGER TO RUN 
#CAN ALSO CHANGE THE MODEL USED AND THE MODEL PARAMETERS. 
simlen = 33000 #33000
N = 800 #800
speedoffset = -15
length = 5
velocity = [30 for i in range(simlen)]
velocity[0:200] = [1/10*(.1*(i)-10)**2+20 for i in range(200)] #polynomial disturbance 
#velocity[0] = 30.1
#velocity[100:150] = [28 for i in range(50)]

#velocity[2300:3300] = [18 for i in range(1000)]

#velocity[0:5] = np.linspace(30,28.9,5) #linear disturbance
#velocity[5:10] = np.linspace(28.9,30,5)

#velocity[50:250] = [-25*math.e**(-1e-3*(i+50-150)**2)+30 for i in range(200)] #exponential disturbance 
#velocity[50:150] = [25 for i in range(100)] #constant speed 

#velocity[50:100] = [i for i in np.linspace(30,5,50)]
#velocity[100:150] = [5 for i in range(50)]
#velocity[150:200] = list(np.linspace(5,30,50)) #piecewise speeds disturbance 

#unm1 = lambda t: -25*math.e**(-1e-3*(t-150)**2) + 30 #exponential squared 
#unm1 = lambda t: math.e**(-.01*(t-150))*.1*(t-150)**2 #polynomial
#velocity = [unm1(i) for i in range(simlen)] #disturbance is defined by anonymous function unm1



velocity = np.asarray(velocity) + speedoffset

t = np.arange(0,100,.1)

#inconsistencies for IDM 
#this grows as expected
#velocity =np.sin(2*math.pi*t/100)+30
#this decays even though it shouldn't. 
#velocity =np.sin(2*math.pi*t/5)+30

#for OVM 
#velocity =np.sin(2*math.pi*t/100)+30
#this decays even though it shouldn't. 
#velocity =np.sin(2*math.pi*t/10)+30


v1 = leadvehicle(0,0,length,velocity,.1)
universe = [v1]



#create followers


#IDM ####################
#p = [33.33,1.5,2,1.1,1.5] #33.33, 1.5, 2, 1.1, 1.5 33.33-1.1-2-.9-1.5
#p=[33.33, 1.2, 2, .9, 1.5]
p = [33.33,1.1,2,.9,1.5]
p2 = res2['x']
#p2[0] = p2[0]*4/3
headway = eql(IDM_b3,30+speedoffset,p, length)
headway2 = eql(linearCAV,30+speedoffset,p2,length)
prev = 0
perturb = 2
indlist = np.arange(0, N,10)
#indlist = []
for i in range(N):
    if i in indlist:
        prev = prev - headway2
        newveh = vehicle([prev,30+speedoffset],0,length,universe[-1],linearCAV,p2,.1)
        universe.append(newveh)
    else:
        prev = prev - headway
        newveh = vehicle([prev,30+speedoffset],0,length,universe[-1],IDM_b3,p,.1)
        universe.append(newveh)
    
#OVM  ##################
#p =  [10*3.3,.086/3.3, 1.545, 1.4, .175 ]
##p = [38.32417637,  0.07614446,  1.5      ,  2        ,  .5]
#maxspeed = optimalvelocity(10000,p)
#headway = eql(OVM_b,30+speedoffset,p, length)
#prev = 0
#for i in range(N):
#    prev = prev - headway
#    newveh = vehicle([prev,30+speedoffset],0,length,universe[-1],OVM_b,p,.1)
#    universe.append(newveh)
#    

#OVM_l
#jam distance, velocity-headway slope (lower = more sensitive), sensitivity parameter (higher = stronger adjustment), velocity cap  
#p =  [20, 1.9,  1, 100, 1] #20, .7, 1.4 60 


#p = [30,.4, 1.4, 90]
#headway = eql(OVM_lb,30+speedoffset,p,length)
#prev = 0
#for i in range(N):
#    prev = prev - headway
#    newveh = vehicle([prev,30+speedoffset],0,length,universe[-1],OVM_lb,p,.1)
#    universe.append(newveh)



#run the simulation and then plot stuff
simulate(universe,simlen-1)

plt.close('all')
plt.figure()
vehplot(universe,interval = 2)
plt.figure()
hd(universe[2],universe[3])
#plt.figure()
#stdplot(universe)

#%% test example of finding autonomous vehicle best strategy 

#from simulation import * 
#model= OVM_lb2 
#model = IDM_b3
#prate = [0,4,5,9,14,15,19,24,25,29]
#testobj = egobj(p,p,model,headway,prate,simlen,N,velocity,30+speedoffset,.1)
#args = (p,model,headway,prate,simlen,N,velocity,30+speedoffset,.1)
##bounds = [(5,40),(.3,2),(1,5),(40,60)] #ovm_lb2
#bounds = [(10,40),(0.1,2),(0.1,3),(.1,3),(.1,3)] #idm_b3
#bfgs = sc.fmin_l_bfgs_b(egobj,p,None,args,1,bounds,maxfun=300)
#
##%test the strategy you find
#
#universe, obj = eguni(bfgs[0],p,model,headway,prate,simlen,N,velocity,30+speedoffset,.1)
#plt.figure()
#vehplot(universe)


#%% #all of these plots need the transfer function fixed to make sense 
#from simulation import * 
#
#fn = xferOVM_l2
#test, test2, test3 = linearapprox2(universe[1],headway,30,p,fn)
#plt.figure()
#
##testdx = rediff(test,.1)
##plt.plot(testdx)
#
#plt.plot(test)
#plt.plot(universe[1].dx)
#plt.plot(universe[0].dx)
#plt.legend(['follower predicted','follower actual','leader'])
#
#plt.figure()
#plt.plot(np.abs(test2[:50]))
#plt.plot(np.abs(test3[:50]))
#actual = np.fft.rfft(np.array(universe[1].dx)-30)
#plt.plot(np.abs(actual[:50]))
#plt.legend(['leader','follower predicted','follower actual'])
#
#plt.figure()
#pltplt =[]
#for i in range(len(test)-500):
#    pltplt.append(fn(headway,30,p,2*math.pi*i/len(test)))
##    pltplt.append(xferIDM(headway,30,p,2*math.pi/(i+1)))
#plt.plot(np.abs(pltplt))
#actualxfer = np.divide(actual,test2)
#plt.plot(np.abs(actualxfer))
#plt.legend(['xfer fn','actual xfer'])

#invfft = np.fft.irfft(test2)
#invfft = rediff(invfft,.1) + 30*np.ones(len(invfft))
#plt.plot(invfft+30)

#%%
#test2 = np.fft.rfft(np.array(velocity)-30)
#new = []
#new2 = []
##fac1 = 2**.5
##fac2 = (-1+0j)
#fac1 = xferIDM(headway,30,p,2*math.pi*10/1000)
#fac2 = xferIDM(headway,30,p,2*math.pi*11/1000)
#
#for i in test2: 
#    new.append(i*fac1)
##    new2.append(i*fac2*abs(fac1)/abs(fac2))
#    new2.append(i*fac2)
#
#plt.close('all')
#plt.figure()
#plt.plot(np.fft.irfft(test2))
#plt.plot(np.fft.irfft(new))
#plt.plot(np.fft.irfft(new2))
    
#%%

#def displacement(universe):
plt.figure()
testveh = universe[-1]
testdx = testveh.dx
dt = testveh.dt
#testdx = data[0][400]
#speedoffset = -10
displacement = [0]
for i in range(len(testdx)):
    d = (testdx[i]-30-speedoffset)*dt
    displacement.append(displacement[-1]+d)
plt.plot(displacement)

#extimes = list(range(0,600))
#exlen = 1/len(extimes)
#ex = 0
#for i in extimes: 
#    ex += displacement[i]
#ex = ex*exlen 
#print('expected displacement is '+str(ex))


#%%
#plt.figure()
#displacementlist = []
#for j in range(len(universe)):
#    testveh = universe[j]
#    testdx = testveh.dx
#    dt = testveh.dt
#    displacement = [0]
#    for i in range(len(testdx)):
#        d = (testdx[i]-30-speedoffset)*dt
#        displacement.append(displacement[-1]+d)
#        
#    displacementlist.append(displacement[-1])
#plt.plot(displacementlist, 'k.')
        

#%%
#dt = .1

#plt.close('all')
#plt.plot(data[0][10])


def myplot():
    indlist = np.arange(0,150,15)
    plt.figure()
    for i in indlist:
        plt.plot(universe[i].dx)

myplot()


#%%
#save simulation results 
data = []
for i in universe:
    data.append(i.dx)
with open('C:/Users/rlk268/OneDrive - Cornell University/important misc/pickle files/universenoAV.pkl','wb') as f:
    pickle.dump(data,f)