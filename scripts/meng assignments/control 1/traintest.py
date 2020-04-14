#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from drl import *
#%% initialize agent (we expect the agent to be awful before training)
#env = gym.make('CartPole-v0')
#model = Model(num_actions=env.action_space.n)
#agent = ACagent(model)
#testenv = gym_env(env)
##%%
#agent.test(testenv,200) #200 timesteps
#print('total reward before training is '+str(testenv.totloss)+' starting from initial with 200 timesteps')
##%%
#    #MWE of training
#rewards = agent.train(testenv)
#plt.plot(rewards)
#plt.ylabel('rewards')
#plt.xlabel('episode')
#agent.test(testenv,200)
#print('total reward is '+str(testenv.totloss))
''' 
gym_env is flexible to pass in other environments, including MountainCar-V0
for MountainCar:
reward := -1 for each time step, until the goal position of 0.5 is reached
The episode ends when you reach 0.5 position, or if 200 iterations are reached.
'''
    #%%
#                    #specify simulation
p = [33.33, 1.2, 2, 1.1, 1.5] #parameters for human drivers
initstate, auxinfo, roadinfo = eq_circular(p, IDM_b3, update2nd_cir, IDM_b3_eql, 41, length = 2, L = None, v = 15, perturb = 2) #create initial state on road
sim, curstate, auxinfo = simulate_cir(initstate, auxinfo,roadinfo, update_cir, timesteps = 25000, dt = .25)
vlist = {i: curstate[i][1] for i in curstate.keys()}
avid = min(vlist, key=vlist.get)
testingtime = 1500
#create simulation environment
testenv = circ_singleav(curstate, auxinfo, roadinfo, avid, drl_reward8,dt = .25)
#%% sanity check
##test baseline with human AV and with control as a simple check for bugs
testenv.simulate_baseline(IDM_b3,p,testingtime) #human model
print('loss for all human scenario is '+str(testenv.totloss)+' starting from initial with '+str(testingtime)+' timesteps')
#    myplot(testenv.sim,auxinfo,roadinfo)
#
testenv.simulate_baseline(FS,[2,.4,.4,3,3,7,15,2], testingtime) #control model
print('loss for one AV with parametrized control is '+str(testenv.totloss)+' starting from initial with '+str(testingtime)+' timesteps')
#    myplot(testenv.sim,auxinfo,roadinfo)
#%% initialize agent
#model = Model(num_actions = 3)
policymodel = PolicyModel(num_actions = 3)
valuemodel = ValueModel()
agent = ACagent(policymodel, valuemodel)
#%%
agent.test(testenv,testingtime, nruns = 1) #200 timesteps
print('before training total reward is '+str(testenv.totloss)+' over '+str(len(testenv.sim[testenv.avid]))+' timesteps')
#%%
#    MWE of training
allrewards = []
for i in range(10):
    rewards = agent.train(testenv, 100)
#    plt.plot(rewards)
#    plt.ylabel('rewards')
#    plt.xlabel('episode')
    allrewards.extend(rewards)
    agent.test(testenv,testingtime,nruns=1)
    print('total reward is '+str(testenv.totloss)+' over '+str(len(testenv.sim[testenv.avid]))+' timesteps')
#%%
avtraj = np.asarray(testenv.sim[testenv.avid])
plt.figure() #plots, in order, position, speed, and headway time series.
plt.subplot(1,3,1)
plt.plot(avtraj[:,0])
plt.subplot(1,3,2)
plt.plot(avtraj[:,1])
plt.subplot(1,3,3)
plt.plot(avtraj[:,2])
#%%
#mcenv = gym.make('MountainCar-v0')
#mcagent = ACagent(PolicyModel(num_actions=mcenv.action_space.n), ValueModel())
#mctestenv = gym_env(mcenv)
#allmcrewards = []
#for i in range(4):
#    rewards = mcagent.train(mctestenv, 100)
##    plt.plot(rewards)
##    plt.ylabel('rewards')
##    plt.xlabel('episode')
#    allmcrewards.extend(rewards)
#    mcagent.test(mctestenv,1000,nruns=1)
#    print('total reward is '+str(mctestenv.totloss)+' over '+str(mcagent.counter)+' timesteps')
#%%
