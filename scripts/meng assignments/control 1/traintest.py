
"""
@author: rlk268@cornell.edu
"""

from havsim.simulation.models import drl_reward88, drl_reward8, drl_reward9, IDM_b3, IDM_b3_eql, FS
from havsim.simulation.simulationold2 import update2nd_cir, eq_circular, simulate_cir, update_cir

from drl import PolicyModel, ValueModel, ACagent, circ_singleav, gym_env, PolicyModel2, ValueModel2
#specify simulation
p = [33.33, 1.2, 2, 1.1, 1.5] #parameters for human drivers
initstate, auxinfo, roadinfo = eq_circular(p, IDM_b3, update2nd_cir, IDM_b3_eql, 41, length = 2, L = None, v = 15, perturb = 2) #create initial state on road
sim, curstate, auxinfo = simulate_cir(initstate, auxinfo,roadinfo, update_cir, timesteps = 25000, dt = .25)
vlist = {i: curstate[i][1] for i in curstate.keys()}
avid = min(vlist, key=vlist.get)
testingtime = 1500
#create simulation environment
testenv = circ_singleav(curstate, auxinfo, roadinfo, avid, drl_reward88,dt = .25)

#%% test baseline with human AV and with control as a simple check for bugs and to compare with agent performance 

testenv.simulate_baseline(IDM_b3,p,testingtime) #human model
print('loss for all human scenario is '+str(testenv.totloss)+' starting from initial with '+str(testingtime)+' timesteps')
#    myplot(testenv.sim,auxinfo,roadinfo)
#
testenv.simulate_baseline(FS,[2,.4,.4,3,3,7,15,2], testingtime) #control model
print('loss for one AV with parametrized control is '+str(testenv.totloss)+' starting from initial with '+str(testingtime)+' timesteps')
#    myplot(testenv.sim,auxinfo,roadinfo)
#%% initialize agent
#model = Model(num_actions = 3)
#policymodel = PolicyModel(num_actions = 3, num_hiddenlayers = 2)
#valuemodel = ValueModel(num_hiddenlayers = 2)
policymodel = PolicyModel2(3)
valuemodel = ValueModel2()
agent = ACagent(policymodel, valuemodel)

#%% training

agent.test(testenv,testingtime, nruns = 1)
print('before training total reward is '+str(testenv.totloss)+' over '+str(len(testenv.sim[testenv.avid]))+' timesteps')

allrewards = []
for i in range(20):
    rewards = agent.train(testenv, 100)
#    plt.plot(rewards)
#    plt.ylabel('rewards')
#    plt.xlabel('episode')
    allrewards.extend(rewards)
    agent.test(testenv,testingtime,nruns=1)
    print('epoch '+str(i+1)+' total reward is '+str(testenv.totloss)+' over '+str(len(testenv.sim[testenv.avid]))+' timesteps')
    
    testenv.plot()
    testenv.trajplot()
    

