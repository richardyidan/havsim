
"""
@author: rlk268@cornell.edu
"""
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as kl
import tensorflow.keras.losses as kls
import matplotlib.pyplot as plt

#disable gpu 
physical_devices = tf.config.experimental.list_physical_devices('GPU') 
try: 
  # Disable first GPU 
  tf.config.set_visible_devices(physical_devices[1:], 'GPU') 
  logical_devices = tf.config.list_logical_devices('GPU') 
  # Logical device was not created for first GPU 
  assert len(logical_devices) == len(physical_devices) - 1 
except: 
  # Invalid device or cannot modify virtual devices once initialized. 
  pass 


import havsim
from havsim.simulation.simulation import *
from havsim.simulation.models import *
from havsim.plotting import plotformat, platoonplot

import copy
import math
import gym
import os
from tqdm import tqdm
from scipy.interpolate import interp1d
#to start we will just use a quantized action space since continuous actions is more complicated
#%%
class ProbabilityDistribution(tf.keras.Model):
    def call(self, logits, **kwargs):
        return tf.squeeze(tf.random.categorical(logits, 1), axis=-1)


class Model(tf.keras.Model):
  def __init__(self, num_actions):
    super().__init__('mlp_policy')
    self.hidden1 = kl.Dense(32, activation='tanh') #hidden layer for actions (policy)
    self.hidden11 = kl.Dense(32,activation = 'tanh')
    self.hidden111 = kl.Dense(32, activation = 'tanh')
    self.hidden2 = kl.Dense(32, activation='tanh') #hidden layer for state-value
    self.hidden22 = kl.Dense(32, activation='tanh')
    self.hidden222 = kl.Dense(32, activation='tanh')
    self.value = kl.Dense(1, name = 'value')
    # Logits are unnormalized log probabilities.
    self.logits = kl.Dense(num_actions, name = 'policy_logits')
    self.dist = ProbabilityDistribution()

  def call(self, inputs, **kwargs):
    x = tf.convert_to_tensor(inputs)
    hidden_logs = self.hidden1(x)
    hidden_logs = self.hidden11(hidden_logs)
    hidden_logs = self.hidden111(hidden_logs)
    hidden_vals = self.hidden2(x)
    hidden_vals = self.hidden22(hidden_vals)
    hidden_vals = self.hidden222(hidden_vals)
    return self.logits(hidden_logs), self.value(hidden_vals)

  def action_value(self, obs):
    logits, value = self.predict_on_batch(obs)
    action = self.dist.predict_on_batch(logits)
    return tf.squeeze(action, axis=-1), tf.squeeze(value, axis=-1)

class PolicyModel(tf.keras.Model):
  def __init__(self, num_actions):
    super().__init__('mlp_policy')
    self.hidden1 = kl.Dense(32, activation='tanh') #hidden layer for actions (policy)
    self.hidden11 = kl.Dense(32,activation = 'tanh')
    self.hidden111 = kl.Dense(32, activation = 'tanh')
    # Logits are unnormalized log probabilities.
    self.logits = kl.Dense(num_actions, name = 'policy_logits')
    self.dist = ProbabilityDistribution()

  def call(self, inputs, **kwargs):
    x = tf.convert_to_tensor(inputs)
    hidden_logs = self.hidden1(x)
    hidden_logs = self.hidden11(hidden_logs)
    hidden_logs = self.hidden111(hidden_logs)
    return self.logits(hidden_logs)

  def action(self, obs):
    logits = self.predict_on_batch(obs)
    action = self.dist.predict_on_batch(logits)
    return tf.squeeze(action, axis=-1)

class ValueModel(tf.keras.Model):
  def __init__(self):
    super().__init__('mlp_policy')
    self.hidden2 = kl.Dense(32, activation='tanh') #hidden layer for state-value
    self.hidden22 = kl.Dense(32, activation='tanh')
    self.hidden222 = kl.Dense(32, activation='tanh')
    self.val = kl.Dense(1, name = 'value')

  def call(self, inputs, **kwargs):
    x = tf.convert_to_tensor(inputs)
    hidden_vals = self.hidden2(x)
    hidden_vals = self.hidden22(hidden_vals)
    hidden_vals = self.hidden222(hidden_vals)
    return self.val(hidden_vals)

  def value(self, obs):
    value = self.predict_on_batch(obs)
    return tf.squeeze(value, axis=-1)

optimizer = tf.keras.optimizers.RMSprop(learning_rate = 9e-4)
class ACagent:
    def __init__(self,policymodel, valuemodel, batch_sz=64, eps = 0.05):
        #self.model = model
        self.policymodel = policymodel
        self.valuemodel = valuemodel
        self.gamma = .9995 #discounting learning_rate = 3e-8
        '''
        self.model.compile(
                optimizer = tf.keras.optimizers.RMSprop(learning_rate = 9e-4), #optimizer = tf.keras.optimizers.RMSprop(learning_rate = 3e-7)
                #optimizer = tf.keras.optimizers.SGD(learning_rate=7e-3,),
                loss = [self._logits_loss, self._value_loss])
        '''
        self.policymodel.compile(
                optimizer = tf.keras.optimizers.RMSprop(learning_rate = 9e-4), #optimizer = tf.keras.optimizers.RMSprop(learning_rate = 3e-7)
                #optimizer = tf.keras.optimizers.SGD(learning_rate=7e-3,),
                loss = [self._logits_loss])
        self.valuemodel.compile(
                optimizer = tf.keras.optimizers.RMSprop(learning_rate = 9e-4), #optimizer = tf.keras.optimizers.RMSprop(learning_rate = 3e-7)
                #optimizer = tf.keras.optimizers.SGD(learning_rate=7e-3,),
                loss = [self._value_loss])
        
        #I set learning rate small because rewards are pretty big, can try changing
        self.logitloss = kls.SparseCategoricalCrossentropy(from_logits=True)
        
        self.batch_sz = batch_sz

        #keep track of how many steps in simulation we have taken 
        self.counter = 0
        #keep track of discounting 
        self.I = 1
        #goal for how long we want the simulation to be ideally (with no early termination)
        self.simlen = 1500
        
        #Weight Checkpoints
        self.checkpoint_path = "trainingcp/cp-{version:04d}.ckpt"
        self.checkpoint_dir = os.path.dirname(self.checkpoint_path)
        
        self.eps = eps
    
    def action_value(self, obs):
        if (np.random.uniform(0,1) < self.eps):
            num_actions = self.policymodel.layers[-2].get_config()['units']
            random_action = np.random.randint(num_actions)
            return tf.convert_to_tensor(random_action, dtype=np.int64), self.valuemodel.value(obs)
            
        else:
            return self.policymodel.action(obs), self.valuemodel.value(obs)
        
    def reset(self, env):
        state = env.reset()
        self.counter = 0
        self.I = 1
        return state
           
    def test_orig(self, env, timesteps, nruns = 4): #Note that this is pretty much the same as simulate_baseline in the environment = circ_singleav
        curstate = self.reset(env)

        run = 0
        losses = []
        while (run < nruns):
            for i in range(timesteps):
                action,value = self.action_value(curstate)
                
                curstate, reward, done = env.step(action, i, timesteps)
                #update state, update cumulative reward
                env.totloss += reward
                
#                if (run == 0):
#                    env.savestate()
                
                if done:
#                    losses.append(env.totloss)
#                    if run != nruns -1:
#                        self.reset(env)
                    break
            losses.append(env.totloss)
            if run != nruns -1:
                self.reset(env)
            run += 1

        env.totloss = np.sum(losses) / nruns
        
    def test(self,env,timesteps,nruns = 4):
        curstate = self.reset(env)
        run = 0
        rewards = []
        rewardslist = []
        eplenlist = []
        while (run < nruns):
            for i in tqdm(range(timesteps)):
                action, value = self.action_value(curstate)
                curstate, reward, done = env.step(action, i, timesteps)
                self.counter += 1
                rewards.append(reward)
                env.totloss += reward
                if done:
                    eplenlist.append(self.counter)
                    rewardslist.append(sum(rewards))
                    if (run + 1 < nruns):
                        curstate = self.reset(env)
                    rewards = []
                    break
            run += 1
        num_eps = len(eplenlist) if len(eplenlist) > 0 else 1
        env.totloss = np.sum(rewardslist) / num_eps
        return rewardslist, eplenlist
    
    def train_step(self,statemem, TDacc, TDerrors) :
        with tf.GradientTape(persistent=True) as tape:
            logits, values = self.model.call(statemem)
            valuesloss = self._value_loss(TDerrors, values)
            logitsloss = self._logits_loss(TDacc, logits)
        valuegradient = tape.gradient(valuesloss, self.model.trainable_variables)
        logitsgradient = tape.gradient(logitsloss, self.model.trainable_variables)
        optimizer.apply_gradients(zip(valuegradient, self.model.trainable_variables))
        optimizer.apply_gradients(zip(logitsgradient, self.model.trainable_variables))        
        
    def hist_weights(self,filename):
        fig, axs = plt.subplots(4,1)
        
        for weight in self.model.layers[0].get_weights():
            axs[0].hist(np.ndarray.flatten(weight), bins=500)
            
        for weight in self.model.layers[1].get_weights():
            axs[1].hist(np.ndarray.flatten(weight), bins=500)
            
        for weight in self.model.layers[2].get_weights():
            axs[2].hist(np.ndarray.flatten(weight), bins=500)
            
        for weight in self.model.layers[3].get_weights():
            axs[3].hist(np.ndarray.flatten(weight), bins=500)
        
        plt.savefig(os.path.join(os.path.dirname(self.checkpoint_path),filename))
        plt.close(fig)
        
    def train(self, env, updates=250, by_eps = False, numeps = 1):   
        curstate = self.reset(env)
        
        batch_sz = self.simlen * numeps if by_eps else self.batch_sz
        statemem = np.empty((batch_sz,env.statememdim))
        rewards = np.empty((batch_sz))
        values = np.empty((batch_sz))
        actions = np.empty(batch_sz)
        dones = np.empty((batch_sz))
        
        ep_rewards = []
        
        action,value = self.action_value(curstate)
        for i in tqdm(range(updates)):
            batchlen = 0 #enable flexible batch sizes if training by episode
            epsdone = 0
            
            firstdone = -1
            gammafactor = self.counter
            
            for bstep in range(batch_sz):
                statemem[bstep] = curstate
                
                nextstate, reward, done = env.step(action,self.counter,self.simlen, False)
                nextaction, nextvalue = self.action_value(nextstate)
                env.totloss += reward
                self.counter += 1
                
                rewards[bstep] = reward
                values[bstep] = value
                dones[bstep] = done
                actions[bstep] = action
                batchlen += 1
                
                action, value = nextaction, nextvalue    
                if done or self.counter >= self.simlen: #reset simulation 
                    ep_rewards.append(env.totloss)
                    curstate = self.reset(env)
                    action,value = self.action_value(curstate)
                    
                    epsdone += 1
                    if by_eps and epsdone == numeps:
                        break
                    if firstdone == -1:
                        firstdone = bstep

            gamma_adjust = np.ones(batchlen)
            adj_idx = firstdone + 1 if (firstdone!= -1) else batchlen #update all gammas if no dones in batch
            gamma_adjust[:adj_idx] = self.gamma**gammafactor
            TDerrors = self._TDerrors(rewards[:batchlen], values[:batchlen], dones[:batchlen], nextvalue, gamma_adjust, 3)
            TDacc = tf.stack([TDerrors, tf.cast(actions[:batchlen], tf.float32)], axis = 1)
        
            self.policymodel.train_on_batch(statemem[:batchlen,:], TDacc)
            self.valuemodel.train_on_batch(statemem[:batchlen,:], TDerrors)

        return ep_rewards
            
    def _TDerrors(self, rewards, values, dones, nextvalue, gamma_adjust, nstep):
        returns = np.append(np.zeros_like(rewards), nextvalue)
        stepreturns = np.zeros_like(rewards)
        
        for t in reversed(range(rewards.shape[0])):
            #cumulative rewards
            returns[t] = rewards[t] + self.gamma*returns[t+1]*(1 - dones[t])
            
            #adjustment for nstep
            if ((t + nstep  < len(returns)-1) and (1 not in dones[t:t+nstep])):
                stepreturns[t] = returns[t] \
                            - self.gamma**nstep*returns[t+nstep]  \
                            + self.gamma**nstep*values[t+nstep]
            else:
                stepreturns[t] = returns[t]
                
        returns = stepreturns * gamma_adjust
        return returns - values

    def _value_loss(self, target, value):        
        return -target*value
    
    def _logits_loss(self,target, logits):
        TDerrors, actions = tf.split(target, 2, axis = -1) 
        '''
        p = tf.math.exp(logits)
        p = p /  tf.repeat(tf.math.reduce_sum(p, axis = 1,keepdims = True), 3,1) #probabilities
        a = tf.expand_dims(tf.range(0, len(actions), dtype = tf.int32), 1)
        actions = tf.concat([a,actions], 1)
        out = tf.gather_nd(p, actions)
        
        num_actions = self.policymodel.layers[-2].get_config()['units']
        out = (1-self.eps)*out + self.eps/num_actions
        
        return TDerrors * -tf.math.log(out)
        
        AttributeError: module 'tensorflow' has no attribute 'repeat'
        '''
        logprob = self.logitloss(actions, logits, sample_weight = TDerrors) #really the log probability is negative of this.
        #equivalent to 
        #logprob = mylogitsloss(actions, logits)  <-- from untitled12.py
        #logprob = TDerrors * logprob
        
        probs = tf.nn.softmax(logits)
        entropy_loss = kls.categorical_crossentropy(probs,probs)

        return logprob - 1e-4*entropy_loss   
        
def NNhelper(out, curstate, *args, **kwargs):
    #this is hacky but basically we just want the action from NN to end up
    #in a[i][1]
    return [curstate[1],out]

def myplot(sim, auxinfo, roadinfo, platoon= []):
    #note to self: platoon keyword is messed up becauase plotformat is hacky - when vehicles wrap around they get put in new keys
    meas, platooninfo = plotformat(sim,auxinfo,roadinfo, starttimeind = 0, endtimeind = math.inf, density = 1)
    platoonplot(meas,None,platooninfo,platoon=platoon, lane=1, colorcode= True, speed_limit = [0,25])
    plt.ylim(0,roadinfo[0])
    # plt.show()

class circ_singleav: #example of single AV environment
    #basically we just wrap the function simulate_step
    #avid = id of AV
    #simulates on a circular road
    
    def __init__(self, initstate,auxinfo,roadinfo,avid,rewardfn,updatefun=update_cir,dt=.25):
        self.initstate = initstate
        self.auxinfo = auxinfo
        self.auxinfo[avid][6] = NNhelper
        self.roadinfo = roadinfo
        self.avid = avid
        self.updatefun = updatefun
        self.dt = dt
        self.rewardfn = rewardfn
        self.sim = []
        
        self.mem = 4
        #stuff for building states (len self.mem+1 tuple of past states)
        self.paststates = [] #holds sequence of states
        self.statecnt = 0
        self.statememdim = (self.mem+1)*5
        self.interp1d = interp1d((1.84,43.13), (0,1),fill_value = 'extrapolate')
        
    def reset(self):
        self.curstate = self.initstate
        del self.sim
        self.sim = {i:[self.curstate[i]] for i in self.initstate.keys()}
        self.vavg = {i:self.initstate[i][1]  for i in self.initstate.keys()}
        self.totloss = 0
   
        self.paststates = []
        self.statecnt = 0
        return self.get_state(self.curstate)

    def get_state(self, curstate):
        avlead = self.auxinfo[self.avid][1]
        avfol = [k for k,v in self.auxinfo.items() if v[1] == self.avid][0] 
        
        extend_seq = (np.interp(curstate[self.avid][1], (0,25.32), (0,1)),
                      np.interp(curstate[avlead][1], (0,25.32), (0,1)),
                     self.interp1d(curstate[self.avid][2])*1,
                      np.interp(curstate[avfol][1], (0,25.32), (0,1)),
                      self.interp1d(curstate[avfol][2])*1
                      )
        self.paststates.extend(extend_seq)
        if self.statecnt < self.mem:
            avstate = list(extend_seq) * int(self.mem + 1)
            self.statecnt += 1
        else:
            avstate = self.paststates[-self.statememdim:]
        
        avstate = tf.convert_to_tensor([avstate])
        return avstate
    
    def get_acceleration(self,action,curstate):
        #action from NN gives a scalar, we convert it to the quantized acceleration
#        acc = tf.cast(action,tf.float32)*.1-1.5 #30 integer actions -> between -1.5 and 1.4 in increments of .1
        acc = tf.cast(action, tf.float32) - 1
        
        nextspeed = curstate[self.avid][1] + self.dt*acc
        if nextspeed < 0:
            acc = -curstate[self.avid][1]/self.dt
        
        return acc
    
    def step(self, action, iter, timesteps, save_state = True, baseline = False): #basically just a wrapper for simulate step to get the next timestep
        #simulate_step does all the updating; first line is just a hack which can be cleaned later
        if baseline:
            acc = action
        else:
            acc = self.get_acceleration(action,self.curstate)
        self.auxinfo[self.avid][5] = acc
        nextstate, _ = simulate_step(self.curstate, self.auxinfo,self.roadinfo,self.updatefun,self.dt)
        
        #update environment state 
        self.curstate = nextstate 
        if save_state:
            self.savestate()
        
        #get reward, update average velocity
        reward, vavg = self.rewardfn(nextstate,self.vavg)
        self.vavg = vavg
        
        allheadways = [ nextstate[i][2] for i in nextstate.keys() ]
        shouldterminate = np.any(np.array(allheadways) <= 0)
        
        nextstate = nextstate if baseline else self.get_state(nextstate)
        
        if shouldterminate:
#            return nextstate, -10000+ reward, True
            return nextstate, reward, True

        return nextstate, reward, False

    def simulate_baseline(self, CFmodel, p, timesteps): #can insert a CF model and parameters (e.g. put in human model or parametrized control model)
        #for debugging purposes to verify that timestepping is done correctly
        #if using deep RL the code to simulate/test is the same except action is chosen from NN
        self.reset()
        avlead = self.auxinfo[avid][1]
        for i in range(timesteps):
            action = CFmodel(p, self.curstate[avid],self.curstate[avlead], dt = self.dt)
            nextstate, reward, done = self.step(action[1],i,timesteps, baseline = True)
            #update state, update cumulative reward
            self.curstate = nextstate
            self.totloss += reward
#            #save current state to memory (so we can plot everything)
#            for j in nextstate.keys():
#                self.sim[j].append(nextstate[j])
            if done:
                break
    
    def savestate(self):
        for j in self.curstate.keys():
            self.sim[j].append(self.curstate[j])
        
class gym_env:
    def __init__(self, env):
        self.env = env
        self.initstate = self.env.reset()
        self.statememdim = self.initstate.shape[0]
        
    def reset(self):
        self.curstate = self.env.reset()
        self.totloss = 0
        return self.get_state(self.curstate)
    
    def get_state(self, curstate):
        return curstate[None, :]
    
    def step(self, action, *_, **__):
        nextstate, reward, done, _ = self.env.step(action.numpy())
        self.curstate = nextstate
        return self.get_state(nextstate), reward, done
    
    def savestate(self):
        pass

#if __name__ == "main":
#    pass
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
mcenv = gym.make('MountainCar-v0')
mcagent = ACagent(PolicyModel(num_actions=mcenv.action_space.n), ValueModel())
mctestenv = gym_env(mcenv)
allmcrewards = []
for i in range(10):
    rewards = mcagent.train(mctestenv, 100)
#    plt.plot(rewards)
#    plt.ylabel('rewards')
#    plt.xlabel('episode')
    allmcrewards.extend(rewards)
    mcagent.test(mctestenv,testingtime,nruns=1)
    print('total reward is '+str(mctestenv.totloss)+' over '+str(mcagent.counter)+' timesteps')
