
"""
@author: rlk268@cornell.edu
"""
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as kl
import tensorflow.keras.losses as kls
import matplotlib.pyplot as plt

import havsim
from havsim.simulation.simulation import *
from havsim.simulation.models import *
from havsim.plotting import plotformat, platoonplot

import copy
import math
import gym
import os
from tqdm import tqdm
#to start we will just use a quantized action space since continuous actions is more complicated
#%%
class ProbabilityDistribution(tf.keras.Model):
    def call(self, logits, **kwargs):
        return tf.squeeze(tf.random.categorical(logits, 1), axis=-1)


class Model(tf.keras.Model):
  def __init__(self, num_actions):
    super().__init__('mlp_policy')
    self.hidden1 = kl.Dense(128, activation='relu') #hidden layer for actions (policy)
    self.hidden2 = kl.Dense(128, activation='relu') #hidden layer for state-value
    self.value = kl.Dense(1, name = 'value')
    # Logits are unnormalized log probabilities.
    self.logits = kl.Dense(num_actions, name = 'policy_logits')
    self.dist = ProbabilityDistribution()

  def call(self, inputs, **kwargs):
    x = tf.convert_to_tensor(inputs)
    hidden_logs = self.hidden1(x)
    hidden_vals = self.hidden2(x)
    return self.logits(hidden_logs), self.value(hidden_vals)

  def action_value(self, obs):
    logits, value = self.predict_on_batch(obs)
    action = self.dist.predict_on_batch(logits)
    return tf.squeeze(action, axis=-1), tf.squeeze(value, axis=-1)

optimizer = tf.keras.optimizers.RMSprop(learning_rate = 7e-3)
class ACagent:
    def __init__(self,model, batch_sz=64):
        self.model = model

        self.gamma = .99 #discounting learning_rate = 3e-8
        self.model.compile(
                optimizer = tf.keras.optimizers.RMSprop(learning_rate = 7e-3), #optimizer = tf.keras.optimizers.RMSprop(learning_rate = 3e-7)
                #optimizer = tf.keras.optimizers.SGD(learning_rate=1e-7,)
                loss = [self._logits_loss, self._value_loss])
        #I set learning rate small because rewards are pretty big, can try changing
        self.logitloss = kls.SparseCategoricalCrossentropy(from_logits=True)
        
        self.batch_sz = batch_sz

        #keep track of how many steps in simulation we have taken 
        self.counter = 0
        #keep track of discounting 
        self.I = 1
        #goal for how long we want the simulation to be ideally (with no early termination)
        self.simlen = 1500
        
        # Weight Checkpoints
        self.checkpoint_path = "trainingcp/cp-{version:04d}.ckpt"
        self.checkpoint_dir = os.path.dirname(self.checkpoint_path)

    def reset(self, env):
        env.reset()
        self.counter = 0
        self.I = 1
           
    def test(self, env, timesteps, nruns = 4): #Note that this is pretty much the same as simulate_baseline in the environment = circ_singleav
        self.reset(env)

        run = 0
        losses = []
        while (run < nruns):
            for i in range(timesteps):
                actval_param = env.get_actval_param(env.curstate)
                action,value = self.model.action_value(actval_param)
                acc = env.get_step_param(action)
                
                env.curstate, reward, done = env.step(acc, i, timesteps)
                #update state, update cumulative reward
                env.totloss += reward
                
                if (run == 0):
                    env.savestate()
                
                if done:
                    losses.append(env.totloss)
                    self.reset(env)
                    
                    break
            run += 1

        env.totloss = np.sum(losses) / nruns
     
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
    
    def train(self, env, updates=250):
        self.reset(env)
        
        statemem = np.empty((self.batch_sz,env.statememdim))
        
        rewards = np.empty((self.batch_sz))
        values = np.empty((self.batch_sz))
        actions = np.empty(self.batch_sz)
        dones = np.empty((self.batch_sz))
        
        ep_rewards = []
        
        actval_param = env.get_actval_param(env.curstate)
        action,value = self.model.action_value(actval_param)
        for i in tqdm(range(updates)):
            for bstep in range(self.batch_sz):
                statemem[bstep] = actval_param
                
                acc = env.get_step_param(action)
                nextstate, reward, done = env.step(acc,self.counter,self.simlen)
                actval_param = env.get_actval_param(nextstate)
                nextaction, nextvalue = self.model.action_value(actval_param)
                env.totloss += reward
                self.counter += 1

                rewards[bstep] = reward
                values[bstep] = value
                dones[bstep] = done
                actions[bstep] = action
                    
                if done or self.counter >=self.simlen: #reset simulation 
                    ep_rewards.append(env.totloss)
                    self.reset(env)
                    actval_param = env.get_actval_param(env.curstate)
                    action,value = self.model.action_value(actval_param)
                action, value = nextaction, nextvalue
                
            TDerrors = self._TDerrors(rewards, values, dones, nextvalue)
            TDacc = tf.stack([TDerrors, tf.cast(actions, tf.float32)], axis = 1)
            
            '''
            self.model.save_weights(self.checkpoint_path.format(version=0))
            v0 = tf.train.latest_checkpoint(self.checkpoint_dir)
            
            self.model.load_weights(v0)
            self.model.train_on_batch(statemem, [TDacc,TDerrors])
            self.model.save_weights(self.checkpoint_path.format(version=1))
            
            self.hist_weights("Batch Weights")
            
            self.model.load_weights(v0)
            self.train_step(statemem, TDacc,TDerrors)
            self.model.save_weights(self.checkpoint_path.format(version=2))  
            
            self.hist_weights("Tape Weights")
            
            import pdb; pdb.set_trace()
            
            # Wall time: 1min 10s with tape
            # Wall time: 1min 7s with original
            '''
            self.model.train_on_batch(statemem, [TDacc,TDerrors])
            
        return ep_rewards
            
    def _TDerrors(self, rewards, values, dones, nextvalue):
        returns = np.append(np.zeros_like(rewards), nextvalue, axis = -1)
        
        for t in reversed(range(rewards.shape[0])):
            returns[t] = rewards[t] + self.gamma*returns[t+1]*(1 - dones[t])
        returns = returns[:-1]
        return returns - values

    def _value_loss(self, target, value):
        '''
        equiv1 = np.any(np.isclose(tf.reshape(-target*value, (64,)),.5 * kls.mean_squared_error(target, value)))
        equiv2 = np.any(np.isclose(-target*value,.5*tf.math.square(target - value)))
        # false
        
        equiv3 = np.all(np.isclose(.5 * kls.mean_squared_error(target, value), tf.reshape(.5*tf.math.square(target - value), (64,))))
        # true
        
        import pdb; pdb.set_trace()
        '''
        
        return -target*value
    
    def _logits_loss(self,target, logits):
        TDerrors, actions = tf.split(target, 2, axis = -1) 
        logprob = self.logitloss(actions, logits, sample_weight = TDerrors) #really the log probability is negative of this.
        
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
        
        self.mem = 4
        #stuff for building states (len self.mem+1 tuple of past states)
        self.paststates = [] #holds sequence of states
        self.statecnt = 0
        self.statememdim = (self.mem + 1)*3
        
    def reset(self):
        self.curstate = self.initstate
        self.sim = {i:[self.curstate[i]] for i in self.initstate.keys()}
        self.vavg = {i:self.initstate[i][1]  for i in self.initstate.keys()}
        self.totloss = 0
        
        self.paststates = []
        self.statecnt = 0

    def get_actval_param(self, curstate):
        avlead = self.auxinfo[self.avid][1]
        extend_seq = (np.interp(curstate[self.avid][1], (0,25.32), (0,1)),
                      np.interp(curstate[avlead][1], (0,25.32), (0,1)),
                      np.interp(curstate[self.avid][2], (1.84,43.13), (0,1))                
                      )
        self.paststates.extend(extend_seq)
        if self.statecnt < self.mem:
            avstate = list(extend_seq) * int(self.mem + 1)
            self.statecnt += 1
        else:
            avstate = self.paststates[-(self.mem+1)*3:]
        
        avstate = tf.convert_to_tensor([avstate])
        return avstate
    
    def get_step_param(self,action):
        #action from NN gives a scalar, we convert it to the quantized acceleration
        acc = tf.cast(action,tf.float32)*.1-1.5 #30 integer actions -> between -1.5 and 1.4 in increments of .1
        return acc
    
    def step(self, action, iter, timesteps): #basically just a wrapper for simulate step to get the next timestep
        #simulate_step does all the updating; first line is just a hack which can be cleaned later
        self.auxinfo[self.avid][5] = action
        nextstate, _ = simulate_step(self.curstate, self.auxinfo,self.roadinfo,self.updatefun,self.dt)

        allheadways = [ nextstate[i][2] for i in nextstate.keys() ]
        shouldterminate = np.any(np.array(allheadways) <= 0)
        if shouldterminate:
            return nextstate, -15**2 * len(allheadways) * (timesteps - iter - 1), True

        #get reward, update average velocity
        reward, vavg = self.rewardfn(nextstate,self.vavg)
        self.vavg = vavg
        return nextstate, reward, False

    def simulate_baseline(self, CFmodel, p, timesteps): #can insert a CF model and parameters (e.g. put in human model or parametrized control model)
        #for debugging purposes to verify that timestepping is done correctly
        #if using deep RL the code to simulate/test is the same except action is chosen from NN
        self.reset()
        avlead = self.auxinfo[avid][1]
        for i in range(timesteps):
            action = CFmodel(p, self.curstate[avid],self.curstate[avlead], dt = self.dt)
            nextstate, reward, done = self.step(action[1],i,timesteps)
            #update state, update cumulative reward
            self.curstate = nextstate
            self.totloss += reward
            #save current state to memory (so we can plot everything)
            for j in nextstate.keys():
                self.sim[j].append(nextstate[j])
            if done:
                break
    
    def savestate(self):
        for j in self.curstate.keys():
            self.sim[j].append(self.curstate[j])
        
class cartpole_env:
    def __init__(self, env):
        self.env = env
        self.initstate = self.env.reset()
        self.statememdim = self.initstate.shape[0]
        
    def reset(self):
        self.curstate = self.env.reset()
        self.totloss = 0
    
    def get_actval_param(self, curstate):
        return curstate[None, :]
    
    def get_step_param(self,action):
        return np.array(action)
    
    def step(self, action, *_, **__):
        nextstate, reward, done, _ = self.env.step(action)
        self.curstate = nextstate
        return nextstate, reward, done
    
    def savestate(self):
        pass

if __name__ == "main":
    pass
    #%% initialize agent (we expect the agent to be awful before training)
    env = gym.make('CartPole-v0')
    model = Model(num_actions=env.action_space.n)
    agent = ACagent(model)
    testenv = cartpole_env(env)
    #%%
    agent.test(testenv,200) #200 timesteps
    print('total reward before training is '+str(testenv.totloss)+' starting from initial with 200 timesteps')
    #%%
        #MWE of training
    rewards = agent.train(testenv)
    plt.plot(rewards)
    plt.ylabel('rewards')
    plt.xlabel('episode')
    agent.test(testenv,200)
    print('total reward is '+str(testenv.totloss))


    #%%
                    #specify simulation
    p = [33.33, 1.2, 2, 1.1, 1.5] #parameters for human drivers
    initstate, auxinfo, roadinfo = eq_circular(p, IDM_b3, update2nd_cir, IDM_b3_eql, 41, length = 2, L = None, v = 15, perturb = 2) #create initial state on road
    sim, curstate, auxinfo = simulate_cir(initstate, auxinfo,roadinfo, update_cir, timesteps = 25000, dt = .25)
    vlist = {i: curstate[i][1] for i in curstate.keys()}
    avid = min(vlist, key=vlist.get)
    
    #create simulation environment
    testenv = circ_singleav(curstate, auxinfo, roadinfo, avid, drl_reward,dt = .25)
    #%% sanity check
    ##test baseline with human AV and with control as a simple check for bugs
    testenv.simulate_baseline(IDM_b3,p,200) #human model
    print('loss for all human scenario is '+str(testenv.totloss)+' starting from initial with 1500 timesteps')
#    myplot(testenv.sim,auxinfo,roadinfo)
    #
    testenv.simulate_baseline(FS,[2,.4,.4,3,3,7,15,2], 200) #control model
    print('loss for one AV with parametrized control is '+str(testenv.totloss)+' starting from initial with 1500 timesteps')
#    myplot(testenv.sim,auxinfo,roadinfo)
    #%% initialize agent (we expect the agent to be awful before training)
    model = Model(num_actions = 30)
    agent = ACagent(model)
    #%%
    agent.test(testenv,800) #200 timesteps
    print('before training total reward is '+str(testenv.totloss)+' over '+str(len(testenv.sim[testenv.avid]))+' timesteps')
    #%%
        #MWE of training
    rewards = agent.train(testenv, 50)
    plt.plot(rewards)
    plt.ylabel('rewards')
    plt.xlabel('episode')
    agent.test(testenv,800)
    print('total reward is '+str(testenv.totloss))
