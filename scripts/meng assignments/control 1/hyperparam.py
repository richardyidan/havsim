#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from drl import *
#%%
p = [33.33, 1.2, 2, 1.1, 1.5] #parameters for human drivers
initstate, auxinfo, roadinfo = eq_circular(p, IDM_b3, update2nd_cir, IDM_b3_eql, 41, length = 2, L = None, v = 15, perturb = 2) #create initial state on road
sim, curstate, auxinfo = simulate_cir(initstate, auxinfo,roadinfo, update_cir, timesteps = 25000, dt = .25)
vlist = {i: curstate[i][1] for i in curstate.keys()}
avid = min(vlist, key=vlist.get)

# store the possible hyperparam values
lr_vals = np.around(np.arange(2e-4,1.1e-3,1e-4),4) # lr arg in ACagent
entropy_vals = [1e-5, 5e-5, 1e-4, 5e-4]  # entropy_const arg in ACagent
nstep_vals = [5,10,20,math.inf] # nTDsteps arg in train()

netdepths = [2,3,4] # num_hiddenlayers
numneuron_vals = [32,64,128] # num_neurons 
activations = [kl.Activation('relu'),kl.LeakyReLU(alpha = 0.3),kl.Activation('tanh')] # activationlayer

statemem_vals = [1,5,10] # statemem in circ_singleav()

def traintest(agent_tt, testenv_tt, nTDsteps_tt):
    # agent_tt.train(testenv_tt, updates=1000, nTDsteps=nTDsteps_tt)
    # rewardslist, eplenlist = agent_tt.test(testenv_tt,1500)
    agent_tt.train(testenv_tt, updates=1, nTDsteps=nTDsteps_tt)
    rewardslist, eplenlist = agent_tt.test(testenv_tt,1)
    return np.mean(rewardslist), np.mean(eplenlist)

def resetPolVal(exp_params):
    policymodel = PolicyModel(num_actions = 3, num_hiddenlayers=exp_params["netdP"],num_neurons=exp_params["nneurP"],activationlayer=exp_params["actP"])
    valuemodel = ValueModel(num_hiddenlayers=exp_params["netdV"],num_neurons=exp_params["nneurV"],activationlayer=exp_params["actV"])
    return policymodel,valuemodel

def exp_setup():
    exp_params = {}
    exp_params["lr"] = lr_vals[0]
    exp_params["ev"] = entropy_vals[0]
    exp_params["nstep"] = nstep_vals[0]
    exp_params["netdP"] = netdepths[0]
    exp_params["nneurP"] = numneuron_vals[0]
    exp_params["actP"] = activations[0]
    exp_params["netdV"] = netdepths[0]
    exp_params["nneurV"] = numneuron_vals[0]
    exp_params["actV"] = activations[0]
    exp_params["sm"] = statemem_vals[0]
    return exp_params

def exp_procedure(exp_params, hyperparam):
    agent = ACagent(exp_params["polmodel"], exp_params["valmodel"], lr=exp_params["lr"], entropy_const = exp_params["ev"])
    totreward, testlen = traintest(agent, exp_params["testenv"], exp_params["nstep"])
    exp_params["polmodel"], exp_params["valmodel"] = resetPolVal(exp_params)
    res_fmt = "{}:\t{} reward over {} timesteps\n".format(hyperparam, totreward, testlen)
    return totreward, res_fmt

div_str = '+'+'-'*42+'+\n'
def boxed_section(title):
    text = div_str
    text += str.ljust("| {}".format(title),43) + "|\n"
    text += div_str
    return text

def hyperparam_selection(exp_params, hyperparam_choices, exp_params_key, title, activation = False, resetP = False, resetV = False, resetEnv = False):
    rewards = [] 
    intermediate_res = boxed_section(title)
    for idx, hyperparam in enumerate(hyperparam_choices):
        exp_params[exp_params_key] = hyperparam
        
        if resetP:
            exp_params["polmodel"] = PolicyModel(num_actions = 3, \
                                                num_hiddenlayers=exp_params["netdP"],\
                                                num_neurons=exp_params["nneurP"],\
                                                activationlayer=exp_params["actP"])
        if resetV:
            exp_params["valmodel"] = ValueModel(num_hiddenlayers=exp_params["netdV"],\
                                                num_neurons=exp_params["nneurV"],\
                                                activationlayer=exp_params["actV"])
        
        if resetEnv:
            testenv = circ_singleav(curstate, auxinfo, roadinfo, avid, drl_reward8,dt = .25,statemem=exp_params["sm"])
        
        
        totreward, res_fmt = exp_procedure(exp_params, ['relu','leaky relu','tanh'][idx] if activation else hyperparam) #"{:.4f}".format(hyperparam))
        
        rewards.append(totreward)
        intermediate_res += res_fmt
    
    exp_params[exp_params_key] = hyperparam_choices[np.argmax(rewards)]
    sel = ['relu','leaky relu','tanh'][np.argmax(rewards)] if activation else exp_params[exp_params_key] #"{:.4f}".format(exp_params[exp_params_key])
    intermediate_res += "\nSelected: {}\n".format(sel)
    
    print(intermediate_res)
    return intermediate_res

# Experiment loop starts here    
exp_params = exp_setup()
res = ''''''
for i in range(1):
    # Record which iteration
    res+= div_str + str.ljust("| Iter: {}".format(i),43) + "|\n"
    print("\n".join(res.splitlines()[-2:]))
    
    # Reset policy model, value model, and test environment
    exp_params["polmodel"], exp_params["valmodel"] = resetPolVal(exp_params)
    exp_params["testenv"] = circ_singleav(curstate, auxinfo, roadinfo, avid, drl_reward8,dt = .25,statemem=exp_params["sm"])
    
    # Learning Rate
    res += hyperparam_selection(exp_params, lr_vals, "lr", "Learning rate")
       
    # Entropy
    res += hyperparam_selection(exp_params, entropy_vals, "ev", "Entropy")
    
    # Nstep for TD errors
    res += hyperparam_selection(exp_params, nstep_vals, "nstep", "Nstep for TD errors")
    
    # Policy Depth
    res += hyperparam_selection(exp_params, netdepths, "netdP", "Depth (Policy)", resetP=True)
    
    # Policy Num Neurons
    res += hyperparam_selection(exp_params, numneuron_vals, "nneurP", "Number of neurons in each layer (Policy)", resetP=True)
    
    # Policy Activation
    res += hyperparam_selection(exp_params, activations, "actP", "Activation (Policy)", activation=True, resetP=True)
    
    # Value Depth
    res += hyperparam_selection(exp_params, netdepths, "netdV", "Depth (Value)", resetV=True)
    
    # Value Num Neurons
    res += hyperparam_selection(exp_params, numneuron_vals, "nneurV", "Number of neurons in each layer (Value)", resetV=True)
    
    # Value Activation
    res += hyperparam_selection(exp_params, activations, "actV", "Activation (Value)", activation=True, resetV=True)
    
    # State Memory
    res += hyperparam_selection(exp_params, statemem_vals, "sm", "State memory", resetEnv = True)