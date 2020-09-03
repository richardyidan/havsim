# imports and load data
from havsim.calibration import deep_learning
import pickle
import numpy as np
import tensorflow as tf

try:
    with open('C:/Users/rlk268/OneDrive - Cornell University/havsim/data/recon-ngsim.pkl', 'rb') as f:
        meas, platooninfo = pickle.load(f) #load data
except:
    with open('/home/rlk268/havsim/data/recon-ngsim.pkl', 'rb') as f:
        meas, platooninfo = pickle.load(f) #load data

#%% generate training data and initialize model/optimizer

nolc_list = []
for veh in meas.keys():
    temp = nolc_list.append(veh) if len(platooninfo[veh][4]) == 1 else None
np.random.shuffle(nolc_list)
train_veh = nolc_list[:-100]
test_veh = nolc_list[-100:]

training, norm = deep_learning.make_dataset(meas, platooninfo, train_veh)
maxhd, maxv, mina, maxa = norm
testing, unused = deep_learning.make_dataset(meas, platooninfo, test_veh)

model = deep_learning.RNNCFModel(maxhd, maxv, mina, maxa)
loss = deep_learning.masked_MSE_loss
opt = tf.keras.optimizers.Adam(learning_rate = .001)

#%% train and save results
deep_learning.training_loop(model, loss, opt, training, nbatches = 2000, nveh = 32, nt = 50)
deep_learning.training_loop(model, loss, opt, training, nbatches = 1000, nveh = 32, nt = 100)
deep_learning.training_loop(model, loss, opt, training, nbatches = 1000, nveh = 32, nt = 200)
deep_learning.training_loop(model, loss, opt, training, nbatches = 1000, nveh = 32, nt = 300)
deep_learning.training_loop(model, loss, opt, training, nbatches = 1000, nveh = 32, nt = 500)

model.save_weights('trained LSTM')
with open('model_aux_info.pkl', 'wb') as f:
     norm = (maxhd, maxv, mina, maxa)
     model_used = 'havsim.calibration.deep_learning.RNNCFModel'
     kwargs = 'lstm_units=20'
     aux_info = 'normalization was '+str(norm)+'\nmodel used was '+str(model_used)+'\nkwargs were '+str(kwargs)
     pickle.dump(aux_info, f)


#%% test by generating entire trajectories
out = deep_learning.generate_trajectories(model, list(testing.keys()), testing, loss=deep_learning.weighted_masked_MSE_loss)
out2 = deep_learning.generate_trajectories(model, list(training.keys()), training, loss=deep_learning.weighted_masked_MSE_loss)

print(' testing loss was '+str(out[-1]))
print(' training loss was '+str(out2[-1]))




