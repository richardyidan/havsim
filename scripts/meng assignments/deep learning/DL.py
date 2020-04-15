
"""
@author: rlk268@cornell.edu
"""
#imports, load data
import tensorflow as tf
import tensorflow.keras.layers as kls
from tensorflow.keras.regularizers import l1, l2
import pickle
import numpy as np
import matplotlib.pyplot as plt
import math
from havsim.calibration.algs import makeplatoonlist
import havsim



def create_input(statemem, j, lead, headway, curmeas):
    if j+1 < statemem:
        leadv = np.append(np.tile(lead[0,1],statemem-j-1),lead[:j+1,1])
        hd = np.append(np.tile(headway[0],statemem-j-1),headway[:j+1])
    else:
        leadv = lead[j+1-statemem:j+1,1]
        hd = headway[j+1-statemem:j+1]

    curx = list(leadv)
    curx.extend(list(hd))
    if j + 1 < len(headway):
        cury = [headway[j+1]]
    else:
        cury = None
    return curx, cury

def normalization_input(xinput, maxheadway, maxvelocity, statemem):
    xinput[:,:statemem] = xinput[:,:statemem]/maxvelocity
    xinput[:,statemem:statemem*2] = xinput[:,statemem:statemem*2]/maxheadway
    return xinput


#comment out and replace with path to pickle files on your computer
path_reconngsim = '/Users/nathanbala/Desktop/meng_project/data/reconngsim.pkl'
path_highd26 = '/Users/nathanbala/Desktop/meng_project/data/highd26.pkl'

# reconstructed ngsim data
with open(path_reconngsim, 'rb') as f:
    data = pickle.load(f)[0]
# highd data
#with open(path_highd26, 'rb') as f:
#    data = pickle.load(f)[0]

meas, platooninfo = makeplatoonlist(data,1, False)




#%% first step is to prepare training/test data
#

#out of all vehicles, we assign them randomly to either train or test
train_or_test = np.random.rand(len(meas.keys()))
train_or_test = train_or_test<.85 #15% of vehicles goes to test rest to train

#need to normalize the input data - speeds and headways
#also need to get the headway from the data
maxvelocity = 0
#get headways for all vehicles
maxheadway = 0
#define how the state should look
statemem = 5

xtrain, ytrain, xtest, ytest = [], [], [], []
for count, i in enumerate(meas.keys()):
    if len(platooninfo[i][4]) != 1:
        continue

    t_nstar, t_n, T_nm1, T_n = platooninfo[i][:4]
    leadinfo, folinfo, rinfo = havsim.calibration.helper.makeleadfolinfo([i], platooninfo, meas)
    relax = havsim.calibration.opt.r_constant(rinfo[0], [t_n, T_nm1], T_n, 5, False)

    if T_nm1 - t_n ==0:
        continue
    lead = np.zeros((T_nm1 - t_n+1,3)) #columns are position, speed, length
    for j in leadinfo[0]:
        curleadid = j[0]
        leadt_nstar = platooninfo[curleadid][0]
        lead[j[1]-t_n:j[2]+1-t_n,:] = meas[curleadid][j[1]-leadt_nstar:j[2]+1-leadt_nstar,[2,3,6]]

    curmeas = meas[i][t_n-t_nstar:T_nm1+1-t_nstar,[2,3,8]] #columns are position, speed, acceleration
    headway = lead[:,0] - curmeas[:,0] - lead[:,2] #headway is distance between front bumper to rear bumper of leader
    headway = np.array(headway) + np.array(relax[0][:T_nm1-t_n+1])





    if train_or_test[count]: #supposed to normalize assuming you have only train data
        temp = max(headway)
        if temp > maxheadway:
            maxheadway = temp
        temp = max(lead[:,1])
        if temp > maxvelocity:
            maxvelocity = temp

    #form samples for the current vehicle
    for j in range(T_nm1+1-t_n):
        if j == 0:
            continue

        curx, cury = create_input(statemem, j, lead, headway, curmeas)
        #new output for model
        if cury == None:
            continue
        if train_or_test[count]:
            xtrain.append(curx)
            ytrain.append(cury)
        else:
            xtest.append(curx)
            ytest.append(cury)





#reshape data into correct dimensions, normalize
xtrain, ytrain, xtest, ytest = np.asarray(xtrain,np.float32), np.asarray(ytrain, np.float32), np.asarray(xtest,np.float32), np.asarray(ytest,np.float32)
maxoutput = max(ytrain[:,0])
minoutput = min(ytrain[:,0])
ytrain = (ytrain + minoutput)/(maxoutput-minoutput)
ytest = (ytest + minoutput)/(maxoutput-minoutput)
xtrain = normalization_input(xtrain, maxheadway, maxvelocity, statemem)
xtest = normalization_input(xtest, maxheadway, maxvelocity, statemem)


 #you'll probably want to save the train_or_test, xtrain, ... ytest in a pickle

xtrain, ytrain, xtest, ytest = tf.convert_to_tensor(xtrain,tf.float32), tf.convert_to_tensor(ytrain,tf.float32), tf.convert_to_tensor(xtest,tf.float32), tf.convert_to_tensor(ytest,tf.float32)

train_ds = tf.data.Dataset.from_tensor_slices(
        (xtrain,ytrain)).shuffle(100000).batch(32)

test_ds = tf.data.Dataset.from_tensor_slices(
        (xtest,ytest)).shuffle(100000).batch(32)
#%%
class Model(tf.keras.Model):
    def __init__(self):
        super().__init__('simple_mlp')
        self.conv1 = kls.Conv1D(32, 3, activation='relu', input_shape=(5, 1))
        self.conv2 = kls.Conv1D(32, 3, activation='relu', input_shape=(5, 1))
        self.hidden1 = kls.Dense(32, activation = 'relu')
        self.hidden2 = kls.Dense(32,activation = 'relu')
        self.hidden3 = kls.Dense(32)
        self.flatten = kls.Flatten()
        self.out = kls.Dense(1)

    def call(self,x):
        # x = self.hidden1(x)
        # x = self.hidden2(x)
        # x = self.hidden3(x)
        # fin = self.out(x)
        # print(fin)
        y1 = self.conv1(tf.expand_dims(x[:, :5], -1))
        y2 = self.conv2(tf.expand_dims(x[:, 5:], -1))
        x1 = self.hidden1(y1)
        x2 = self.hidden2(y2)
        con = tf.concat([x1, x2], 1)
        x = self.hidden3(con)
        flat = self.flatten(x)
        return self.out(flat)

model = Model()

#%% Set up training
#x = input, y = output, yhat = labels (true values)

optimizer = tf.keras.optimizers.RMSprop(learning_rate=2e-5)

loss_fn = tf.keras.losses.MeanSquaredError(name='train_test_loss')
def mytestmetric(y,yhat):
    return tf.math.reduce_mean((y - yhat)**2)

#note: can use model.fit and model.evaluate instead for this simple case

@tf.function
def train_step(x,yhat, loss_fn, optimizer):
    with tf.GradientTape() as tape:
        y = model(x)
        loss = loss_fn(y,yhat)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients,model.trainable_variables))

#@tf.function
def test(dataset, minacc, maxacc):
    mse = []
    for x, yhat in dataset:
        y = model(x)
        m = mytestmetric(y,yhat)
        mse.append(m)
    return (tf.math.reduce_mean(mse)**.5)*(maxacc-minacc)-minacc


def test1(dataset, minacc, maxacc):
    mse = []
    predicted_output = []
    for x, yhat in dataset:
        y = model(x)
        predicted_output.append(y)
        m = mytestmetric(y,yhat)
        mse.append(m)
    return (tf.math.reduce_mean(mse)**.5)*(maxacc-minacc)-minacc, predicted_output

def create_output(xtest, ytest, minoutput, maxoutput, maxvelocity, maxheadway, simulated_trajectory_lst, headway, lead, j):
    xtest[-1][:statemem] = xtest[-1][:statemem]/maxvelocity
    xtest[-1][statemem:statemem*2] = xtest[-1][statemem:statemem*2]/maxheadway
    #vector to put into NN
    xtest = np.asarray(xtest, np.float32)
    cury = [0]
    ytest.append(cury)
    ytest = (ytest + minoutput)/(maxoutput-minoutput)
    ytest = tf.convert_to_tensor(ytest,tf.float32)
    test_ds = tf.data.Dataset.from_tensor_slices(
            (xtest,ytest)).shuffle(100000).batch(32)


    output, predicted_acc = test1(test_ds,minoutput,maxoutput)
    simulated_headway = (predicted_acc[0].numpy()[0][0]) * (maxoutput - minoutput) - minoutput
    if j + 1 < len(lead):
        simulated_trajectory = lead[j+1,0] - lead[j+1,2] - simulated_headway
        headway[j+1] = simulated_headway
    else:
        simulated_trajectory = lead[j,0] - lead[j,2] - simulated_headway
    curr_trajectory = simulated_trajectory

    return curr_trajectory


def predict_trajectory(model, vehicle_id, input_meas, input_platooninfo, maxoutput, minaoutput, maxvelocity, maxheadway, previous_sim_traj):
    #how many samples we should look back
    statemem = 5



    #obtain t_n and T-nm1 for vehicle_id
    t_nstar, t_n, T_nm1, T_n = input_platooninfo[vehicle_id][:4]
    if T_nm1 - t_n ==0:
        return None, None, None
    #obtain leadinfo for vehicle_id
    leadinfo, folinfo, rinfo = havsim.calibration.helper.makeleadfolinfo([vehicle_id], input_platooninfo, input_meas)
    relax = havsim.calibration.opt.r_constant(rinfo[0], [t_n, T_nm1], T_n, 5, False)#change this

    #form the lead trajectory for vehicle_id
    lead = np.zeros((T_nm1 - t_n+1,3)) #columns are position, speed, length



    for j in leadinfo[0]:
        curleadid = j[0]
        leadt_nstar = input_platooninfo[curleadid][0]
        lead[j[1]-t_n:j[2]+1-t_n,:] = meas[curleadid][j[1]-leadt_nstar:j[2]+1-leadt_nstar,[2,3,6]]

    curmeas = meas[vehicle_id][t_n-t_nstar:T_nm1+1-t_nstar,[2,3,8]] #columns are position, speed, acceleration
    headway = lead[:,0] - curmeas[:,0] - lead[:,2] #headway is distance between front bumper to rear bumper of leader
    headway = np.array(headway) + np.array(relax[0][:T_nm1-t_n+1])

    curr_trajectory = curmeas[0,0]
    simulated_trajectory_lst = [curr_trajectory]
    #iterating through simulated times
    for j in range(T_nm1 - t_n+1):
        if j == 0:
            continue
        xtest = []
        ytest = []



        curx, cury = create_input(statemem, j, lead, headway, curmeas)
        xtest.append(curx)
        new_traj = create_output(xtest, ytest, minoutput, maxoutput, maxvelocity, maxheadway, simulated_trajectory_lst, headway, lead, j)
        simulated_trajectory_lst.append(new_traj)
    x = (simulated_trajectory_lst)
    x_hat = (meas[vehicle_id][t_n-t_nstar:T_nm1+1-t_nstar,2])
    error = (tf.sqrt(tf.losses.mean_squared_error(x, x_hat)))

    return x, x_hat, error





def predict_platoon_trajectory(model, vehicle_id, input_meas, input_platooninfo, maxoutput, minaoutput, maxvelocity, maxheadway, previous_sim_traj, start_time, end_time):
    #how many samples we should look back
    statemem = 5

    #obtain leadinfo for vehicle_id
    leadinfo, unused, unused = havsim.calibration.helper.makeleadfolinfo([vehicle_id], input_platooninfo, input_meas)

    #obtain t_n and T-nm1 for vehicle_id
    t_nstar, t_n, T_nm1, T_n = input_platooninfo[vehicle_id][:4]
    if T_nm1 - t_n ==0:
        return None, None, None


    if start_time != None and (start_time > T_nm1 or t_n < start_time):
        return (meas[vehicle_id][t_n-t_nstar:T_nm1+1-t_nstar,2]), (meas[vehicle_id][t_n-t_nstar:T_nm1+1-t_nstar,2]), None, T_nm1, t_n

    #form the lead trajectory for vehicle_id
    lead = np.zeros((T_nm1 - t_n+1,3)) #columns are position, speed, length



    for j in leadinfo[0]:
        curleadid = j[0]
        leadt_nstar = input_platooninfo[curleadid][0]
        lead[j[1]-t_n:j[2]+1-t_n,:] = meas[curleadid][j[1]-leadt_nstar:j[2]+1-leadt_nstar,[2,3,6]]

    curmeas = meas[vehicle_id][t_n-t_nstar:T_nm1+1-t_nstar,[2,3,8]] #columns are position, speed, acceleration
    headway = lead[:,0] - curmeas[:,0] - lead[:,2] #headway is distance between front bumper to rear bumper of leader


    #lead_car simulated trajectory that overlaps timewise
    if start_time != None:
        previous_sim_traj = previous_sim_traj[t_n-start_time:]



    curr_trajectory = curmeas[0,0]
    simulated_trajectory_lst = [curr_trajectory]
    # simulated_headway = [headway[0]]
    change_time = 0
    if end_time != None:
        change_time = T_nm1 - end_time
    #iterating through simulated times
    for j in range(T_nm1 - t_n+1 - (change_time)):
        xtest = []
        ytest = []
        if j+1 < statemem:
            leadv = np.append(np.tile(lead[0,1],statemem-j-1),lead[:j+1,1])
            vehv = np.append(np.tile(curmeas[0,1],statemem-j-1),curmeas[:j+1,1])
            hd = np.append(np.tile(headway[0],statemem-j-1),headway[:j+1])
        else:
            leadv = lead[j+1-statemem:j+1,1]
            vehv = curmeas[j+1-statemem:j+1,1]
            hd = headway[j+1-statemem:j+1]
        curx = list(leadv)
        curx.extend(list(vehv))
        curx.extend(list(hd))
        #current sample is curx to be fed into NN
        xtest.append(curx)
        #normalizing
        xtest[0][:statemem*2] = xtest[0][:statemem*2]/maxvelocity
        xtest[0][statemem*2:statemem*3] = xtest[0][statemem*2:statemem*3]/maxheadway
        #vector to put into NN
        xtest = np.asarray(xtest, np.float32)
        cury = cury = [((-1/2) * [curmeas[j,2]][0] * (0.1 ** 2)) - (vehv[-1] * 0.1)]
        ytest.append(cury)
        ytest = (ytest + minoutput)/(maxoutput-minoutput)
        ytest = tf.convert_to_tensor(ytest,tf.float32)
        test_ds = tf.data.Dataset.from_tensor_slices(
                (xtest,ytest)).shuffle(100000).batch(32)


        output, predicted_acc = test1(test_ds,minoutput,maxoutput)
        unormalized_val = (predicted_acc[0].numpy()[0][0]) * (maxoutput - minoutput) - minoutput
        simulated_headway = hd[-1] + (leadv[-1] * .1) + (unormalized_val)
        simulated_trajectory = 0
        if j + 1 < len(curmeas) - change_time:
            if start_time == None:
                simulated_trajectory = lead[j+1,0] - lead[j+1,2] - simulated_headway
            else:
                simulated_trajectory = previous_sim_traj[j+1] - lead[j+1,2] - simulated_headway
            headway[j+1] = simulated_headway
        else:
            if start_time == None:
                simulated_trajectory = lead[j,0] - lead[j,2] - simulated_headway
            else:
                simulated_trajectory = previous_sim_traj[j] - lead[j,2] - simulated_headway
        simulated_trajectory_lst.append(simulated_trajectory)
        curr_trajectory = simulated_trajectory


    if change_time != 0:
        for i in range(T_nm1 - end_time):
            next_traj = simulated_trajectory_lst[-1] + np.mean(curmeas[:,1]) * 0.1
            simulated_trajectory_lst.append(next_traj)

    x = (simulated_trajectory_lst)[1:]
    x_hat = (meas[vehicle_id][t_n-t_nstar:T_nm1+1-t_nstar,2])
    error = (tf.sqrt(tf.losses.mean_squared_error(x, x_hat)))

    return x, x_hat, error, T_nm1, t_n




#%% training and testing
# m = test(test_ds,minoutput,maxoutput)
# m2 = test(train_ds, minoutput,maxoutput)
# print('before training rmse on test dataset is '+str(tf.cast(m,tf.float32))+' rmse on train dataset is '+str(m2))

for epoch in range(6):
    for x, yhat in train_ds:
        train_step(x,yhat,loss_fn, optimizer)
    m = test(test_ds,minoutput,maxoutput)
    m2 = test(train_ds, minoutput,maxoutput)
    print('epoch '+str(epoch)+' rmse on test dataset is '+str(m)+' rmse on train dataset is '+str(m2))




# RMSE calculationg
total_error = 0
total_count = 0

lane_error = 0
lane_count =0
for count, i in enumerate(meas.keys()):
    if i == 0:
        continue
    if len(platooninfo[i][4]) != 1:
        pred_traj, acc_traj, rmse = predict_trajectory(model,i ,meas, platooninfo, maxoutput, minoutput, maxvelocity, maxheadway, None)
        print(pred_traj)
        print(acc_traj)
        print(rmse)
        if tf.is_tensor(rmse):
            total_error += rmse
            total_count += 1
    else:
        pred_traj, acc_traj, rmse = predict_trajectory(model,i ,meas, platooninfo, maxoutput, minoutput, maxvelocity, maxheadway, None)
        print(pred_traj)
        print(acc_traj)
        print(rmse)
        if tf.is_tensor(rmse):
            lane_error += rmse
            lane_count += 1

print("------------THIS IS THE FINAL RMSE FOR NONE LANE CHANGE CARS-------------")
print(total_error/total_count)

print("------------THIS IS THE FINAL RMSE FOR LANE CHANGE CARS-------------")
print(lane_error/lane_count)
