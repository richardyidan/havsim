
"""
@author: rlk268@cornell.edu
"""

import tensorflow as tf
import numpy as np
from havsim.plotting import compute_headway2

def make_dataset(meas, platooninfo, dt = .1):
    # select training/testing vehicles
    nolc_list = []
    for veh in meas.keys():
        temp = nolc_list.append(veh) if len(platooninfo[veh][4]) == 1 else None
    np.random.shuffle(nolc_list)
    train_veh = nolc_list[:-100]
    test_veh = nolc_list[-100:]

    # get normalization for inputs, and get input data
    training = {}
    maxheadway, maxspeed = 0, 0
    minacc, maxacc = 1e4, -1e4
    for veh in train_veh:
        headway = compute_headway2(veh, meas, platooninfo, h = dt)
        t0, t1, t2, t3 = platooninfo[veh][:4]
        lead = platooninfo[veh][4][0]
        lt0, lt1, lt2, lt3 = platooninfo[lead][:4]
        leadspeed = meas[lead][t1-lt0:t2+1-lt0,3]
        vehpos = meas[veh][t1-t0:t2+1-t0,2]
        leadpos = headway + vehpos
        IC = [meas[veh][t1-t0,2], meas[veh][t1-t0,3]]

        vehacc = [(vehpos[i+2] - 2*vehpos[i+1] + vehpos[i])/(h**2) for i in range(len(vehpos)-2)]
        minacc, maxacc = min(minacc, min(vehacc)), max(maxacc, max(vehacc))
        maxheadway = max(max(headway), maxheadway)
        maxspeed = max(max(leadspeed), maxspeed)
        training[veh] = {'IC':IC, 'times':[t1,t2], 'posmem':vehpos,
                         'lead posmem':leadpos, 'lead speedmem':leadspeed}

    testing = {}
    for veh in test_veh:
        headway = compute_headway2(veh, meas, platooninfo, h = dt)
        t0, t1, t2, t3 = platooninfo[veh][:4]
        lead = platooninfo[veh][4][0]
        lt0, lt1, lt2, lt3 = platooninfo[lead][:4]
        leadspeed = meas[lead][t1-lt0:t2+1-lt0,3]
        vehpos = meas[veh][t1-t0:t2+1-t0,2]
        leadpos = headway + vehpos
        IC = [meas[veh][t1-t0,2], meas[veh][t1-t0,3]]

        testing[veh] = {'IC':IC, 'times':[t1,t2], 'posmem':vehpos,
                        'lead posmem':leadpos, 'lead speedmem':leadspeed}

    return training, testing, maxheadway, maxspeed, minacc, maxacc


class RNNCFModel(tf.keras.Model):
    def __init__(self, maxhd, maxv, mina, maxa, learning_rate = .001, dt = .1):
        super().__init__()
        # architecture
        self.lstm_cell = tf.keras.layers.LSTMCell(20)
        self.dense1 = tf.keras.layers.Dense(1)

        # normalization constants
        self.maxhd = maxhd
        self.maxv = maxv
        self.mina = mina
        self.maxa = maxa
        self.dt = dt

        # optimizer and loss function
        self.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate),
                     loss = tf.keras.losses.MeanSquaredError())

    def call(self, inputs):
        # prepare data for call
        lead_inputs, init_state, hidden_states = inputs
        lead_inputs = tf.unstack(lead_inputs, axis=1)  # unpacked over time dimension
        cur_pos, cur_speed = tf.unstack(init_state, axis=1)
        outputs = []
        for cur_lead_input in lead_inputs:
            # normalize data for current timestep
            cur_lead_pos, cur_lead_speed = tf.unstack(cur_lead_input, axis=1)
            curhd = cur_lead_pos-cur_pos
            curhd = curhd/self.maxhd
            cur_lead_speed = cur_lead_speed/self.maxv
            norm_veh_speed = cur_speed/self.maxv
            cur_inputs = tf.stack([curhd, norm_veh_speed, cur_lead_speed],axis=1)

            # call to model
            x, hidden_states = self.lstm_cell(cur_inputs, hidden_states)
            x = self.dense1(x)  # output of the model is current acceleration for the batch

            # update vehicle states
            cur_acc = (self.maxa-self.mina)*x + self.mina
            cur_speed = cur_speed + self.dt*cur_acc
            cur_pos = cur_pos + self.dt*cur_acc
            outputs.append(cur_pos)

        outputs = tf.concat(outputs, -1)
        return outputs

def make_batch(ds, nveh = 20, nt = 5, lstm_units = 20):
    """Create batch of data to send to model.

    Args:
        ds - dataset, from make_dataset
        nveh - number of vehicles in batch
        nt - number of timesteps in batch
        lstm_units - number of LSTM units in model
    Returns:
        lead_inputs: nested python list with shape (nveh, nt, 2), giving the leader position and speed at
            each timestep. Padded with zeros
        true_traj: nested python list with shape (nveh, nt) giving the true vehicle position at each time.
            Padded with zeros
        loss_weights: nested python list with shape (nveh, nt) with either 1 or 0, used to weight each sample
            of the loss function. If 0, it means the input at the corresponding index doesn't contribute
            to the loss.
        init_state: nested python list with shape (nveh, 2) giving the vehicle position and speed at the
            starting timestep.
        hidden_states: list of the two hidden states, each hidden state has shape of (nveh, lstm_units).
            Initialized as all zeros for the first timestep.
    """
    # select vehicles to put in the batch
    vehlist = list(ds.keys())
    np.random.shuffle(vehlist)
    vehs = vehlist[:nveh]
    # stores current time index, maximum time index (length - 1) for each vehicle
    vehs_counter = {veh: (0, ds[veh]['IC'][1]-ds[veh]['IC'][0]) for veh in vehs}

    lead_inputs = []
    true_traj = []
    loss_weights = []
    for veh in vehs:
        t0, tmax = vehs_counter[veh]
        leadpos, leadspeed = ds[veh]['lead posmem'], ds[veh]['lead speedmem']
        posmem = ds[veh]['posmem']
        curlead = []
        curtraj = []
        curweights = []
        for i in range(nt):
            if t0+i < tmax:
                curlead.append([leadpos[t0+i], leadspeed[t0+i]])
                curtraj.append(posmem[t0+i])
                curweights.append(1)
            else:
                curlead.append([0,0])
                curtraj.append(0)
                curweights.append(0)
        lead_inputs.append(curlead)
        true_traj.append(curtraj)

    init_state = [ds[veh]['IC'] for veh in vehs]

    hidden_states = [tf.zeros((nveh, lstm_units)),  tf.zeros((nveh, lstm_units))]

    return lead_inputs, true_traj, loss_weights, init_state, hidden_states


def training_loop(ds, nveh = 20, nt = 5, lstm_units = 20):

    lead_inputs, true_traj, loss_weights, init_state, hidden_states = make_batch(ds, nveh, nt, lstm_units)






