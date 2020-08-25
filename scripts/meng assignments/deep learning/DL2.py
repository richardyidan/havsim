
"""
@author: rlk268@cornell.edu
"""

import tensorflow as tf
import numpy as np
from havsim.plotting import compute_headway2

def make_dataset(meas, platooninfo, dt = .1):
    """Makes dataset from meas and platooninfo

    Args:
        meas
        platooninfo
        dt: timestep
    Returns:
        training: dictionary of vehicles, values are a dictionary with keys
            'IC' - (initial conditions) list of starting position/speed for vehicle
            'times' - list of starting time, last time with observed leader
            'posmem' - (1d,) numpy array of observed positions for vehicle, 0 index corresponds to times[0]
            'lead posmem' - (1d,) numpy array of positions for leaders, 0 index corresponds to times[0].
                length is subtracted from the lead position.
            'lead speedmem' - (1d,) numpy array of speeds for leaders.
        testing: dictionary of vehicles in a format like training
    """
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
    def __init__(self, maxhd, maxv, mina, maxa, dt = .1):
        """Inits RNN based CF model.

        Args:
            maxhd: max headway (for nomalization of inputs)
            maxv: max velocity (for nomalization of inputs)
            mina: minimum acceleration (for nomalization of outputs)
            maxa: maximum acceleration (for nomalization of outputs)
            dt: timestep
        """
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

    def call(self, inputs):
        """Updates states for a batch of vehicles.

        Args:
            inputs: list of lead_inputs, init_state, hidden_states - see make_batch
        Returns:
            outputs: tensor of vehicle trajectories, shape of (number of vehicles, number of timesteps)
            hidden_states: last hidden states for LSTM. Tuple of tensors, where each tensor has shape of
                (number of vehicles, number of LSTM units)
        """
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
        return outputs, hidden_states

def make_batch(vehs, vehs_counter, ds, nt = 5, lstm_units = 20):
    """Create batch of data to send to model.

    Args:
        vehs: list of vehicles in current batch
        vehs_counter: dictionary where keys are vehicles, values are tuples of (current time index,
            max time index)
        ds: dataset, from make_dataset
        nt: number of timesteps in batch
        lstm_units: number of LSTM units in model
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
    nveh = len(vehs)

    lead_inputs = []
    true_traj = []
    loss_weights = []
    for veh in vehs:
        t, tmax = vehs_counter[veh]
        leadpos, leadspeed = ds[veh]['lead posmem'], ds[veh]['lead speedmem']
        posmem = ds[veh]['posmem']
        curlead = []
        curtraj = []
        curweights = []
        for i in range(nt):
            if t+i < tmax:
                curlead.append([leadpos[t+i], leadspeed[t+i]])
                curtraj.append(posmem[t+i])
                curweights.append(1)
            else:
                curlead.append([0,0])
                curtraj.append(0)
                curweights.append(0)
        lead_inputs.append(curlead)
        true_traj.append(curtraj)

    return lead_inputs, true_traj, loss_weights

def train_step(x, y_true, sample_weight, model, loss_fn, optimizer):
    """Updates parameters from a single batch of examples.

    Args:
        x: input to model
        y_true: target for loss function
        sample_weight" weight for loss function
        model: tf.keras.Model
        loss_fn: function takes in y_true, y_pred, sample_weight, and returns the loss
        optimizer: tf.keras.optimizer
    Returns:
        y_pred: output from model
        hidden_state: hidden_state for model
    """
    with tf.GradientTape() as tape:
        y_pred, hidden_state = model(x)
        loss = loss_fn(y_true, y_pred, sample_weight = sample_weight)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients,model.trainable_variables))
    return y_pred, hidden_state


def training_loop(model, loss, optimizer, ds, nveh = 20, nt = 5, lstm_units = 20):
    """Trains model by repeatedly calling train_step.

    Args:

    """
    # initialization
    # select vehicles to put in the batch
    vehlist = list(ds.keys())
    np.random.shuffle(vehlist)
    vehs = vehlist[:nveh].copy()
    # vehs_counter stores current time index, maximum time index (length - 1) for each vehicle
    vehs_counter = {veh: (0, ds[veh]['IC'][1]-ds[veh]['IC'][0]) for veh in vehs}
    # make inputs for network
    init_state = [ds[veh]['IC'] for veh in vehs]
    hidden_states = [tf.zeros((nveh, lstm_units)),  tf.zeros((nveh, lstm_units))]
    lead_inputs, true_traj, loss_weights = make_batch(vehs, vehs_counter, ds, nveh, nt, lstm_units)

    for i in range(nbatches):
        veh_states, hidden_states = train_step([lead_inputs, init_state, hidden_states], true_traj,
                                               loss_weights, model, loss, optimizer)

        # check if any vehicles in batch have had their entire trajectory simulated
        cur_state = veh_states[:,-1]
        need_new_vehs = []
        for count, veh in enumerate(vehs):
            vehs_counter[veh][0] += nt
            if vehs_counter[veh][0] >= vehs_counter[veh][1]:
                need_new_vehs.append(count)
        # update vehicles in batch
        np.random.shuffle(vehlist)
        new_vehs = vehlist[:len(need_new_vehs)]
        for count, ind in enumerate(need_new_vehs):
            vehs[ind] = new_vehs[count]









    model.train_on_batch([lead_inputs, init_state, hidden_states], y = true_traj, sample_weight = loss_weights)



self.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate),
                     loss = tf.keras.losses.MeanSquaredError())




