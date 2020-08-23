
"""
@author: rlk268@cornell.edu
"""

import tensorflow as tf
import numpy as np
from havsim.plotting import compute_headway2

def make_dataset(meas, platooninfo, h = .1):
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
        headway = compute_headway2(veh, meas, platooninfo, h = h)
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
        training[veh] = [IC, [t1,t2], vehpos, leadpos, leadspeed]

    testing = {}
    for veh in test_veh:
        headway = compute_headway2(veh, meas, platooninfo, h = h)
        t0, t1, t2, t3 = platooninfo[veh][:4]
        lead = platooninfo[veh][4][0]
        lt0, lt1, lt2, lt3 = platooninfo[lead][:4]
        leadspeed = meas[lead][t1-lt0:t2+1-lt0,3]
        vehpos = meas[veh][t1-t0:t2+1-t0,2]
        leadpos = headway + vehpos
        IC = [meas[veh][t1-t0,2], meas[veh][t1-t0,3]]

        testing[veh] = [IC, [t1,t2], vehpos, leadpos, leadspeed]

    return training, testing, maxheadway, maxspeed, minacc, maxacc


class RNNCFModel(tf.keras.Model):
    def __init__(self, learning_rate = .001):
        super().__init__()
        self.mask = tf.keras.layers.Masking(mask_value=0)
        self.lstm_cell1 = tf.keras.layers.LSTMCell(20)
        self.dense1 = tf.keras.layers.Dense(1)

        self.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate))

    def call(self, x):
        x = self.lstm_cell1(x)
        x = self.dense1(x)
        return x

def training_loop(nveh = 20, nt = 5):
    # select vehicles to put in the batch






