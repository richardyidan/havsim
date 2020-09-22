# imports and load data
from havsim.calibration import calibration
import pickle
import numpy as np
import tensorflow as tf
import math
import time

try:
    with open('/Users/nathanbala/Desktop/MENG/havsim/data/recon-ngsim.pkl', 'rb') as f:
        meas, platooninfo = pickle.load(f) #load data
except:
    with open('/home/rlk268/havsim/data/recon-ngsim.pkl', 'rb') as f:
        meas, platooninfo = pickle.load(f) #load data

curplatoon = [1013, 1023, 1030, 1037, 1045]
pguess =  [40,1,1,3,10,25] #[80,1,15,1,1,35] #
mybounds = [(20,120),(.1,5),(.1,35),(.1,20),(.1,20),(.1,75)]
cal = calibration.make_calibration(curplatoon, meas, platooninfo, .1, calibration.CalibrationVehicle)
start = time.time()
cal.simulate(pguess)
