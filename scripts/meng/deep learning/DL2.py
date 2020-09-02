
"""
@author: rlk268@cornell.edu
"""

# TODO really this function should just take in a vehicle list,
    # the data, and return the dataset, and normalization if desired.
    nolc_list = []
    for veh in meas.keys():
        temp = nolc_list.append(veh) if len(platooninfo[veh][4]) == 1 else None
    np.random.shuffle(nolc_list)
    train_veh = nolc_list[:-100]
    test_veh = nolc_list[-100:]




if __name__ == '__main__':
    training, testing, maxhd, maxv, mina, maxa = make_dataset(meas, platooninfo)
    model = RNNCFModel(maxhd, maxv, mina, maxa)
    loss = masked_MSE_loss
    opt = tf.keras.optimizers.Adam(learning_rate = .001)

    training_loop(model, loss, opt, training, nbatches = 1000, nveh = 32, nt = 100, lstm_units = 20)
    training_loop(model, loss, opt, training, nbatches = 1000, nveh = 32, nt = 200, lstm_units = 20)
    training_loop(model, loss, opt, training, nbatches = 1000, nveh = 32, nt = 300, lstm_units = 20)
    training_loop(model, loss, opt, training, nbatches = 1000, nveh = 32, nt = 500, lstm_units = 20)

    model.save_weights('trained LSTM')
    #maxhd, maxv, mina, maxa = (845.00927900, 88.571525144, -46.5518387, 25.27559136)

    out = generate_trajectories(model, list(testing.keys()), testing, loss = loss)
    out2 = generate_trajectories(model, list(training.keys()), training, loss = loss)




