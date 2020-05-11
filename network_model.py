import random
import numpy as np
import tflearn
import tensorflow as tf
import matplotlib.pyplot as plt
from tflearn.layers.conv import conv_3d, max_pool_3d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

# Model constants
keep_rate = 0.8
network_output_size = 6 # 6 potential outputs - B, A, LEFT, DOWN, RIGHT, NONE
learning_rate = 1e-3

def neuralNetworkModel():
    # Training in batches of n, where each batch is (4 x 101 x 46 x 3)
    network = input_data(shape=[None, 4, 101, 46, 3], name="input")

    network = conv_3d(network, nb_filter=16, filter_size=8, strides=4, activation="relu")
    network = max_pool_3d(network, 2)

    network = conv_3d(network, 32, 4, 2, activation="relu")
    network = max_pool_3d(network, 2)

    network = fully_connected(network, 512, activation="relu") # Check that this network input size is correct
    network = dropout(network, keep_rate)

    network = fully_connected(network, 2048, activation="relu")
    network = dropout(network, keep_rate)

    network = fully_connected(network, 256, activation="relu")
    network = dropout(network, keep_rate)

    network = fully_connected(network, network_output_size, activation="relu")
    network = regression(network, optimizer="adam", learning_rate=learning_rate, loss="categorical_crossentropy", name="targets")

    model = tflearn.DNN(network, tensorboard_dir="log")
    return model

def trainDQN(training_data, model=False):
    x = np.asarray([i[0] for i in training_data])
    # y = [i[1] for i in training_data]
    y = [i[1][3] for i in training_data] # Just used for old model that only predicts one output per prediction
    # 0 is first action, 3 is last action
    #y is now a list of lists of 4 move elements, e.g [  [ [0, 1, 0, 0, 0, 0] ... [move4] ], [ [] ... [] ] ... ]

    if not model:
        model = neuralNetworkModel()

    model.fit({"input": x}, {"targets": y}, n_epoch=5, snapshot_step=500, show_metric=True, run_id="PuyoOpenAI")

    return model


#TODO : Add storage for weights and biases
weights = {}
biases = {}