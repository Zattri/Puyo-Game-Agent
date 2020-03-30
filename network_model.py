import random
import numpy as np
import tflearn
import tensorflow as tf
from tflearn.layers.conv import conv_3d, max_pool_3d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

keep_rate = 0.8
dqn_output_size = 6
learning_rate = 1e-3

def neuralNetworkModel():
    network = input_data(shape=[None, 4, 101, 46, 3], name="input") # Training in batches of 4, where each batch it 101 x 46 x 3

    network = conv_3d(network, nb_filter=16, filter_size=8, strides=4, activation="relu")
    network = max_pool_3d(network, 2)

    network = conv_3d(network, 32, 4, 2, activation="relu")
    network = max_pool_3d(network, 2)

    network = fully_connected(network, 512, activation="relu")
    network = dropout(network, keep_rate)

    network = fully_connected(network, 2048, activation="relu")
    network = dropout(network, keep_rate)

    network = fully_connected(network, 256, activation="relu")
    network = dropout(network, keep_rate)

    network = fully_connected(network, dqn_output_size, activation="softmax")
    network = regression(network, optimizer="adam", learning_rate=learning_rate, loss="categorical_crossentropy", name="targets")

    model = tflearn.DNN(network, tensorboard_dir="log")
    return model


def trainDQN(training_data, model=False):
    x = np.expand_dims([i[0] for i in training_data], axis=0)
    obs_num = x.shape[1]
    print(f"X Shape: {x.shape} | Total obs: {obs_num}")

    batches = obs_num // 4
    x = x.reshape(batches, 4, 101, 46, 3)

    print(f"New x shape: {x.shape}")

    y = [i[1] for i in training_data]

    if not model:
        model = neuralNetworkModel()

    model.fit({"input": x}, {"targets": y}, n_epoch=5, snapshot_step=500, show_metric=True, run_id="PuyoOpenAI")

    return model


#TODO : Add storage for weights and biases
weights = {}
biases = {}