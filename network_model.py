import random
import numpy as np
import tflearn
import tensorflow as tf
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

keep_rate = 0.8
dqn_output_size = 6
learning_rate = 1e-3

def neuralNetworkModel(inputSize):
    network = input_data(shape=[None, inputSize, 1], name="input")

    network = fully_connected(network, 2048, activation="relu")
    network = dropout(network, keep_rate)

    network = fully_connected(network, 256, activation="relu")
    network = dropout(network, keep_rate)

    network = fully_connected(network, dqn_output_size, activation="softmax")
    network = regression(network, optimizer="adam", learning_rate=learning_rate, loss="categorical_crossentropy", name="targets")

    model = tflearn.DNN(network, tensorboard_dir="log")
    return model


def trainDQN(training_data, model=False):
    print(training_data[0][0])
    x = np.array([i[0] for i in training_data]).reshape(-1, len(training_data[0][0]), 1)
    y = [i[1] for i in training_data]

    if not model:
        print(f"Shape = {len(x[0])}")
        model = neuralNetworkModel(inputSize=len(x[0]))

    model.fit({"input": x}, {"targets": y}, n_epoch=3, snapshot_step=500, show_metric=True, run_id="PuyoOpenAI")

    return model


#TODO : Add storage for weights and biases
weights = {}
biases = {}

def convolutionalNetworkModel(shape):

    #TODO: Change to 3D and set the shape to the shape of the compressed image
    #TODO: Start using this as network input instead of the DQN taking input in
    convnet = input_data(shape=[None, 28, 28, 1], name="input")

    convnet = conv_2d(convnet, nb_filter=16, filter_size=8, strides=4, activation="relu")
    convnet = max_pool_2d(convnet, 2)

    convnet = conv_2d(convnet, 32, 4, 2, activation="relu")
    convnet = max_pool_2d(convnet, 2)

    convnet = fully_connected(convnet, 512, activation="relu")
    convnet = dropout(convnet, keep_rate)

    convnet = fully_connected(convnet, 512, activation="softmax") # Don't know if we want to softmax or just pass the input straight into DQN
    convnet = regression(convnet, optimizer="adam", learning_rate="0.01", loss="categorical_crossentropy")

    model = tflearn.DNN(convnet)

    return model

