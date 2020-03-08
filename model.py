import random
import numpy as np
import tflearn
import tensorflow as tf
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

keep_rate = 0.8
dqn_output_size = 5 # Might need to change this to the available state size for each action
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


weights = {}
biases = {}

def convolutionalNetworkModel(shape):

    # Set the shape to the input data shape
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

