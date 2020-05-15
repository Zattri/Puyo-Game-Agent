import argparse
import numpy as np
import matplotlib.pyplot as plt
import json

import training_loop as TrainLoop
import experience_replay as ExpRep
import network_model as NetModel
import normaliser as Normaliser


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', help='model name for saving, without .model extension')
    parser.add_argument('--verbose', '-v', type=int, default=0, help='print verbose logging of actions, rewards and game steps')
    args = parser.parse_args()

    model_name = args.model
    replay_files = ["test", "test2", "testing", "test3"]

    training_data = []
    replay_data = []

    for fileName in replay_files:
        replay_data = replay_data + ExpRep.readFile(fileName)

    normalised_data = Normaliser.normaliseActionsFromFile(replay_data)

    for data in normalised_data:
        arrayOfActions = []
        for actionNumber in data[1]:
            arrayOfActions.append(TrainLoop.parseIntToNetworkOutput(actionNumber))
        training_data.append([data[0], arrayOfActions])

    model = NetModel.trainDQN(training_data)

    if not args.model:
        model_name = input("Save model? (type name to save as or 'n' to not save)\n>>> ")

    if (model_name.lower() != "n"):
        TrainLoop.saveModel(model, model_name)
    else:
        print("Model not saved, exiting program...")

if __name__ == '__main__':
    main()