import argparse
import gym
import retro
import random
import time
import numpy as np
import matplotlib.pyplot as plt
import json

import experience_replay as ExpRep
import network_model as NetModel


def getRandomState(difficulty=0):
    if difficulty == 0:
        stage = random.randint(1,3)
    else:
        stage = difficulty

    return f"p1_s{stage}_01"


def parseIntToActionArray(action):
    ar = np.zeros(12, "int8")

    if action == 0: #B
        ar[0] = 1
    elif action == 1: #A
        ar[1] = 1
    elif action == 2: #DOWN
        ar[5] = 1
    elif action == 3: #LEFT
        ar[6] = 1
    elif action == 4: #RIGHT
        ar[7] = 1

    return ar

def parseNetworkOutputToString(array):
    if array[0] == 1: #B
        return "B"
    elif array[1] == 1: #A
        return "A"
    elif array[2] == 1: #DOWN
        return "DOWN"
    elif array[3] == 1: #LEFT
        return "LEFT"
    elif array[4] == 1: #RIGHT
        return "RIGHT"
    elif array[5] == 1: #NONE
        return "NONE"


def parseIntToNetworkOutput(action):
    ar = np.zeros(6, "int8")
    ar[action] = 1

    return ar


def actionNumToString(action):
    if action == 0:
        return "B"
    elif action == 1:
        return "A"
    elif action == 2:
        return "DOWN"
    elif action == 3:
        return "LEFT"
    elif action == 4:
        return "RIGHT"
    else:
        return " "


def saveModel(model, model_name, model_path="models"):
    model_path = f"{model_path}/{model_name}/{model_name}.model"
    model.save(model_path)
    print(f"Saved model to {model_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--game', '-ga', default='Puyo-Genesis', help='the name or path for the game to run')
    parser.add_argument('--state', '-st', default=retro.State.DEFAULT, help='the initial state file to load, minus the extension')
    parser.add_argument('--difficulty', '-d', default=0, help='the difficulty stage of the game state')
    parser.add_argument('--scenario', '-sc', default='scenario', help='the scenario file to load, minus the extension')
    parser.add_argument('--obs-type', '-o', default='image', choices=['image', 'ram'], help='the observation type, either `image` (default) or `ram`')
    parser.add_argument('--players', '-p', type=int, default=1, help='number of players/agents (default: 1)')
    parser.add_argument('--model', '-m', help='model name for saving, without .model extension')
    parser.add_argument('--verbose', '-v', type=int, default=0, help='print verbose logging of actions, rewards and game steps')
    args = parser.parse_args()

    if args.state == "random":
        args.state = getRandomState(args.difficulty)


    # Retro Env Setup
    obs_type = retro.Observations.IMAGE if args.obs_type == 'image' else retro.Observations.RAM
    env = retro.make(args.game, args.state, scenario=args.scenario, players=args.players, obs_type=obs_type)


    # Class Objects
    exp_rep = ExpRep.ExperienceReplay()
    model_name = args.model


    # AI Settings
    frames_per_action = 8
    reward_threshold = 20
    obs_mem_size = 4
    action_mem_size = 4
    obs_record_rate = 8


    # Training Settings / Variables
    total_steps = 0
    goal_steps = 20000
    training_episodes = 1000
    training_data = []
    game_memory = []

    # Training Episodes | Goal Steps | Frames Per Action | Observation Record Rate | Reward Threshold | Observation Memory Size | Action Memory Size
    model_settings_string = f"TE: {training_episodes} | GS: {goal_steps} | FPA: {frames_per_action} | ORR: {obs_record_rate} | RT: {reward_threshold} | OMS: {obs_mem_size} | AMS: {action_mem_size}"

    # Training Loop
    for episode in range(training_episodes):
        observation = env.reset()
        current_play_time, last_play_time = None, None
        game_memory.clear()
        obs_memory = []
        action_memory = []

        if args.verbose == 2:
            print(f"Ep {episode} | Observations {len(training_data)}")

        for step in range(goal_steps):
            #env.render()

            if step % frames_per_action == 0:
                chosen_action = random.randint(0, 5)

                if len(action_memory) == action_mem_size:
                    action_memory.pop(0)

                action_memory.append(chosen_action)

            else:
                # Do nothing
                chosen_action = 5

            action = parseIntToActionArray(chosen_action)

            observation_, reward, done, info = env.step(action)

            if step % obs_record_rate == 0:
                # Button Mappings - ['B', 'A', 'MODE', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'C', 'Y', 'X', 'Z']
                # P1 Well - [4: 206, 18: 110] | P2 Well - [4: 206, 210: 302]
                obs_img = observation[4: 206, 18: 110]
                compressed = exp_rep.compressObservation(obs_img)

                if len(obs_memory) == obs_mem_size:
                    obs_memory.pop(0)

                obs_memory.append(compressed)

            if reward >= reward_threshold:
                compressed_array = np.asarray(obs_memory)
                game_memory.append([compressed_array[:], action_memory[:]])
                action_memory.clear()
                obs_memory.clear()
                #exp_rep.appendObservation(episode, step, info, action, reward, obs_img)

            if step % 60 == 0:
                last_play_time = current_play_time
                current_play_time = info.get("play_time")

            if step >= goal_steps or done or last_play_time == current_play_time:
                total_steps += step
                break

            if args.verbose == 1:
                if step % 1000 == 0:
                    debug_string = f"Ep {episode} step {step}: {info} | {action} - {reward}"
                    print(debug_string)

            observation = observation_

        for data in game_memory:
            # for k in range(0, 3):
            #     plt.imshow(game_memory[i][0][k])
            #     plt.show()

            arrayOfActions = []
            for actionNumber in data[1]:
                arrayOfActions.append(parseIntToNetworkOutput(actionNumber))
            #taken_action = parseIntToNetworkOutput(data[1])
            training_data.append([data[0], arrayOfActions])

    print(f"Captured Observations: {len(training_data)} | Episodes: {training_episodes}, Total Steps: {total_steps}")
    print(model_settings_string)
    #exp_rep.saveFile("data01")

    model = NetModel.trainDQN(training_data)

    if not args.model:
        model_name = input("Save model? (type name to save as or 'n' to not save)\n>>> ")

    if (model_name.lower() != "n"):
        saveModel(model, model_name)
    else:
        print("Model not saved, exiting program...")

    env.close()

if __name__ == '__main__':
    main()