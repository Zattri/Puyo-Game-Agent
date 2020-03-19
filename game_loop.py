import argparse

import gym
import retro
import random
import time
import numpy as np
import matplotlib.pyplot as plt
import json

import experience_replay as ExpRep
import model as Model


def getRandomState(difficulty=0):
    if difficulty == 0:
        stage = random.randint(1,3)
    else:
        stage = difficulty

    return f"p1_s{stage}_01"


def parseIntToActionArray(action):
    ar = np.zeros(12, "int8")
    if action == 0:
        ar[0] = 1
        return ar
    elif action == 1:
        ar[1] = 1
        return ar
    elif action == 2:
        ar[5] = 1
        return ar
    elif action == 3:
        ar[6] = 1
        return ar
    elif action == 4:
        ar[7] = 1
        return ar
    elif action == 5:
        return ar


def parseIntToNetworkOutput(action):
    ar = np.zeros(6, "int8")
    ar[action] = 1

    return ar


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--game', '-ga', default='Puyo-Genesis', help='the name or path for the game to run')
    parser.add_argument('--state', '-st', default=retro.State.DEFAULT, help='the initial state file to load, minus the extension')
    parser.add_argument('--difficulty', '-d', default=0, help='the difficulty stage of the game state')
    parser.add_argument('--scenario', '-sc', default='scenario', help='the scenario file to load, minus the extension')
    parser.add_argument('--record', '-r', action='store_true', help='record bk2 movies')
    parser.add_argument('--obs-type', '-o', default='image', choices=['image', 'ram'], help='the observation type, either `image` (default) or `ram`')
    parser.add_argument('--players', '-p', type=int, default=1, help='number of players/agents (default: 1)')
    args = parser.parse_args()

    if args.state == "random":
        args.state = getRandomState(args.difficulty)

    obs_type = retro.Observations.IMAGE if args.obs_type == 'image' else retro.Observations.RAM
    env = retro.make(args.game, args.state, scenario=args.scenario, record=args.record, players=args.players, obs_type=obs_type)

    exp_rep = ExpRep.ExperienceReplay()

    total_steps = 0
    goal_steps = 10000
    score_requirement = 1000
    training_episodes = 1

    training_data = []

    for episode in range(training_episodes):
        observation = env.reset()
        current_play_time, last_play_time = None, None
        game_memory = []

        for step in range(goal_steps):
            #env.render()

            action_num = random.randint(0, 5)
            action = parseIntToActionArray(action_num)

            observation_, reward, done, info = env.step(action)

            if step % 4 == 0:
                # Button Mappings - ['B', 'A', 'MODE', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'C', 'Y', 'X', 'Z']
                # P1 Well - [4: 206, 18: 110] | P2 Well - [4: 206, 210: 302]
                obs_img = observation[4: 206, 18: 110]
                # Obs_img.shape = (202, 92, 3)
                compressed = exp_rep.compressObservation(obs_img).flatten()

                # TODO: Need to append observations in episodes not just loads of observations
                #exp_rep.appendObservation(episode, step, info, action, reward, obs_img)
                game_memory.append([compressed, action_num])

            if step % 60 == 0:
                debug_string = f"Ep {episode} step {step}: {info} | {action} - {reward}"
                print(debug_string)
                last_play_time = current_play_time
                current_play_time = info.get("play_time")

            if step >= goal_steps or done or last_play_time == current_play_time:
                total_steps += step
                break

            observation = observation_

        for data in game_memory:
            taken_action = parseIntToNetworkOutput(data[1])
            training_data.append([data[0], taken_action])

    print(f"Captured Observations: {len(training_data)} | Episodes: {training_episodes}, Total Steps: {total_steps}")
    #exp_rep.saveFile("data01")

    model = Model.trainDQN(training_data)
    #model.save('modelName.model')
    env.close()

    playGame(model, args)


def playGame(model, args):
    choices = []
    goal_steps = 10000
    total_steps = 0
    current_play_time, last_play_time = None, None
    num_of_games = 3

    exp_rep = ExpRep.ExperienceReplay()

    obs_type = retro.Observations.IMAGE if args.obs_type == 'image' else retro.Observations.RAM
    env = retro.make(args.game, args.state, scenario=args.scenario, record=args.record, players=args.players, obs_type=obs_type)

    for each_game in range(num_of_games):
        observation = env.reset()

        for step in range(goal_steps):
            env.render()
            time.sleep(0.005)

            if step % 4 == 0:
                obs_img = observation[4: 206, 18: 110]
                compressed = exp_rep.compressObservation(obs_img).flatten()
                action = model.predict(compressed.reshape(-1, len(compressed), 1))[0].astype(int)
                choices.append(env.get_action_meaning(action))
                print(choices[-1])
            else:
                action = np.zeros(12, "int8")

            observation_, reward, done, info = env.step(action)

            if step % 60 == 0:
                last_play_time = current_play_time
                current_play_time = info.get("play_time")

            if step >= goal_steps or done or last_play_time == current_play_time:
                total_steps += step
                break

            observation = observation_

    env.close()

if __name__ == '__main__':
    main()