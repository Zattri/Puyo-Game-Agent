import argparse
import gym
import retro
import random
import time
import numpy as np

import experience_replay as ExpRep
import network_model as NetModel
import training_loop as TrainLoop

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--game', '-ga', default='Puyo-Genesis', help='the name or path for the game to run')
    parser.add_argument('--state', '-st', default=retro.State.DEFAULT, help='the initial state file to load, minus the extension')
    parser.add_argument('--difficulty', '-d', default=0, help='the difficulty stage of the game state')
    parser.add_argument('--scenario', '-sc', default='scenario', help='the scenario file to load, minus the extension')
    parser.add_argument('--record', '-rec', action='store_true', help='record bk2 movies')
    parser.add_argument('--obs-type', '-o', default='image', choices=['image', 'ram'], help='the observation type, either `image` (default) or `ram`')
    parser.add_argument('--players', '-p', type=int, default=1, help='number of players/agents (default: 1)')
    parser.add_argument('--model', '-m', default='models/model', help='model name, without .model extension')
    parser.add_argument('--rounds', '-ro', type=int, default=3, help='number of rounds the model will play')
    parser.add_argument('--verbose', '-v', type=bool, default=False, help='print verbose logging of actions, rewards and game steps')
    args = parser.parse_args()

    if args.state == "random":
        args.state = TrainLoop.getRandomState(args.difficulty)

    model = NetModel.neuralNetworkModel()
    model.load("models/" + args.model + "/" + args.model + '.model')

    exp_rep = ExpRep.ExperienceReplay()

    chosen_actions = []
    goal_steps = 10000
    current_play_time, last_play_time = None, None
    num_of_games = args.rounds

    obs_type = retro.Observations.IMAGE if args.obs_type == 'image' else retro.Observations.RAM
    env = retro.make(args.game, args.state, scenario=args.scenario, record=args.record, players=args.players, obs_type=obs_type)

    observations = []
    scores = []

    for game in range(num_of_games):
        observation = env.reset()

        for step in range(goal_steps):
            env.render()
            time.sleep(0.005)

            obs_img = observation[4: 206, 18: 110]
            compressed = exp_rep.compressObservation(obs_img)
            if len(observations) < 4:
                observations.append(compressed)

            if step % 4 == 0 and step != 0:
                observedFrames = np.asarray(observations)
                shapedArray = np.expand_dims(observedFrames, axis=0)
                prediction = model.predict(shapedArray)
                # print(prediction) - See what the prediction is
                action = model.predict(shapedArray)[0].astype(int)
                action_button = TrainLoop.parseNetworkOutputToString(action)
                #print(action_button)
                chosen_actions.append(action_button)

                observations.clear()

                if args.verbose:
                    debug_string = f"Ep {game} step {step}: {info} | {action_button} - {reward}"
                    print(debug_string)
                #else:
                    #print(action_button)
            else:
                action = np.zeros(12, "int8")

            observation_, reward, done, info = env.step(action)

            if step % 60 == 0:
                last_play_time = current_play_time
                current_play_time = info.get("play_time")

            if step >= goal_steps or done or last_play_time == current_play_time:
                scores.append(info.get("p1_score"))
                print(f"EP{game}: Player Score: {info.get('p1_score')}")
                break

            observation = observation_

    print(scores)

    env.close()

if __name__ == '__main__':
    main()