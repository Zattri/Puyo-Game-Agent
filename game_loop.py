import argparse
import retro
import random
import time
import numpy as np
import matplotlib.pyplot as plt
import json

import experience_replay as ExpRep


def getRandomState(difficulty=0):
    if difficulty == 0:
        stage = random.randint(1,3)
    else:
        stage = difficulty

    return f"p1_s{stage}_01"


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
    goal_steps = 5000
    training_episodes = 1

    for episode in range(training_episodes):
        observation = env.reset()
        current_play_time, last_play_time = None, None
        for step in range(goal_steps):
            env.render()
            # Button Mappings - ['B', 'A', 'MODE', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'C', 'Y', 'X', 'Z']
            # Actions currently being made randomly from a selection of available actions
            action = env.action_space.sample()
            observation_, reward, done, info = env.step(action)

            # IMPORTANT - Game moves a puyo every 4 frames

            if step % 4 == 0:
                #[4: 206, 18: 110] for P1 Well
                obs_img = observation[4: 206, 210: 302]
                # Obs_img.shape = (202, 92, 3)
                exp_rep.appendObservation(episode, step, info, action, reward, obs_img)
                debug_string = f"Ep {episode} step {step}: {info} | {action} - {reward}"
                print(debug_string)

            if step % 60 == 0:
                last_play_time = current_play_time
                current_play_time = info.get("play_time")

            if step >= goal_steps or done or last_play_time == current_play_time:
                total_steps += step
                break

            observation = observation_

    print(f"Captured Observations: {len(exp_rep.observations)} | Episodes: {training_episodes}, Total Steps: {total_steps}")
    #exp_rep.saveFile("data01")

    img = exp_rep.getObservation(-1, -1)
    plt.imshow(img)
    plt.show()

if __name__ == '__main__':
    main()