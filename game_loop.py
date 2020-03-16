import argparse
import retro
import time
import numpy as np
import matplotlib.pyplot as plt
import json
import skimage.measure

import experience_replay as ExpRep

parser = argparse.ArgumentParser()
parser.add_argument('--game', default='Puyo-Genesis', help='the name or path for the game to run')
parser.add_argument('--state', default=retro.State.DEFAULT, help='the initial state file to load, minus the extension')
parser.add_argument('--scenario', '-s', default='scenario', help='the scenario file to load, minus the extension')
parser.add_argument('--record', '-r', action='store_true', help='record bk2 movies')
parser.add_argument('--obs-type', '-o', default='image', choices=['image', 'ram'], help='the observation type, either `image` (default) or `ram`')
parser.add_argument('--players', '-p', type=int, default=1, help='number of players/agents (default: 1)')
args = parser.parse_args()

obs_type = retro.Observations.IMAGE if args.obs_type == 'image' else retro.Observations.RAM
env = retro.make(args.game, args.state, scenario=args.scenario, record=args.record, players=args.players, obs_type=obs_type)
exp_rep = ExpRep.ExperienceReplay()

total_steps = 0
goal_steps = 5000
training_episodes = 1
observations = []
experiences = []

for episode in range(training_episodes):
    observation = env.reset()
    current_play_time, last_play_time = None, None
    for step in range(goal_steps):
        env.render()
        # Button Mappings - ['B', 'A', 'MODE', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'C', 'Y', 'X', 'Z']
        # Actions currently being made randomly from a selection of available actions
        action = env.action_space.sample()
        observation_, reward, done, info = env.step(action)

        # Make a list of frames every 4 steps, take an action on it based on what the model thinks it should do and record the reward

        # IMPORTANT - Game moves a puyo every 4 frames
        # Initial Crop - [4: 208, 208: 304]

        if step % 4 == 0:
            obs_img = observation[4: 206, 210: 302]
            # Obs Shape = (204, 96, 3)
            exp_rep.appendObservation(episode, step, info, action, reward, obs_img)
            infoString = f"Ep {episode} step {step}: {info} | {action} - {reward}"
            print(infoString)

        if step % 60 == 0:
            last_play_time = current_play_time
            current_play_time = info.get("play_time")

        if step >= goal_steps or done or last_play_time == current_play_time:
            total_steps += step
            break

        observation = observation_

print(f"Captured Observations: {len(exp_rep.observations)} | Episodes: {training_episodes}, Total Steps: {total_steps}")
#exp_rep.saveFile("data01")

sampleImage = exp_rep.getObservation(-1, -1)
# test = skimage.measure.block_reduce(sampleImage, (2, 2, 1), np.max)
# plt.imshow(test)
# plt.show()
plt.imshow(sampleImage)
plt.show()
