import argparse
import retro
import time
import numpy as np
import matplotlib.pyplot as plt
# Need to import models later


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

goal_steps = 10000
training_episodes = 3
current_episode = 1
observedImages = []
experiences = []

for episode in range(training_episodes):
    observation = env.reset()
    current_play_time, prev_play_time = None, None
    for step in range(goal_steps):
        env.render()
        # Button Mappings - ['B', 'A', 'MODE', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'C', 'Y', 'X', 'Z']
        action = env.action_space.sample()
        observation_, reward, done, info = env.step(action)

        # Store the observations here - Need to store things in memory including reward and action
        # Make a list of frames every 6 steps, take an action on it based on what the model thinks it should do and record the reward

        # Doing this every available movement
        # IMPORTANT - Game moves a puyo every 4 frames
        if step % 15 == 0:
            observedImages.append(observation[4: 208, 208: 304])
            infoString = f"Episode {episode} - step {step}: {info} | {action}"
            print(infoString)

        observation = observation_

        # Doing this every second
        if step % 60 == 0:
            # Hybrid done condition
            prev_play_time = current_play_time
            current_play_time = info.get("play_time")

        if step >= goal_steps or done or prev_play_time == current_play_time:
            break
        # Can do some check that logs the info.gameTime and checks if it changed since last frame, if its the same, done

finishedString = f"Captured Observations: {len(observedImages)} | Episodes: {training_episodes}"
print(finishedString)

# plt.imshow(observedImages[-8])
# plt.show()
# plt.imshow(observedImages[-7])
# plt.show()
# plt.imshow(observedImages[-6])
# plt.show()
# plt.imshow(observedImages[-5])
# plt.show()
# plt.imshow(observedImages[-4])
# plt.show()
# plt.imshow(observedImages[-3])
# plt.show()
# plt.imshow(observedImages[-2])
# plt.show()
sampleImage = observedImages[-1]
plt.imshow(sampleImage)
plt.show()
