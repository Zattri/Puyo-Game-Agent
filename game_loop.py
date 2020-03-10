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
parser.add_argument('--verbose', '-v', action='count', default=1, help='increase verbosity (can be specified multiple times)')
parser.add_argument('--quiet', '-q', action='count', default=0, help='decrease verbosity (can be specified multiple times)')
parser.add_argument('--obs-type', '-o', default='image', choices=['image', 'ram'], help='the observation type, either `image` (default) or `ram`')
parser.add_argument('--players', '-p', type=int, default=1, help='number of players/agents (default: 1)')
args = parser.parse_args()

obs_type = retro.Observations.IMAGE if args.obs_type == 'image' else retro.Observations.RAM
env = retro.make(args.game, args.state, scenario=args.scenario, record=args.record, players=args.players, obs_type=obs_type)
verbosity = args.verbose - args.quiet

goal_steps = 10000
training_episodes = 10
current_episode = 1
observedImages = []
experiences = []

for episode in range(training_episodes):
    observation = env.reset()
    current_play_time = None
    for step in range(goal_steps):
        env.render()
        # Button Mappings - ['B', 'A', 'MODE', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'C', 'Y', 'X', 'Z']
        action = env.action_space.sample()
        observation_, reward, done, info = env.step(action)
        # Store the observations here - Need to store things in memory including reward and action
        # Make a list of frames every 6 steps, take an action on it based on what the model thinks it should do and record the reward
        observedImages.append(observation[4: 208, 208: 304])
        observation = observation_

        # Doing this every second
        # IMPORTANT - Game moves a puyo every 4 frames
        if step % 60 == 0:
            infoString = f"Episode {episode} - step {step}: {info} | {action}"
            print(infoString)

            # Hybrid done condition
            prev_play_time = current_play_time
            current_play_time = info.get("play_time")

        if step >= goal_steps or done or (prev_play_time == current_play_time):
            break
        # Can do some check that logs the info.gameTime and checks if it changed since last frame, if its the same, done

print("Captured Observation Num:", len(observedImages))
sampleImage = observedImages[len(observedImages)-1]
plt.imshow(sampleImage)
plt.show()