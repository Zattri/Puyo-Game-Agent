# AI Game Agent with Difficulty Adjustment
An AI agent was developed for the video game Puyo Puyo Tsu (a competitive Tetris block breaking game) using the OpenAI Gym playtform for retro game emulation and AI programming. This project was made for my final year project of my BSc with the goal of developing an agent to use only visual information to make its decisions, and have a difficulty scaling element allowing it to be tailored to a player's skill level. It utilised a convolutional neural network and simple prediction based system for generating game inputs, which were then staggered or delayed artificially to adjust the speed of the AI for different difficulty settings.

Images were captured from the game environment and pre-processed to extract key information as shown below:
<div style="text-align: center">

Raw game footage             |  Processed Image
:---------------------------:|:-------------------------:
![image](https://user-images.githubusercontent.com/9122074/210259501-ef8beed1-3a24-477b-9b9e-e99ff909a5b4.png) | ![image](https://user-images.githubusercontent.com/9122074/210259557-0b5f0fc3-510f-476d-8851-5c2239cbc7dd.png)

</div>

## Training Platform
There is also an additional repo coupled with this project that creates a training platform for gathering data from players to train the game agent model, found here - https://github.com/Zattri/Puyo-Game-Agent-Training

## Project Dependencies:

Linux operating system only (sorry Windows guys)
### Python3 (V3.7+)
See - https://www.python.org/downloads/release/python-370/

### cmake:
Used for gym-retro integration UI
```
sudo apt-get install cmake
```

### Capnproto:
Used for gym-retro integration UI
```
sudo apt-get install capnproto
```

### pip:
```
sudo apt-get install python3-pip

pip3 install <package>
```
- gym
- gym-retro
- tensorflow-gpu (or tensorflow for non-gpu)
- numpy (just for good measure)
- matplotlib (same as above)
- scikit-image

## Integration Setup Steps:
```
sudo apt-get install capnproto libcapnp-dev libqt5opengl5-dev qtbase5-dev
cmake .
make
./gym-retro-integration
```

## Setup Local Game Env
Move `Puyo-Genesis` folder in repo to `/home/<username>/.local/lib/python3.7/site-packages/retro/data/stable`
