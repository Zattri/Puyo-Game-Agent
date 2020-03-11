# puyo-gym
Puyo Puyo Tsu - Dynamic Game Agent with Difficulty Adjustment Functionality using OpenAI Gym Retro

## Project Dependencies:

### Python3 (V3.7+):
Install: Google it

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


## Integration Setup Steps:
```
sudo apt-get install capnproto libcapnp-dev libqt5opengl5-dev qtbase5-dev
cmake .
make
./gym-retro-integration
```
