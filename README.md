# RoBoost

## 🥇 **Winner of the EUROPE EMBODIED 48h Hackathon**

### Overall Winner and Winner for Intel's Industrial Arm Challenge.

## The idea

Robots driven by Vision Language Models (VLMs) can *see* objects, but they don't know what those objects *feel* like. Two bottles can look exactly the same while one is full and the other is empty.

RoBoost lets a robot use physical properties as an input alongside vision. In this case, that property is the weight of an object. The robot can then make decisions based on what it sees *and* on what it senses.

## How it works

1. **See:** A depth camera looks at the workspace and finds the objects in it, such as bottles and cans.
2. **Grab:** The robot arm moves to an object and picks it up.
3. **Feel:** While holding the object, the robot measures how heavy it is.
4. **Act:** The robot uses that weight to decide what to do with the object, for example where to put it down.

## Project layout

- `main.py`: the main program that ties vision, the arm and networking together
- `live.py`: a live camera view with object detection
- `listen.py`: a small tool for checking the data sent over the network
- `libraries/`: the vision, arm and networking code
- `setup.sh`: installs the Intel RealSense camera drivers
- `requirements.txt`: Python dependencies

## Getting started

```bash
./setup.sh
pip install -r requirements.txt
python main.py
```

You need an Intel RealSense camera and the robot arm connected for the full system to run.

## Recognition

RoBoost won first place and was awarded €2000 in cash and €5000 in Claude credits.
