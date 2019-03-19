[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

# Project 1: Navigation

### Introduction

For this project, I have trained an agent to navigate (and collect bananas!) in a large, square world.  

![Trained Agent][image1]

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, my agent will get an average score of +15 over 100 consecutive episodes, although the environmeent is considered 'solved' with an average score of 13+.

### Getting Started

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    
    (_For Windows users_) A Windows folder for the agent is included in this repo (Banana_Windows_x86_64). In that folder, follow the link in the README to download the environment files for a Windows agent.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.


### Instructions
In order to run the scripts and train an agent you will need to install a few dependencies. We need Pytorch to build our neural nets, UnityAgents to interact with the 3D Unity environment.

Numpy - https://www.scipy.org/scipylib/download.html
Numpy provides a very useful N-dimensional array object. Visit the Numpy download page for installation instructions.

Pytorch - https://pytorch.org/
Pytorch is the library used to build our neural nets. The full list of install options is on Pytorch home page.

Or if you're on Windows with Python 3.6, and CUDA 9 (like me) you can easily use pip to install Pytorch
pip3 install https://download.pytorch.org/whl/cu90/torch-1.0.1-cp36-cp36m-win_amd64.whl
pip3 install torchvision

Unity ML-Agents - https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Installation.md
You'll need Unity ML-Agents, which is the API that allows us to to communicate with the standalone Unity environment. The install options are on the Unity ml-agents Github page.

Matplotlib - https://matplotlib.org/users/installing.html#installing-an-official-release
To graph the learning history of the agent we'll use Matplotlib. Follow the link for installation instructions.


### How to train the agent
A training script is provided in Report.ipynb.  Once the dependencies are in place simply run the cell titled **Train the Agent** to watch the agent train. The 3D training environment will launch and you can watch the agent as it trains, and the training score history is also printed as the agent trains.


### How to watch a trained agent
Weights for a trained agent are included in the repo, so you do not need to first train an agent to watch a trained agent perform. Assuming the required dependencies are in place just run the cell titled **Watch a trained agent!**.  The 3D Unity environment will load and the trained agent will begin collecting yellow bananas while avoiding the blue ones. Cool!
