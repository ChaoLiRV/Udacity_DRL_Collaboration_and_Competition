[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135623-e770e354-7d12-11e8-998d-29fc74429ca2.gif "Trained Agent"
[image2]: https://user-images.githubusercontent.com/10624937/42135622-e55fb586-7d12-11e8-8a54-3c31da15a90a.gif "Soccer"


# Project 3: Collaboration and Competition

![Trained Agent][image1]

### Overview

This is the 3rd project for the Udacity Deep Reinforcement Learning Nanodegree program. The goal of this project is to use deep neural network to solve a multi-agent task, in a collaborative or competitive manner. For this project, the [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment is utilized.

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.  Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 24 dimensions corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation.  Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping. 

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single **score** for each episode.

The environment is solved at **1126** episodes with the code attached, when the average (over 100 episodes) of those **scores** is at least +0.5.

### File Instruction
_**Tennis.ipynb**_: This is the python jupyter notebook file which performs the followings:
1. Start the environment and examine the state and action space
2. Take random actions in the environement to become familar with the environment API
3. Train the agent using deep neural network to solve the environement

_**agent.py**_: The python code to implement the [Multi-agent deep deterministic policy agent (MADDPG) algorithm](https://arxiv.org/pdf/1706.02275.pdf). The MADDPG is a multi-agent variant of DDPG, a model-free, off-policy, policy gradient-based algorithm. Each agent has its own actor but share the critic.	

_**model.py**_: The python code to configure the neural network for both actor and critic.

_**checkpoint_actor|critic_agent0|1.pth**_: Both agents' actor and critic model weights are saved in the checkpoint file

### Requirements
1. Install Anaconda distribution of python3

2. Install PyTorch, Jupyter Notebook in the Python3 environment

3. Download the environment from one of the links below and place it in the same directory folder. Only select the environment that matches the operating system:
- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)

(_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

(_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux_NoVis.zip) to obtain the "headless" version of the environment.  You will **not** be able to watch the agent without enabling a virtual screen, but you will be able to train the agent.  (_To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above._)

4. Open Jupyter Notebook and run the Tennis.ipynb file to train the agent. 

5. To watch the agents to play tennis, load the model weights from the 4 checkpiont files by executing all the notebook cells except **Training** session.