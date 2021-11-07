[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

# Udacity Navigation Project 1

### Introduction

The goal of this project is to train an agent using the Deep Q Network algorithm, to collect blue bananas, and avoid blue bananas in a large, square world.  A reward of +1 is given for collecting a yellow banana, and a reward of -1 is given for collecting a blue banana.  The gif below depicts a trained agent carrying out its task.  The environment is considered solved when the agent is able to achieve an average score of +13 over 100 consecutive episodes.  See the Report.pdf for more details on the algorithms that I tried to implement, with varying amounts of success/failure.

![Trained Agent][image1]

### Environment Details

The environment is provided by [Unity](https://unity.com/), a company that specializes in building worlds that can be used for video game development, simulation, animation, and architecture/design.  The following is the description of the state space and actions available to the agent:

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

### Dependencies

If 64x Windows is being used, the repo already contains the correct environment.  Other operating systems may download the correct environment from the list below.

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)

2. The code expects the Banana.exe file to be located in the following directory of the repo  "./Banana_Windows_x86_64/Banana.exe"

3. Create a new conda environment with the provided requirements.txt file. Ex. conda create --name <env> --file requirements.txt

## Using the Code

The code may be run using the command `python navigator.py <config.json file>` for training mode, or `python navigator.py <network .pth file>` for running mode.  

### Training Mode

Example `python navigator.py config.json`

The arguments to the navigator program are provided using a .json file, with the following structure:

```python
{
    "base_name": "final_model",     # A base file name given to the output networks.  For hyper-parameter tuning, each output network is post-pended with an integer 0, 1, ..., k
    "algorithms": [],               # list of 0 or more algorithm features, from this list : ["double_dqn", "prioritized_replay"].  The base behavior uses DQN
    "replay_buffer_size": 1e5,      # size of the replay buffer from which transitions are sampled
    "batch_size": 64,               # size of sampling batch siz
    "gamma": 0.99,                  # Reward discount factor
    "tau": 1e-3,                    # Q-learning update factor
    "learning_rate": 1e-4,          # Network training update factor
    "learn_every": 4,               # Periodic number of transitions to gather before training
    "num_episodes": 2000,           # Total number of episodes to run
    "max_time": 1000,               # Maximum number of time steps in an episode
    "eps_start": 1.0,               # Epsilon starting value, for Epsilon-Greedy policy evaluation
    "eps_end": 0.01,                # Epsilon stop value
    "eps_decay": 0.995,             # Factor by which to decrease epsilon on each episode
    "alpha": 0.5,                   # Priority factor, used for prioritized replay.  alpha=0 samples transitions from a uniform distribution, while alpha=1 uses sampling proportional to error
    "beta": 0.5                     # Priority weight scaling factor.  Beta = 0 turns off any weight scaling, while Beta=1 provides full compensation for the bias in prioritized sampling
}

```

Any parameter enclosed in list brackets will iterate its parameters.  For example the following argument will train the agent with 4 different values of the learning rate.

```python
"learning_rate": [1e-5, 1e-4, 1e-3, 1e-2],
```

Using a list of parameters for multiple arguments will execute training runs for the cartesian product of the argument lists.  For example, if the arguments below would lead to 12 different training runs.  Note how the algorithms are specified with lists of 0 or more algorithmic features.

```python
"algorithms": [
    [], ['double_dqn'], ['double_dqn', 'prioritized_replay']
], 
"learning_rate": [1e-5, 1e-4, 1e-3, 1e-2],
```

### Running Mode

Example `python navigator.py final_model_0.pth`

In this mode, the agent is initialized using a saved network file, so you can watch it collect bananas!



