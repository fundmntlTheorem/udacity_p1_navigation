import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

'''
    Starting with the model.py from the 
    Udacity\deep-reinforcement-learning\dqn\solution
'''
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, fc1_units=64, fc2_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class DuelingQNetwork(nn.Module):
    def __init__(self, state_size, action_size, fc1_units=64, fc2_units=64):
        '''
        Initialize parameters and build model.
        Based on https://arxiv.org/pdf/1511.06581.pdf
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        '''
        super().__init__()
        self.value_net = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(state_size, fc1_units)),
            ('relu1', nn.ReLU(inplace=True)),
            ('fc2', nn.Linear(fc1_units, fc2_units)),
            ('relu2', nn.ReLU(inplace=True)),
            # in this implementation, the only difference between the
            # networks is that the value network outputs a scalar
            ('fc3', nn.Linear(fc2_units, 1))
        ]))

        self.advantage_net = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(state_size, fc1_units)),
            ('relu1', nn.ReLU(inplace=True)),
            ('fc2', nn.Linear(fc1_units, fc2_units)),
            ('relu2', nn.ReLU(inplace=True)),
            ('fc3', nn.Linear(fc2_units, action_size))
        ]))

    def forward(self, state):
        '''
            Evaluate each network and combine using equation 9 from the paper.
        '''
        # batch_size x 1
        value = self.value_net(state).squeeze(0)
        # batch_size x action_size
        advantage = self.advantage_net(state)
        # average over the action dimension, batch_size x 1
        advantage_mean = torch.mean(advantage, dim=1).unsqueeze(1)

        return value + (advantage - advantage_mean)