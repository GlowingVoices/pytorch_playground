import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
from torch import nn
from torch import optim as optim
from torch import functional as F
import gymnasium as gym

print(torch)
print(gym)

env = gym.make("CartPole-v1")

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# A single transition - a change in the environment
# state =
# action = what the model did
# next_state
# reward = reward

Transition = namedtuple('Transition',
    ('state', 'action', 'next_state', 'reward'))


# Initialize the class known as replay memory
# Inherit from object
#
class ReplayMemory(object):

    #initialization function accepts a 'capacity'
    # CAPACITY IS HOW MUCH STUFF IT CAN REMEMBER
    def __init__(self, capacity):
        # create a double-ended queue (WE STUDIED THIS AT SHP :))
        self.memory = deque([], maxlen=capacity)


    # Remember a new event
    def push(self, *args):
        "Save a transition"
        self.memory.append(Transition(*args))

    # I'm taking a random sample from the model's memory?
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    # Getting the length of the model's memory
    def __len__(self):
        return len(self.memory)


# DETERMINISTIC ENVIRONMENT

#Maximizing the sum of the rewards.
#discount = a number that esures the sum converges
    #lower discount = far future is less important than near future
    #present & future tradeoff

#neural networks are *universal function approximators*
#https://en.wikipedia.org/wiki/Universal_approximation_theorem

class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128,128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self,x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
