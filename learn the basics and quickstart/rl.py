import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
from torch import nn
from torch import optim as optim
from torch.nn import functional as F
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

#BATCH_SIZE is the number of transitions sampled from the replay buffer
#GAMMA is the discount factor
#EPS_START is the starting value of epsilon
#EPS_FINAL is the final value of epsilon
#EPS_DECAY controls the rate of decay: a higher value = slower decay
#TAU = update target
#LR = learning rate of AdamW (Adam doesn't have a learning rate, AdamW does)

BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

#get the number of actions
n_actions = env.action_space.n

#get the number of state observations
state, info = env.reset()
n_observations = len(state)

policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

#optimizer = the optimizer I'm using for the task
#lr = loss function
#amsgrad = an optimization method that helps fix adam's problem with convergence
#policy_net.parameters()
optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)

#memory is the memory of the system
memory = ReplayMemory(10000)

#steps_done = ???
steps_done = 0

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START-EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done +=1

    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1).indices.view(1,1)

    else:
        return torch.tensor([[env.action_space.sample()]],device=device, dtype=torch.long)

episode_durations = []

def plot_durations(show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title("Training...")
    plt.xlabel("Episode")
    plt.ylabel("Duration")
    plt.plot(durations_t.numpy())

    #Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0,100,1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99),means))
        plt.plot(means.numpy())

    plt.pause(0.001)
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)

    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements:
    # (a final state would be the one after the simulation ends)

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                        batch.next_state)), device=device, dtype = torch.bool)

    non_final_next_states = torch.cat([s for s in batch.next_state
                                        if s is not None
                            ])

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    #Compute Q(s_t, a) - the model computes Q(s_t) and we select the columns of each action taken
    # These are actions that would have been taken for each batch state according to the policy_net

    state_action_values = policy_net(state_batch).gather(1,action_batch)

    #Compute V(s_{t+1}) for all next states

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values

    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()

    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

if torch.cuda.is_available():
    num_episodes = 600
else:
    num_episodes = 50

for i_episode in range(num_episodes):
    #Initialize environment and get state

    state, info = env.reset()

    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

    for t in count():
        action = select_action(state)
        observation, reward, terminated, truncated, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)
        done = terminated or truncated

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        memory.push(state, action, next_state, reward)

        state = next_state

        optimize_model()

        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)

        target_net.load_state_dict(target_net_state_dict)

        if done:
            episode_durations.append(t+1)
            plot_durations()
            break

print("Complete")
plot_durations(show_result=True)
plt.ioff()
plt.show()
