import torch
#the learnable parameters are stored in model.parameters()
#a state_dict is a python dictionary object that maps each layer to its parameter tensor
#a state_dict is exactly what it sounds like: a dictionary that details the model state.

import torch.nn as nn
import torch.functional as F
import torch.optim as optim

class Net (nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        #conv2d layer
        self.conv1 = nn.Conv2d(3,6,5)

        #pool
        self.pool = nn.MaxPool2d(2,2)

        #conv2
        self.conv2 = nn.Conv2d(6,16,5)

        #fully connected
        self.fc1 = nn.Linear(16*5*5, 120)

        #fc2
        self.fc2 = nn.Linear(120,84)

        #fc3: reduce our output to a tensor of len=10: 10-class classification
        self.fc3 = nn.Linear(84,10)

net = Net()
print(net)

learning_rate = 0.001
momentum = 0.9

#defining our optimizer
optimizer = optim.SGD(net.parameters(), lr = learning_rate, momentum=momentum)

#print model's state_dict
print("Model's state_dict:")
