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
        #I have 3 input channels of size 5x5: I'm going to have 3 5x5 vectors within my model
        #I have 6 output channels: the output channels are the number of feature maps I'm creating. That means I have SIX filters.

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
#print(net)

learning_rate = 0.001
momentum = 0.9

#defining our optimizer
optimizer = optim.SGD(net.parameters(), lr = learning_rate, momentum=momentum)

print("Model's state_dict:")
for param_tensor in net.state_dict():
    #param_tensor is the string name for whatever we're accessing
    #the state_dict[param_tensor] gets us the tensor (torch.float32).

    tensor = net.state_dict()[param_tensor]
    print(f"The param_tensor is {param_tensor}, the datatype is {tensor.dtype} and the tensor shape is {tensor.shape}")
    #torch.Size and shape are the same thing; shape was added to make it more similar to numpy

print()
print("Optimizer's state_dict:")
for var_name in optimizer.state_dict():
    print(var_name, "\t", optimizer.state_dict()[var_name])

print("w", optimizer.state_dict()["param_groups"][0]["params"][0].dtype)
    #I have no idea what's going on with params here
