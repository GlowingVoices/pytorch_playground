from numpy import r_
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

ds = datasets.FashionMNIST(
    root="data",
    train=True,
    transform=ToTensor(), #converting our data into tensors.
    target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0,torch.tensor(y), value=1)) #modifying the labels
)

#Creating a transformation for the label.
#The transformation will take the y label (0-9) and convert it to a one-hot encoding:

"""

2
tensor([0., 0., 1., 0., 0., 0., 0., 0., 0., 0.])
8
tensor([0., 0., 0., 0., 0., 0., 0., 0., 1., 0.])
5
tensor([0., 0., 0., 0., 0., 1., 0., 0., 0., 0.])

"""

#so this creates a zeros tensor of size 10 and adds 1 to the index of the number (which means that the value 3 would boost the 4th index since tensor indexing starts from 0)
#target_transform = Lambda(lambda y:torch.zeros(10,dtype=torch.float).scatter_(dim=0,index=torch.tensor(y),value=1))

"""
def transform(y):
    y = int(y)
    return torch.zeros(10, dtype=torch.float).scatter_(0,torch.tensor(y), value=1)

while(True):
    r = input()
    print(transform(r))
"""
