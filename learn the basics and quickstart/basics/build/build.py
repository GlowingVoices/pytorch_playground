import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# initializing the device
device = (
    "cuda" #gpu
    if torch.cuda.is_available()
    else "mps" #on macs with the metal framework
    if torch.backends.mps.is_available()
    else "cpu" #cpu
)

# print the device being used
print(f"Using {device} device")

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512,10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

my_nn = NeuralNetwork().to(device)
print(my_nn)

"""
How the data flows:
    We have some input vector, and some labels vector.
    We start by flattening our input vector (originally of 28x28 pixels) into 784 points
    Our first Linear layer turns those 784 points into 512 points.
    We run this through a ReLU (I'm reasonable sure this is equivalen to a Dense Layer with activation="relu" in tensorflow)
    We do this again (in = 512, out = 512)
    ReLU
    Again (in = 512, out = 10)
    This last bit seems a bit off, so I wouldn't be suprised if we need a softmax somewhere. It's the output layer, after all.
"""

#... and the model construction itself:

# a random image (generated of 784 random points)
X = torch.rand(1,28,28,device=device)

# We pass it to the model
logits = my_nn(X)

# We get the predictions and - what do you know - run them through a softmax
predicted_probabilities = nn.Softmax(dim=1)(logits)

# our label_predicttion!
y_pred = predicted_probabilities.argmax(1)

#... and, output!
# this is going to be completely random right now: the model has neither training nor weights - if we can call them that - of any great use.
print(f"Predicted class: {y_pred}")
print(f"Predicted probabilities: {predicted_probabilities}")

"""
Predicted class: tensor([0], device='mps:0')
Predicted probabilities: tensor([[0.1093, 0.1043, 0.0986, 0.1033, 0.0981, 0.0981, 0.0884, 0.1075, 0.0978,
predicted probabilities returns the probabilities of being part of any given class
"""

input_image = torch.rand(3,28,28)
print(input_image.size())

print(f"Model structure: {my_nn} \n\n")

for name, param in my_nn.named_parameters():
    print(f"Layer: {name} | Size {param.size()} | Values: {param[:2]} \n")
