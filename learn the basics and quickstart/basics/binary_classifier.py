import torch
import torch.nn

trainet =
trainLoader =

testset =
testloader =

classes = ('ROI', 'nROI')

class BN(nn.Module:
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d()
        self.mp1 = nn.MaxPool2d()
        self.conv2 = nn.Conv2d()
        self.mp2 = nn.MaxPool2d()
        self.fl = nn.Flatten()
        self.d1 = nn.Linear()
        self.d2 = nn.Linear()

    def forward(self, x):
        x = self.mp1(self.conv1(x))
        x = self.mp2(self.conv2(x))
        x = self.f1(x)
        x = F.relu(self.d1(x))
        x = F.relu(self.d2(x))
        x = F.
