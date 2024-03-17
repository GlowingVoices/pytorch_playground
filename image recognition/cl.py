import os
import pandas as pd
import torch

from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from torchvision.io import read_image

from torch.utils import nn
import torch.nn.functional as F
import torch.optim as optim

def apply_transformation(data):
    transformed = torch.ToTensor(data)
    print(transformed)

class ImageReader(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = apply_transformation(transform)
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self,index):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[index,0])
        image = read_image(img_path)
        label = self.img_labels.iloc[index,1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


train_labels = "train_labels.csv"
train_dir = "images/train"

test_labels = "test_labels.csv"
test_dir = "images/test"

train_ireader = ImageReader(train_labels,train_dir)
test_ireader = ImageReader(test_labels, test_dir)

#macs don't have cuda
device = (
    "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

class NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.filter_block_1 = nn.Sequential (
            nn.Conv2d()
            nn.ReLU()
            nn.MaxPool2d()
        )

        self.filter_block_2 = nn.Sequential (
            nn.Conv2d()
            nn.ReLU()
            nn.MaxPool2d()
        )


        self.flatten = nn.Flatten()

        self.dense = nn.Sequential(
            nn.Linear()
            nn.ReLU()
            nn.Linear()
            nn.ReLU()
        )


    def forward(self, x):
        logits = self.pool(nn.ReLU(self.conv1(x)))
        logits = self.filter_block_1(logits)
        logits = self.flatten(self.filter_block_2(logits))
        logits = self.dense(logits)
        return logits


net = NN()

loss_fn = F.binary_cross_entropy()
batch_size = 64
lr= 0.0001


optimizer = optim.Adam(net.parameters(),lr)
#you'll need to make sure the multithreading is working: it might not be a problem here, but training time will definitely increase when I'm doing unet, vision transformer and yolo stuff

def train_model():


def test_model():
