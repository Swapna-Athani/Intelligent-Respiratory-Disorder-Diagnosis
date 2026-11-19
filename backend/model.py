import torch
from torch import nn
class ScreeningCNN(nn.Module):
    def __init__(self,channels,classes):
        super().__init__()
        self.features=nn.Sequential(nn.Conv2d(channels,16,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(16,32,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(32,64,3,padding=1),nn.ReLU(),nn.AdaptiveAvgPool2d(1))
        self.classifier=nn.Sequential(nn.Flatten(),nn.Dropout(.25),nn.Linear(64,classes))
    def forward(self,x):return self.classifier(self.features(x))
