"""HybridSN architecture, adapted from the existing local Pavia implementation.
Reference: Roy et al., DOI 10.1109/LGRS.2019.2918719.
"""
import torch
from torch import nn

class HybridSN(nn.Module):
    def __init__(self, n_classes=9):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv3d(1, 8, (7,3,3)), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv3d(8,16,(5,3,3)), nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv3d(16,32,(3,3,3)), nn.ReLU())
        self.conv4 = nn.Sequential(nn.Conv2d(96,64,3), nn.ReLU())
        self.fc1 = nn.Sequential(nn.Linear(64*17*17,256), nn.ReLU(), nn.Dropout(.4))
        self.fc2 = nn.Sequential(nn.Linear(256,128), nn.ReLU(), nn.Dropout(.4))
        self.fc3 = nn.Linear(128,n_classes)
    def forward(self,x):
        x=self.conv3(self.conv2(self.conv1(x.unsqueeze(1))))
        x=x.reshape(x.shape[0],-1,x.shape[3],x.shape[4])
        x=self.conv4(x).flatten(1)
        return self.fc3(self.fc2(self.fc1(x)))
