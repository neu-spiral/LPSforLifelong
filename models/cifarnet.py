from __future__ import print_function
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CifarNet(nn.Module):
    def __init__(self,input_size, num_classes, shrink=1):
        super(CifarNet, self).__init__()
        #self.inNorm = nn.BatchNorm2d(3)
        self.shrink=shrink
        self.conv1 = nn.Conv2d(3, int(input_size*shrink), 3, 1, bias=False)
        #self.norm1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(int(input_size*shrink), int(32*shrink), 3, 1, bias=False)
        self.conv3 = nn.Conv2d(int(32*shrink), int(64*shrink), 3, 1, bias=False)
        self.conv4 = nn.Conv2d(int(64*shrink), int(64*shrink), 3, 1, bias=False)
        self.fc1 = nn.Linear(5*5*int(64*shrink), 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.drp1 = nn.Dropout(p=.25)
        self.drp2 = nn.Dropout(p=.5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.drp1(F.max_pool2d(x, 2, 2))
        
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.drp1(F.max_pool2d(x, 2, 2))
        
        x = x.view(-1, 5*5*int(64*self.shrink))
        x = self.drp2(F.relu(self.fc1(x)))
        x = self.fc2(x)
        #x = self.fc3(x)
        return F.log_softmax(x, dim=1)