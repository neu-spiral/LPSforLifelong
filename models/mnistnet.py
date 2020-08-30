from __future__ import print_function
import torch.nn as nn
import torch.nn.functional as F


class MnistNet(nn.Module):
    def __init__(self, input_size, num_classes, shrink=1):
        super(MnistNet, self).__init__()
        #self.inNorm = nn.BatchNorm2d(3)
        self.fc1 = nn.Linear(input_size, int(2000*shrink))
        self.fc2 = nn.Linear(int(2000*shrink), 2000)
        self.fc3 = nn.Linear(2000, num_classes)
        self.drp1 = nn.Dropout(p=.5)

    def forward(self, x):
        x = self.drp1(F.relu(self.fc1(x)))
        x = self.drp1(F.relu(self.fc2(x)))
        
        
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)
# model = ConvNet()
# for name, W in model.named_parameters():
#     print(W.size())