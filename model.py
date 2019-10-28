import torch
import torch.nn as nn
import torch.nn.functional as F

nclasses = 43 

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 128, kernel_size=3)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3)
        self.conv3 = nn.Conv2d(128, 64, kernel_size=3)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3)
        self.conv_bn = nn.BatchNorm2d(64)
        self.conv_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, 50)
        self.finallayer = nn.Linear(50, nclasses)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv_drop(self.conv1(x)), 2))
        x = F.relu(F.max_pool2d(self.conv_drop(self.conv2(x)), 2))
        x = F.relu(self.conv_bn(self.conv3(x)))
        x = F.relu(self.conv_bn(self.conv4(x)))
        x = x.view(-1, 256)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training, p=0.2)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, training=self.training, p=0.2)
        x = self.finallayer(x)
        return F.log_softmax(x)