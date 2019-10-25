import torch
import torch.nn as nn
import torch.nn.functional as F

nclasses = 43 # GTSRB as 43 classes

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=3)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=3)
        self.conv3 = nn.Conv2d(20, 20, kernel_size=3)
        self.conv_drop = nn.Dropout2d(p=0.1)
        self.fc1 = nn.Linear(80, 60)
        self.fc2 = nn.Linear(60, 60)
        self.fc3 = nn.Linear(60, 55)
        self.fc4 = nn.Linear(55, 55)
        self.fc5 = nn.Linear(55, 50)
        self.finallayer = nn.Linear(50, nclasses)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv_drop(self.conv1(x)), 2))
        # print(x.size())
        x = F.relu(F.max_pool2d(self.conv_drop(self.conv2(x)), 2))
        x = F.relu(F.max_pool2d(self.conv_drop(self.conv3(x)), 2))
        # print(x.size())
        x = x.view(-1, 80)
        x = F.relu(self.fc1(x))
        resx = x.clone()
        x = F.dropout(x, p=0.4, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x += resx
        x = F.relu(self.fc3(x))
        resx = x.clone()
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.fc4(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x += resx
        x = F.relu(self.fc5(x))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.finallayer(x)
        return F.log_softmax(x)
