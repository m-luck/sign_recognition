import torch
import torch.nn as nn
import torch.nn.functional as F

nclasses = 43 # GTSRB as 43 classes

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 108, 3)
        self.conv2 = nn.Conv2d(108, 200, 3)
        self.fc1 = nn.Linear(15 * 15 * 108 + 6 * 6 * 200, 100)
        self.fc2 = nn.Linear(100, nclasses)

    def forward(self, x):
        x1 = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x2 = F.max_pool2d(F.relu(self.conv2(x1)), 2)
        # print(x1.size())
        # print(x2.size())
        x1 = x1.view(-1, 108*15*15)
        x2 = x2.view(-1, 200*6*6)
        out = [x1, x2]
        out = torch.cat(out, 1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        return F.log_softmax(out, dim=1)
