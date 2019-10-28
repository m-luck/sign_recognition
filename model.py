import torch
import torch.nn as nn
import torch.nn.functional as F

nclasses = 43 # GTSRB as 43 classes

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1_1 = nn.Conv2d(1, 64, 5)
        self.conv2_1 = nn.Conv2d(64, 64, 3)
        self.conv2_2 = nn.Conv2d(64, 64, 3)
        self.conv2_3 = nn.Conv2d(64, 64, 3)
        self.fc1 = nn.Linear(64*7*7 + 64*5*5, 64)
        self.fc2 = nn.Linear(64, nclasses)

        print(""" \
            self.conv1_1 = nn.Conv2d(1, 64, 5)
            self.conv2_1 = nn.Conv2d(64, 64, 3)
            self.conv2_2 = nn.Conv2d(64, 64, 3)
            self.conv2_3 = nn.Conv2d(64, 64, 3)
            self.fc1 = nn.Linear(64*5*5 + 64*1*1, 64)
            self.fc2 = nn.Linear(64, nclasses)
              """)

    def forward(self, x):
        x1 = F.max_pool2d(F.relu(self.conv1_1(x)), 4)
        x2 = F.max_pool2d(F.relu(self.conv2_1(x1)), 3)
        x2 = F.relu(self.conv2_2(x1))
        x2 = F.relu(self.conv2_3(x1))
        # print(x1.size())
        # print(x2.size())
        x1 = x1.view(-1, 64*7*7)
        x2 = x2.view(-1, 64*5*5)
        out = [x1, x2]
        out = torch.cat(out, 1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        return F.log_softmax(out, dim=1)
