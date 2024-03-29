from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

# Training settings
parser = argparse.ArgumentParser(description='PyTorch GTSRB example')
parser.add_argument('--data', type=str, default='data', metavar='D',
                    help="folder where data is located. train_data.zip and test_data.zip need to be found in the folder")
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
print("lr",args.lr)
print("momentum",args.momentum)

torch.manual_seed(args.seed)

### Data Initialization and Loading
from data import initialize_data, spec_trans, spec_trans_end, randoTrans # data.py in the same folder
from data import data_transforms
initialize_data(args.data) # extracts the zip files, makes a validation set

train_loader = torch.utils.data.DataLoader(
    torch.utils.data.ConcatDataset([
        datasets.ImageFolder(args.data + "/train_images", transform=spec_trans(transforms.CenterCrop(32))),
        datasets.ImageFolder(args.data + "/train_images", transform=spec_trans(transforms.ColorJitter(brightness=0.5))),
        datasets.ImageFolder(args.data + "/train_images", transform=spec_trans(transforms.ColorJitter(contrast=0.7))),
        datasets.ImageFolder(args.data + "/train_images", transform=spec_trans(transforms.ColorJitter(hue=0.5))),
        datasets.ImageFolder(args.data + "/train_images", transform=spec_trans(transforms.RandomAffine(80))),
        datasets.ImageFolder(args.data + "/train_images", transform=spec_trans(transforms.RandomAffine(0, translate=((0.10,0.10))))),
        datasets.ImageFolder(args.data + "/train_images", transform=spec_trans(transforms.RandomAffine(0, scale=(1.0, 1.24), shear=13))),
        # datasets.ImageFolder(args.data + "/train_images", transform=spec_trans(transforms.RandomHorizontalFlip(p=1.0))),
        # datasets.ImageFolder(args.data + "/train_images", transform=spec_trans(transforms.RandomVerticalFlip(p=1.0))),
        datasets.ImageFolder(args.data + "/train_images", transform=spec_trans_end(transforms.RandomErasing(p=1.0, value='random'))),
        datasets.ImageFolder(args.data + "/train_images", transform=randoTrans),
        datasets.ImageFolder(args.data + "/train_images", transform=spec_trans(transforms.RandomPerspective()))
    ]),
    batch_size=args.batch_size, shuffle=True, num_workers=4)

# train_loader = torch.utils.data.DataLoader(
#     datasets.ImageFolder(args.data + '/train_images',
#                          transform=data_transforms),
#     batch_size=args.batch_size, shuffle=True, num_workers=1)
val_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(args.data + '/val_images',
                         transform=data_transforms),
    batch_size=args.batch_size, shuffle=False, num_workers=1)

### Neural Network and Optimizer
# We define neural net in model.py so that it can be reused by the evaluate.py script
from model import Net
model = Net()
# model = torch.hub.load('pytorch/vision', 'mobilenet_v2', pretrained=False)
above_thres = False
optimizer = optim.Adam(model.parameters(), lr=args.lr)

def train(epoch):
    model.train()
    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    # if above_thres:
    #     # optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.7)
    #     optimizer = optim.Adam(model.parameters(), lr=0.00025)
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        # loss = F.cross_entropy(output, target)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data.item()))
    print(target)

def validation():
    model.eval()
    validation_loss = 0
    correct = 0
    for data, target in val_loader:
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        validation_loss += F.nll_loss(output, target, size_average=False).data.item() # sum up batch loss
        # validation_loss += F.cross_entropy(output, target, size_average=False).data.item() # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    validation_loss /= len(val_loader.dataset)
    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        validation_loss, correct, len(val_loader.dataset),
        100. * correct / len(val_loader.dataset)))
    # if validation_loss > 90:
    #     above_thres = False
    # else:
    #     above_thres = False


for epoch in range(1, args.epochs + 1):
    train(epoch)
    validation()
    model_file = 'model_' + str(epoch) + '.pth'
    torch.save(model.state_dict(), model_file)
    print('\nSaved model to ' + model_file + '. You can run `python evaluate.py --model' + model_file + '` to generate the Kaggle formatted csv file')
