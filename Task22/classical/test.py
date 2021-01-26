import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

import nni
from nni.nas.pytorch import mutables
from nni.algorithms.nas.pytorch.classic_nas import get_and_apply_next_architecture

import argparse
import logging
from collections import OrderedDict

logger = logging.getLogger('CIFAR10-Classical-NAS')


class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        # Conv2d: in_channels, out_channels, kernel_size, stride

        self.conv1 = mutables.LayerChoice(OrderedDict([
            ("conv3*3", nn.Conv2d(3, 8, 3, 1)),
            ("conv5*5", nn.Conv2d(3, 8, 5, 1))
        ]), key='conv1')

        # self.conv1 = nn.Conv2d(3, 8, 3, 1)
        self.mid_conv = mutables.LayerChoice([
            nn.Conv2d(8, 8, 3, 1, padding=1),
            nn.Conv2d(8, 8, 5, 1, padding=2)
        ], key='mid_conv')
        # self.mid_conv = nn.Conv2d(8, 8, 5, 1)
        self.conv2 = nn.Conv2d(8, 16, 5, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.func1 = nn.Linear(16 * 5 * 5, 120)
        self.func2 = nn.Linear(120, 84)
        self.func3 = nn.Linear(84, 10)
        self.input_switch = mutables.InputChoice(n_candidates=2, n_chosen=1, key="skip_conv")

    def forward(self, x):
        # x = self.pool(F.relu(self.conv1(x)))
        # print("x.shape: {}".format(x.shape))
        # x = self.conv1(x)
        # print("x.shape: {}".format(x.shape))
        # x = F.relu(x)
        # print("x.shape: {}".format(x.shape))
        # x = self.pool(x)
        # print("1st stage, x.shape: {}".format(x.shape))
        # x = self.conv2(x)
        # print("x.shape: {}".format(x.shape))
        # x = F.relu(x)
        # print("x.shape: {}".format(x.shape))
        # x = self.pool(x)
        # print("2nd stage, x.shape: {}".format(x.shape))


        # x = self.pool(F.relu(self.conv2(x)))

        # old_x = x
        # x = F.relu(self.mid_conv(x))
        # x = self.pool(F.relu(self.conv2(x)))
        # zero_x = torch.zeros_like(old_x)
        # skip_x = self.input_switch([zero_x, old_x])
        # x = torch.add(x, skip_x)
        # x = F.relu(self.mid_conv(x))
        # x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv1(x)))
        old_x = x
        zero_x = torch.zeros_like(old_x)
        skip_x = self.input_switch([zero_x, old_x])
        x = F.relu(self.mid_conv(x))
        x += skip_x
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.func1(x))
        x = F.relu(self.func2(x))
        x = self.func3(x)
        return x


# class Net(nn.Module):
#     def __init__(self, hidden_size):
#         super(Net, self).__init__()
#         # two options of conv1
#         self.conv1 = mutables.LayerChoice(OrderedDict([
#             ("conv5x5", nn.Conv2d(1, 20, 5, 1)),
#             ("conv3x3", nn.Conv2d(1, 20, 3, 1))
#         ]), key='first_conv')
#         # two options of mid_conv
#         self.mid_conv = mutables.LayerChoice([
#             nn.Conv2d(20, 20, 3, 1, padding=1),
#             nn.Conv2d(20, 20, 5, 1, padding=2)
#         ], key='mid_conv')
#         self.conv2 = nn.Conv2d(20, 50, 5, 1)
#         self.fc1 = nn.Linear(4*4*50, hidden_size)
#         self.fc2 = nn.Linear(hidden_size, 10)
#         # skip connection over mid_conv
#         self.input_switch = mutables.InputChoice(n_candidates=2,
#                                         n_chosen=1,
#                                         key='skip')
# 
#     def forward(self, x):
#         x = F.relu(self.conv1(x)) # 28 -> 24 or 28 -> 26
#         x = F.max_pool2d(x, 2, 2) # 24 -> 12 or 26 -> 13
#         old_x = x
#         x = F.relu(self.mid_conv(x)) # 12 or 13
#         zero_x = torch.zeros_like(old_x)
#         skip_x = self.input_switch([zero_x, old_x])
#         x = torch.add(x, skip_x) # 12 or 13
#         x = F.relu(self.conv2(x)) # 8 or 9
#         x = F.max_pool2d(x, 2, 2)
#         x = x.view(-1, 4*4*50)
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         return F.log_softmax(x, dim=1)


def train(model, trainloader, criterion, optimizer):
    print('training...')
    model.train()
    training_loss = 0.0
    for idx, data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        training_loss += loss.item()
        if (idx + 1) % 2560 == 0:
            logger.info('epoch: %d, loss: %.4f' % (idx + 1, training_loss / 2560.0))
            training_loss = 0.0
    print('training done')


def test(model, testloader):
    print('testing...')
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for idx, data in enumerate(testloader):
            inputs, labels = data
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
    accuracy = 100.0 * correct / total
    print('testing done, accuracy: %.4f' % accuracy)
    return accuracy


def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=128, metavar='BZ')
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--lr", type=float, default=0.01, metavar='LR')
    parser.add_argument("--epochs", type=int, default=20, metavar='E')
    parser.add_argument("--momentum", type=float, default=0.9, metavar='M')
    parser.add_argument("--log_interval", type=int, default=2560, metavar='L')
    args, _ = parser.parse_known_args()
    return args


def main(args):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    trainset = datasets.CIFAR10(root=args['data_dir'], transform=transform_train, train=True)
    testset = datasets.CIFAR10(root=args['data_dir'], transform=transform_test, train=False)
    trainloader = DataLoader(trainset, batch_size=args['batch_size'], shuffle=True, num_workers=2)
    testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
    model = NeuralNet()
    get_and_apply_next_architecture(model) # classical NAS
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args['lr'], momentum=args['momentum'], weight_decay=5e-4)
    for epoch in range(1, args['epochs'] + 1):
        train(model, trainloader, criterion, optimizer)
        test_accuracy = test(model, testloader)
        if epoch < args['epochs']:
            nni.report_intermediate_result(test_accuracy)
        else:
            nni.report_final_result(test_accuracy)
    print('done.')


if __name__ == '__main__':
    try:
        params = vars(get_params())
        main(params)
    except Exception as exception:
        logger.exception(exception)
        raise
