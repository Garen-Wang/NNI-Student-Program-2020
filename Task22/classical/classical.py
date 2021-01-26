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

logger = logging.getLogger('CIFAR10-Classical-NAS')


class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.conv1 = mutables.LayerChoice([
            nn.Conv2d(3, 8, 3, 1, padding=1),
            nn.Conv2d(3, 8, 5, 1, padding=2)
        ], key="conv1")
        self.mid_conv = mutables.LayerChoice([
            nn.Conv2d(8, 8, 3, 1, padding=1),
            nn.Conv2d(8, 8, 5, 1, padding=2)
        ], key="mid_conv")
        self.conv2 = nn.Conv2d(8, 16, 5, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.func1 = nn.Linear(16 * 5 * 5, 120)
        self.func2 = nn.Linear(120, 84)
        self.func3 = nn.Linear(84, 10)
        # self.input_switch = mutables.InputChoice(n_candidates=1)
        self.input_switch = mutables.InputChoice(n_candidates=2, n_chosen=1, key="skip_conv")

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        # skip_x = self.input_switch([x])
        # x = F.relu(self.mid_conv(x))
        # if skip_x is not None:
        #     x += skip_x
        old_x = x
        x = self.mid_conv(x)
        zero_x = torch.zeros_like(old_x)
        skip_x = self.input_switch([old_x, zero_x])
        x += skip_x
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.func1(x))
        x = F.relu(self.func2(x))
        x = self.func3(x)
        return x


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
    return accuracy


def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=128, metavar='BZ')
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--lr", type=float, default=0.01, metavar='LR')
    parser.add_argument("--epochs", type=int, default=10, metavar='E')
    parser.add_argument("--momentum", type=float, default=0.9, metavar='M')
    parser.add_argument("--log_interval", type=int, default=600, metavar='L')
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
    trainset = datasets.CIFAR10(root=args['data_dir'], transform=transform_train)
    testset = datasets.CIFAR10(root=args['data_dir'], transform=transform_test)
    trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
    testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
    model = NeuralNet()
    get_and_apply_next_architecture(model) # classical NAS
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args['lr'], momentum=args['momentum'], weight_decay=5e-4)
    for epoch in range(args['epochs']):
        train(model, trainloader, criterion, optimizer)
        test_accuracy = test(model, testloader)

if __name__ == '__main__':
    try:
        params = vars(get_params())
        main(params)
    except Exception as exception:
        logger.exception(exception)
        raise
