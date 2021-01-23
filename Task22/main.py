import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import nni
from nni.algorithms.nas.pytorch import darts
from nni.nas.pytorch.callbacks import ArchitectureCheckpoint, LRSchedulerCallback
from nni.nas.pytorch import mutables

from mobilenet import *


class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        # self.conv1 = nn.Conv2d(3, 6, 5)
        # self.pool = nn.MaxPool2d(2, 2)
        # self.conv2 = nn.Conv2d(6, 16, 5)
        # self.func1 = nn.Linear(16 * 5 * 5, 120)
        # self.func2 = nn.Linear(120, 84)
        # self.func3 = nn.Linear(84, 10)
        self.conv1 = mutables.LayerChoice([
            nn.Conv2d(3, 8, 3, 1, padding=1),
            nn.Conv2d(3, 8, 5, 1, padding=2)
        ], key='conv1')
        self.mid_conv = mutables.LayerChoice([
            nn.Conv2d(8, 8, 3, 1, padding=1),
            nn.Conv2d(8, 8, 5, 1, padding=2)
        ], key='mid_conv')
        self.conv2 = nn.Conv2d(8, 16, 5, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.func1 = nn.Linear(16 * 5 * 5, 120)
        self.func2 = nn.Linear(120, 84)
        self.func3 = nn.Linear(84, 10)
        self.input_switch = mutables.InputChoice(n_candidates=2, n_chosen=1, key='skip')

    def forward(self, x):
        # x = self.pool(F.relu(self.conv1(x)))
        # x = self.pool(F.relu(self.conv2(x)))
        # x = x.view(-1, 16 * 5 * 5)
        # x = F.relu(self.func1(x))
        # x = F.relu(self.func2(x))
        # x = self.func3(x)
        x = self.pool(F.relu(self.conv1(x)))
        old_x = x
        zero_x = torch.zeros_like(old_x)
        x = self.pool(F.relu(self.mid_conv(x)))
        skip_x = self.input_switch([zero_x, old_x])
        x = x + skip_x
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.func1(x))
        x = F.relu(self.func2(x))
        x = self.func3(x)
        return x


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
trainset = torchvision.datasets.CIFAR10(root='./data', transform=transform_train)
testset = torchvision.datasets.CIFAR10(root='./data', transform=transform_test)
trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

lr = 0.001
model = NeuralNet()
optimizer = optim.Adamax(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()


def train(args):
    global model, trainloader, optimizer, criterion

    for epoch in range(10):
        print('epoch %d running...' % epoch)
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if i % 2000 == 1999:
                print('[%5d, %5d] loss = %.5f' % (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
        # accuracy = test()
        # print("accuracy: %.4f" % accuracy)

    torch.save(model.state_dict(), './cifar_net_result.pth')
    print('Training Finished')


def test():
    global model, testloader, classes
    # model.load_state_dict(torch.load(path))
    class_correct = list(0.0 for i in range(10))
    class_total = list(0.0 for i in range(10))
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = model(images)
            _, predicated = torch.max(outputs, 1)
            c = (predicated == labels).squeeze()
            for i in range(10):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    correct = 0
    total = 0
    for i in range(10):
        # print('Accuracy of %s: %.2f%%' % (classes[i], 100.0 * class_correct[i] / class_total[i]))
        correct += class_correct[i]
        total += class_total[i]
    accuracy = 100.0 * correct / total
    print('Testing Finished')
    return accuracy


def main():
   trainer = darts.DartsTrainer(
        model=model,
        loss=criterion,
        # metrics=labmda output, target: accuracy(output, target, topk=(1, )),
        optimizer=optimizer,
        num_epochs=10, # (not certain
        dataset_train=trainset,
        dataset_valid=testset,
        batch_size=128,
        log_frequency=60,
        unrolled=False
    )
    trainer.train()
    trainer.export(file='./final_architecture.json')


if __name__ == '__main__':
    main()

