import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import nni

import numpy as np
import matplotlib.pyplot as plt


class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.func1 = nn.Linear(16 * 5 * 5, 120)
        self.func2 = nn.Linear(120, 84)
        self.func3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.func1(x))
        x = F.relu(self.func2(x))
        x = self.func3(x)
        return x


def train(trainloader, path):
    neuralnet = NeuralNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(neuralnet.parameters(), lr=0.001, momentum=0.9)
    neuralnet.train()
    for epoch in range(10):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = neuralnet(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if i % 2000 == 1999:
                print('[%5d, %5d] loss = %.5f' % (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    torch.save(neuralnet.state_dict(), path)
    print('Training Finished')


def test(testloader, path, classes):
    neuralnet = NeuralNet()
    neuralnet.load_state_dict(torch.load(path))
    class_correct = list(0.0 for i in range(10))
    class_total = list(0.0 for i in range(10))
    neuralnet.eval()
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = neuralnet(images)
            _, predicated = torch.max(outputs, 1)
            c = (predicated == labels).squeeze()
            for i in range(4):
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


def showimg(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def showImages(trainloader):
    dataiter = iter(trainloader)
    images, labels = dataiter.next()
    showimg(torchvision.utils.make_grid(images))


def main():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', transform=transform)
    trainloader = DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)
    testloader = DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    path = './cifar_net.pth'
    # showImages(trainloader)
    train(trainloader, path)
    accuracy = test(testloader, path, classes)


if __name__ == '__main__':
    main()
