import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import nni
from models import *

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
trainset = torchvision.datasets.CIFAR10(root='./data', transform=transform_train, train=True)
testset = torchvision.datasets.CIFAR10(root='./data', transform=transform_test, train=False)
trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


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


model = None
optimizer = None


def train(args):
    global model, trainloader, optimizer
    criterion = nn.CrossEntropyLoss()
    if args['model'] == 'vgg':
        model = VGG('VGG19')
    elif args['model'] == 'resnet18':
        model = ResNet18()
    elif args['model'] == 'googlenet':
        model = GoogLeNet()
    elif args['model'] == 'densenet121':
        model = DenseNet121()
    elif args['model'] == 'mobilenet':
        model = MobileNet()
    elif args['model'] == 'dpn92':
        model = DPN92()
    elif args['model'] == 'shufflenetg2':
        model = ShuffleNetG2()
    elif args['model'] == 'senet18':
        model = SENet18()
    elif args['model'] == 'cnn':
        model = NeuralNet()

    if args['optimizer'] == 'Adadelta':
        optimizer = optim.Adadelta(model.parameters(), lr=args['lr'])
    elif args['optimizer'] == 'Adagrad':
        optimizer = optim.Adagrad(model.parameters(), lr=args['lr'])
    elif args['optimizer'] == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args['lr'])
    elif args['optimizer'] == 'Adamax':
        optimizer = optim.Adamax(model.parameters(), lr=args['lr'])
    else:
        optimizer = optim.SGD(model.parameters(), lr=args['lr'], momentum=0.9, weight_decay=5e-4)
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

            if i % 2560 == 2559:
                print('[%5d, %5d] loss = %.5f' % (epoch + 1, i + 1, running_loss / 2560))
                running_loss = 0.0

    torch.save(model.state_dict(), './cifar_net({},{},{}).pth'.format(args['lr'], args['optimizer'], args['model']))
    print('Training Finished')


def test():
    global model, testloader, classes
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
    best_accuracy = 0.0
    args = nni.get_next_parameter()
    for t in range(2):
        train(args)
        accuracy = test()
        if accuracy > best_accuracy:
            best_accuracy = accuracy
        print("accuracy: %.4f" % accuracy)
        nni.report_intermediate_result(accuracy)
    nni.report_final_result(best_accuracy)
    print("best accuracy: %.4f" % best_accuracy)


if __name__ == '__main__':
    main()

