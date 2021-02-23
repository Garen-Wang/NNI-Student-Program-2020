import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# import nni
# from models import *
from vgg import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
trainset = torchvision.datasets.CIFAR10(root='./data', transform=transform_train, train=True, download=True)
testset = torchvision.datasets.CIFAR10(root='./data', transform=transform_test, train=False, download=True)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def train(args):
    global model, trainloader, optimizer
    criterion = nn.CrossEntropyLoss()
    trainloader = DataLoader(trainset, batch_size=args['batch_size'], shuffle=True, num_workers=2)
    model = VGG('VGG16')
    model = model.to(device)
    if device == 'cuda':
        model = torch.nn.DataParallel(model)
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
    for epoch in range(200):
        model.train()
        print('epoch %d running...' % epoch)
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if i % 2560 == 2559:
                print('[%5d, %5d] loss = %.5f' % (epoch + 1, i + 1, running_loss / 2560))
                running_loss = 0.0
        if (epoch + 1) % 10 == 0:
            # torch.save(model.state_dict(), './cifar_net({},{},{}).pth'.format(args['lr'], args['optimizer'], args['model']))
            torch.save(model.state_dict(), 'cifar_net.pth')
            accuracy = test()
            print('accuracy: %.4f' % accuracy)
    print('Training Finished')


def test():
    global model, testloader, classes
    testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
    class_correct = list(0.0 for i in range(10))
    class_total = list(0.0 for i in range(10))
    model.eval()
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
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
    best_accuracy = 0.0
    # args = nni.get_next_parameter()
    args = {
        "batch_size": 100,
        "lr": 0.01,
        "optimizer": "Adamax",
        "model": "vgg"
    }
    for t in range(1):
        train(args)
        accuracy = test()
        if accuracy > best_accuracy:
            best_accuracy = accuracy
        # nni.report_intermediate_result(accuracy)
    # nni.report_final_result(best_accuracy)


if __name__ == '__main__':
    main()
