import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from models import *

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
testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def test():
    global model
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
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

if __name__ == '__main__':
    try:
        name = 'googlenet'
        if name == 'vgg':
            model = VGG('VGG19')
        elif name == 'resnet18':
            model = ResNet18()
        elif name == 'googlenet':
            model = GoogLeNet()
        elif name == 'densenet121':
            model = DenseNet121()
        elif name == 'mobilenet':
            model = MobileNet()
        elif name == 'dpn92':
            model = DPN92()
        elif name == 'shufflenetg2':
            model = ShuffleNetG2()
        elif name == 'senet18':
            model = SENet18()
        model.load_state_dict(torch.load('cifar_net(0.001,Adam,googlenet).pth'))
        accuracy = test()
        print('%.4f' % accuracy)
    except Exception as e:
        raise e
