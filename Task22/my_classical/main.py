import nni
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, datasets
from torch.utils.data import DataLoader


def getArguments():
    parser = argparse.ArgumentParser('classicalNAS')
    parser.add_argument('--epochs', default=100, type=int)
    args = parser.parse_args()
    return args


args = getArguments()
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, stride, 1, bias=False, groups=in_channels)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x


def _make_mobilenet_layers(in_channels):
    cfg = [64, (128, 2), 128, (256, 2), 256, (512, 2), 512, 512, 512, 512, 512, (1024, 2), 1024]
    layers = nn.ModuleList()
    for x in cfg:
        out_channels = x if isinstance(x, int) else x[0]
        stride = 1 if isinstance(x, int) else x[1]
        layers.append(Block(in_channels, out_channels, stride))
        in_channels = out_channels
    return nn.Sequential(*layers)


class MobileNet(nn.Module):
    def __init__(self):
        super(MobileNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = _make_mobilenet_layers(32)
        self.pool = nn.AvgPool2d(2)
        self.linear = nn.Linear(1024, 10)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layers(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x


model = MobileNet()
model = model.to(device)

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
trainset = datasets.CIFAR10(root='../data', transform=transform_train, train=True)
testset = datasets.CIFAR10(root='../data', transform=transform_test, train=False)
trainloader = DataLoader(trainset, batch_size=100, shuffle=True, num_workers=2)
testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), 0.025, momentum=0.9, weight_decay=3.0E-4)
# lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args['epochs'], eta_min=0.001)


def train(epoch_id):
    global trainloader, model, optimizer
    model.train()
    print('epoch %d is running...' % epoch_id)
    running_loss = 0.0
    for idx, (inputs, labels) in enumerate(trainloader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        running_loss += loss
        loss.backward()
        optimizer.step()
        if (idx + 1) % 100 == 0:
            print('running loss: %.4f' % (running_loss / 100))
            running_loss = 0.0
    print('training done')


def test():
    global testloader, model
    model.eval()
    class_correct = list(0.0 for i in range(10))
    class_total = list(0.0 for i in range(10))
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicated = torch.max(outputs, 1)
            c = (predicated == labels).squeeze()
            for i in range(100):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
        for i in range(10):
            # print('Accuracy of %s: %.2f%%' % (classes[i], 100.0 * class_correct[i] / class_total[i]))
            correct += class_correct[i]
            total += class_total[i]
        accuracy = 100.0 * correct / total
        print('accuracy: %.4f%%' % (accuracy))

if __name__ == '__main__':
    for epoch in range(args.epochs):
        train(epoch)
        test()
