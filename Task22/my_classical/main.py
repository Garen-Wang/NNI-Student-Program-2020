import nni
from nni.nas.pytorch.mutables import LayerChoice, InputChoice
from nni.algorithms.nas.pytorch.classic_nas import get_and_apply_next_architecture
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def getArguments():
    parser = argparse.ArgumentParser('classicalNAS')
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--batch_size', default=100, type=int)
    args = parser.parse_args()
    return args


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(Block, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
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
    return layers


class MobileNet(nn.Module):
    def __init__(self):
        super(MobileNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = _make_mobilenet_layers(32)
        self.pool = nn.AvgPool2d(2)
        self.linear = nn.Linear(1024, 10)
        self.skipconnect1 = InputChoice(n_candidates=2, n_chosen=1, key='skip1')
        self.skipconnect2 = InputChoice(n_candidates=2, n_chosen=1, key='skip2')
        self.skipconnect3 = InputChoice(n_candidates=2, n_chosen=1, key='skip3')
        self.skipconnect4 = InputChoice(n_candidates=2, n_chosen=1, key='skip4')
        self.skipconnect5 = InputChoice(n_candidates=2, n_chosen=1, key='skip5')
        self.skipconnect6 = InputChoice(n_candidates=2, n_chosen=1, key='skip6')
        self.skipconnect7 = InputChoice(n_candidates=2, n_chosen=1, key='skip7')
        self.skipconnect8 = InputChoice(n_candidates=2, n_chosen=1, key='skip8')

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        i = 1
        for block in self.layers:
            old_x = x
            x = block(x)
            if block.in_channels == block.out_channels:
                zero_x = torch.zeros_like(old_x)
                skipconnect = eval('self.skipconnect{}'.format(i))
                skip_x = skipconnect([zero_x, old_x])
                x = torch.add(x, skip_x)
                i += 1
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
trainset = datasets.CIFAR10(root='../data', transform=transform_train, train=True, download=True)
testset = datasets.CIFAR10(root='../data', transform=transform_test, train=False, download=True)
trainloader = DataLoader(trainset, batch_size=100, shuffle=True, num_workers=2)
testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), 0.025, momentum=0.9, weight_decay=3.0E-4)


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
        print('accuracy: %.4f%%' % accuracy)
        return accuracy


if __name__ == '__main__':
    args = getArguments()
    best_accuracy = 0.0
    get_and_apply_next_architecture(model)
    for epoch in range(args.epochs):
        train(epoch)
        accuracy = test()
        if accuracy > best_accuracy:
            best_accuracy = accuracy
        if (epoch + 1) % 10 == 0:
            nni.report_intermediate_result(accuracy)
    nni.report_final_result(best_accuracy)

