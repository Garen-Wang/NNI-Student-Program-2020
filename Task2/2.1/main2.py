import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
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


def img_show(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def train(train_loader, Path):
    net = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(2):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 2000 == 1999:
                print('!!!! [%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    torch.save(net.state_dict(), Path)


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)  # 事实上就输出了 cuda:0

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_set = torchvision.datasets.CIFAR10(root="./data", transform=transform)  # 训练集
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=10, shuffle=True, num_workers=2)
    # 注意大小写

    test_set = torchvision.datasets.CIFAR10(root="./data", transform=transform)  # 测试集
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=10, shuffle=False, num_workers=2)
    classes = ('plane', 'car', 'bird', 'cat', "deer", "dog", "frog", 'horse', 'ship', 'truck')
    dataiter = iter(train_loader)
    images, labels = dataiter.next()  # 提取数据

    # img_show(torchvision.utils.make_grid(images)) #展示数据
    '''for i in labels:
        print(classes[i],end=' ')'''
    Path = './cifar_net.pth'
    train(train_loader, Path)
    print("---Train end---")
    net = Net()
    dataiter = iter(test_loader)
    images, labels = dataiter.next()
    '''for i in labels:
        print(classes[i],end=" ")'''
    net.load_state_dict(torch.load(Path))  # 导入模型
    outputs = net(images)
    _, predicted = torch.max(outputs, 1)
    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))
    print("---Test end---")
    pass


if __name__ == "__main__":
    main()
