# Task 2 进阶任务 HPO + NAS

<<<<<<< HEAD
- [x] [Task2.1实验报告](Task2.1/README.md)
- [x] [Task2.2实验报告](Task2.2/README.md)
=======
## Task 2.1

### CIFAR10简介

CIFAR10数据集共有60000张分辨率为32*32的彩色图像，分为十类，每类都有6000张图像。

50000张图像构成训练集，10000张图像构成测试集。

![](./Images/1.png)

### 实现流程

我们使用PyTorch编写卷积神经网络来解决这项图像分类任务。

大体流程如下：

1. 使用torchvision下载数据集，读取数据集
2. 定义解决该问题的卷积神经网络
3. 训练神经网络
4. 测试神经网络

### 实验配置

使用conda环境下的Python3.8，使用PyTorch框架运行程序。使用CPU进行训练。

### 代码分析

#### 神经网络的定义

我们利用了`torch.nn`模块定义了本任务的神经网络`NeuralNet`。

```python
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
        x = x.view(-1, 16 * 5 * 5) # -1 means uncertain number
        x = F.relu(self.func1(x))
        x = F.relu(self.func2(x))
        x = self.func3(x)
        return x
```

该模型与最经典的卷积神经网络LeNet非常相像，唯二的区别是输入的单通道变成了RGB三通道，激活函数换成了很常见的ReLU。

在[神经网络简明教程第8步/18.0-经典卷积神经网络模型](https://github.com/microsoft/ai-edu/blob/master/A-基础教程/A2-神经网络基本原理/第8步%20-%20卷积神经网络/18.0-经典卷积神经网络模型.md#1801-lenet-1998)中，有对LeNet和其他模型的简明介绍，可供了解。

我们详细分析`forward`函数中的运算，以及`x`的变化：

CIFAR10数据集中大小固定为32x32，假设`batch_size`为4，那么最开始的`x`在3维空间上，shape为(4, 32, 32)。

经过`self.conv1`后，`x`升到6维空间，shape变为(4, 28, 28)，再经过2x2的池化后，shape变为(4, 14, 14)。

经过`self.conv2`后，`x`升到16维空间，shape变为(4, 10, 10)，再经过2x2的池化后，shape变为(4, 5, 5)。

之后的`x = x.view(-1, 16 * 5 * 5)`，将同一数据的特征reshape到同一列，-1所代表的就是`batch_size`。

接下来三个全连接层，将$16 \times 5 \times 5 = 400$个feature逐步映射到10个类别，最终实现了10分类。

#### 神经网络的训练

而训练过程中，使用PyTorch的写法是这样的：

```python
def train(trainloader, path):
    neuralnet = NeuralNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(neuralnet.parameters(), lr=0.001, momentum=0.9)
    neuralnet.train()
    for epoch in range(10):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            # training template for PyTorch
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
```

#### 神经网络的测试

```python
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
```

### 结果分析

经10个epoch的训练，最终输出结果如下：

```
[    1,  2000] loss = 2.16590
[    1,  4000] loss = 1.82480
[    1,  6000] loss = 1.64638
[    1,  8000] loss = 1.56156
[    1, 10000] loss = 1.49378
[    1, 12000] loss = 1.46539
[    2,  2000] loss = 1.39108
[    2,  4000] loss = 1.38308
[    2,  6000] loss = 1.36254
[    2,  8000] loss = 1.30314
[    2, 10000] loss = 1.30563
[    2, 12000] loss = 1.26935
[    3,  2000] loss = 1.21411
[    3,  4000] loss = 1.21809
[    3,  6000] loss = 1.17786
[    3,  8000] loss = 1.18651
[    3, 10000] loss = 1.16956
[    3, 12000] loss = 1.16728
[    4,  2000] loss = 1.10504
[    4,  4000] loss = 1.11141
[    4,  6000] loss = 1.07836
[    4,  8000] loss = 1.10194
[    4, 10000] loss = 1.07333
[    4, 12000] loss = 1.06928
[    5,  2000] loss = 0.98897
[    5,  4000] loss = 1.01186
[    5,  6000] loss = 1.01296
[    5,  8000] loss = 1.01628
[    5, 10000] loss = 1.02610
[    5, 12000] loss = 1.03693
[    6,  2000] loss = 0.94843
[    6,  4000] loss = 0.94470
[    6,  6000] loss = 0.96298
[    6,  8000] loss = 0.96035
[    6, 10000] loss = 0.98843
[    6, 12000] loss = 0.96657
[    7,  2000] loss = 0.87795
[    7,  4000] loss = 0.90013
[    7,  6000] loss = 0.91402
[    7,  8000] loss = 0.94256
[    7, 10000] loss = 0.93912
[    7, 12000] loss = 0.91624
[    8,  2000] loss = 0.84444
[    8,  4000] loss = 0.85796
[    8,  6000] loss = 0.90461
[    8,  8000] loss = 0.89855
[    8, 10000] loss = 0.89341
[    8, 12000] loss = 0.89116
[    9,  2000] loss = 0.79060
[    9,  4000] loss = 0.83296
[    9,  6000] loss = 0.84468
[    9,  8000] loss = 0.85216
[    9, 10000] loss = 0.86738
[    9, 12000] loss = 0.87915
[   10,  2000] loss = 0.76653
[   10,  4000] loss = 0.80672
[   10,  6000] loss = 0.82791
[   10,  8000] loss = 0.80691
[   10, 10000] loss = 0.83649
[   10, 12000] loss = 0.84138
Training Finished
Accuracy of plane: 81.14%
Accuracy of car: 92.10%
Accuracy of bird: 74.58%
Accuracy of cat: 47.94%
Accuracy of deer: 65.08%
Accuracy of dog: 61.28%
Accuracy of frog: 71.88%
Accuracy of horse: 73.24%
Accuracy of ship: 86.18%
Accuracy of truck: 66.52%
Testing Finished
```
可以看出，损失值总体稳定下降，对车、飞机、船等图像分类准确率较高，而对猫、狗、卡车等图像的准确率较不理想。

如何提高部分不理想的分类准确率？请看Task 2.2......


## Task 2.2

to be continued...
>>>>>>> master
