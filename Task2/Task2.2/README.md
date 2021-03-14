# Task2.2 实验报告

## 超参调优

NNI支持通过配置搜索空间自定义搜索结构，不仅能够运用SOTA的高效率算法进行自动超参调优，更能够在多个模型与超参中选择出性能更优的组合，从而提高模型准确率。

搜索空间配置文件如下：

```json
{
    "lr":{"_type":"choice", "_value":[0.01, 0.001]},
    "optimizer":{"_type":"choice", "_value":["Adadelta", "Adagrad", "Adam", "Adamax"]},
    "model":{"_type":"choice", "_value":["vgg", "resnet18", "googlenet", "densenet121", "mobilenet", "dpn92", "shufflenetg2","senet18"]}
}
```

在代码中，当前trial的超参组合可从`nni.get_next_parameter()`获得，并以dict的形式保存。

### 实验结果

由于设备性能有限，在进行NNI的HPO实验时每个trial的epoch数并没能设定得足够多，这极有可能会导致最终的实验结果与retrain的性能不相符。我们训练了HPO实验中metric最优的超参组合，后续效果却不尽人意。反而是在非最优的超参组合中，我们训练出了较好的效果。

最终我们选定了学习率为0.01，优化器为Adamax，神经网络模型为mobilenet的组合，并在重训练了200个epoch后取得了96.66%的准确率。

## Classic NAS

通过上一步得到的模型与参数的组合，我们尝试在搜索空间上定义随机结构，测试模型的性能。

神经网络的随机结构可以借助NNI的经典NAS算法来实现，随机架构的搜索tuner可以在NNI的example中找到。

编写随机结构的搜索空间时，可以使用`nni.nas.pytorch.mutables`中的`LayerChoice`和`Inputchoice`来进行实现。

这两种定义待选连接的方式都很方便。`LayerChoice`在代码使用上可视作与其他普通神经网络等同，而要实现`InputChoice`所代表的跳过连接，则与`InputChoice`的输出进行concat操作即可。

这里定义了MobileNet的随机结构，代码如下：

```python
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
    cfg = [64, (128, 2), 128, 128, (256, 2), 256, 256, (512, 2), 512, 512, 512, 512, 512, 512, (1024, 2), 1024, 1024]
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

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        i = 0
        cnt = 1
        for block in self.layers:
            old_x = x
            x = block(x)
            if block.in_channels == block.out_channels:
                i += 1
            else:
                i = 0
            if i >= 2 and i % 2 == 0:
                zero_x = torch.zeros_like(old_x)
                skipconnect = eval('self.skipconnect{}'.format(cnt))
                skip_x = skipconnect([zero_x, old_x])
                x = torch.add(x, skip_x)
                cnt += 1

        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x
```

### 最终结果

![](images/capture.png)

![](images/capture2.png)

## One-Shot NAS

上一步的经典NAS算法对原始模型的改动较小，也得不到较大的优化效果。并且，在6层可选跳过的情况下，神经网络的深度会下降，挖掘深层特征的潜力也会有所降低，所以最后准确率不仅没有升高，反而效果不尽人意。

在这一步，我们尝试One-Shot NAS，通过定义DARTS的搜索空间，大幅度改变原有模型，尝试真正意义上提高预测的精确度至97%以上。

### DARTS原理简要分析

DARTS全称Differentiable Architecture Search，是NAS领域中著名的算法之一。该算法的特色是将若干个待搜索的架构从互不关联的“黑箱优化”问题变成可松弛的连续优化问题，通过梯度下降来进行更新。

由于需求只是构造MobileNet的搜索空间，这里只对CNN的DARTS进行分析。

首先，如果将一个状态看作一个节点，把一种操作看作一条边，那么CNN的网络模型就可以抽象成一个有向无环图（DAG）。

而在我们进行搜索的过程中，两点之间其实包含有“重边”。这些“重边”虽然两端节点相同，但各代表着不同的操作。我们需要做的，就是在这些待选边中找出整体最适合的一条边来成为DAG的一部分，实现架构的搜索。

我们首先给cell下定义。一个cell是一个包含了$N$个节点的有向无环图。其中编号为$i$的节点$x^{(i)}$代表着特征所存在着的状态，而从$i$到$j$的一条有向边就代表着一种操作，这种操作记为$o^{(i,j)}$。

接下来定义cell的输入与输出。一个cell会有两个输入，而只会有一个输出。这个cell的输出是将所有前面节点的操作concat起来的结果。用公式写出来就是：

$$x^{(j)} = \int_{i<j} o^{(i, j)}(x^{(i)})$$

这里$o^{(i,j)}(x)$代表着将$x$所代表的状态经过$(i,j)$这条有向边所代表的操作后所得到的新状态。正如直观感觉一般，也就是可以抽象成一个函数。

一条边所代表的，可以是一个池化层，可以是标准的Conv+BatchNorm组合，也可以是SkipConnect等其他的子模型。

定义的这些模型只需要满足一个共性：需要满足原数据的width和height不能改变。也就是类似于：

当卷积核size为3x3时，padding为1；当kernal size为5x5时，padding为2；当kernal size为1x1时，padding为0...

接下来令$x^{(i)}$和$x^{(j)}$这两点之间的$n$条所有候选边的集合为$\mathcal{O}$，$\alpha_o^{(i,j)}$是一个$n$维的向量，分别代表着每一个待选操作的得分。将这些得分进行softmax运算进行松弛，公式如下：

$$\overline o^{(i,j)}(x) = \sum_{o\in \mathcal{O}}\frac{\exp(\alpha_o^{(i,j)})}{\sum_{o'\in \mathcal{O}} \exp(\alpha_{o'}^{(i,j)})}o(x)$$

$\overline o^{(i,j)}(x)$最终最可能会选中得分最高的架构，即：

$$o^{(i,j)}=\argmax_{o\in \mathcal{O}}\alpha_o^{(i,j)}$$

最终我们所求的是集合$\alpha=\{\alpha^{(i,j)}\}$。

如何求得$\alpha$？我们需要训练集和验证集的协助。

设训练集和验证集的loss分别为$\mathcal{L}_{train}$和$\mathcal{L}_{val}$。我们要求$\alpha$，最理想的状况是存在最优架构$\alpha^\star$, 能够使得$\mathcal L_{val}(w^\star, \alpha^\star)$达到最小。而最优参数$w^\star$是通过训练集不断地训练出来的。也就是$w^\star= \argmin _w \mathcal{L}_{train}(w, \alpha^\star)$。

这样就需要解决一个双优化问题：

$$\begin{aligned}
    & \min_\alpha \mathcal L_{val}(w^*, \alpha^*) \\
    &s. t. \quad w^*=\argmin_w \mathcal L_{train}(w, \alpha^*)
\end{aligned}$$

求解这个双优化问题的算法大体的思路是：固定架构参数，用训练数据集训练模型参数，再固定模型参数，用验证数据集训练架构参数。

DARTS算法将动辄耗费上千个GPU天的神经网络架构搜索缩短至1至4个GPU天，使得NAS应用的门槛和成本大幅度降低。

## 代码实现部分

由于NNI中对DARTS的基本单位cell做了封装，候选边已经包含了常见的SepConv3x3、SepConv5x5、DilConv3x3、DivConv5x5、平均池化层、最大池化层、跳过连接层等候选架构，可以通过调用`nni.nas.pytorch.search_space_zoo.DartsCell`直接使用默认cell结构。

最终待搜索的模型可以基于`DartsCell`来构造：

```python
class DartsStackedCells(nn.Module):
    def drop_path_probability(self, p):
        for module in self.modules():
            if isinstance(module, DropPath):
                module.p = p

    def __init__(self, in_channels, channels, n_classes, n_layers, n_nodes=4, stem_multiplier=2):
        super(DartsStackedCells, self).__init__()
        self.in_channels = in_channels
        self.channels = channels
        self.n_classes = n_classes
        self.n_layers = n_layers
        cur_channels = stem_multiplier * self.channels
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, cur_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(cur_channels)
        )
        pp_channels, p_channels, cur_channels = cur_channels, cur_channels, channels
        self.cells = nn.ModuleList()
        p_reduction, cur_reduction = False, False
        for i in range(n_layers):
            p_reduction, cur_reduction = cur_reduction, False
            if i in [n_layers // 3, 2 * n_layers // 3]:
                cur_channels *= 2
                cur_reduction = True
            self.cells.append(DartsCell(n_nodes, pp_channels, p_channels, cur_channels, p_reduction, cur_reduction))
            cur_channels_out = cur_channels * n_nodes
            pp_channels, p_channels = p_channels, cur_channels_out
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(p_channels, n_classes)

    def forward(self, x):
        s0 = s1 = self.stem(x)
        for cell in self.cells:
            s0, s1 = s1, cell(s0, s1)
        output = self.gap(s1)
        output = output.view(output.size(0), -1)
        output = self.linear(output)
        return output

```

最终搜索出的结构如下：

```json
{
  "normal_n2_p0": "sepconv5x5",
  "normal_n2_p1": "sepconv3x3",
  "normal_n2_switch": [
    "normal_n2_p0",
    "normal_n2_p1"
  ],
  "normal_n3_p0": "skipconnect",
  "normal_n3_p1": "sepconv3x3",
  "normal_n3_p2": [],
  "normal_n3_switch": [
    "normal_n3_p0",
    "normal_n3_p1"
  ],
  "normal_n4_p0": "skipconnect",
  "normal_n4_p1": "sepconv3x3",
  "normal_n4_p2": [],
  "normal_n4_p3": [],
  "normal_n4_switch": [
    "normal_n4_p0",
    "normal_n4_p1"
  ],
  "normal_n5_p0": "dilconv3x3",
  "normal_n5_p1": "dilconv5x5",
  "normal_n5_p2": [],
  "normal_n5_p3": [],
  "normal_n5_p4": [],
  "normal_n5_switch": [
    "normal_n5_p0",
    "normal_n5_p1"
  ],
  "reduce_n2_p0": "maxpool",
  "reduce_n2_p1": "maxpool",
  "reduce_n2_switch": [
    "reduce_n2_p0",
    "reduce_n2_p1"
  ],
  "reduce_n3_p0": "maxpool",
  "reduce_n3_p1": [],
  "reduce_n3_p2": "skipconnect",
  "reduce_n3_switch": [
    "reduce_n3_p0",
    "reduce_n3_p2"
  ],
  "reduce_n4_p0": "maxpool",
  "reduce_n4_p1": [],
  "reduce_n4_p2": "skipconnect",
  "reduce_n4_p3": [],
  "reduce_n4_switch": [
    "reduce_n4_p0",
    "reduce_n4_p2"
  ],
  "reduce_n5_p0": "avgpool",
  "reduce_n5_p1": [],
  "reduce_n5_p2": "skipconnect",
  "reduce_n5_p3": [],
  "reduce_n5_p4": [],
  "reduce_n5_switch": [
    "reduce_n5_p0",
    "reduce_n5_p2"
  ]
}
```

retrain日志如下（仅截取第453个epoch和最后两个epoch）：

```
[2021-02-22 16:16:01] INFO (nni/MainThread) Epoch 453 LR 0.003524
[2021-02-22 16:16:02] INFO (nni/MainThread) Train: [454/600] Step 000/520 Loss 0.146 Prec@(1,5) (94.8%, 100.0%)
[2021-02-22 16:16:04] INFO (nni/MainThread) Train: [454/600] Step 010/520 Loss 0.142 Prec@(1,5) (96.5%, 100.0%)
[2021-02-22 16:16:07] INFO (nni/MainThread) Train: [454/600] Step 020/520 Loss 0.151 Prec@(1,5) (96.5%, 100.0%)
[2021-02-22 16:16:10] INFO (nni/MainThread) Train: [454/600] Step 030/520 Loss 0.137 Prec@(1,5) (96.9%, 100.0%)
[2021-02-22 16:16:13] INFO (nni/MainThread) Train: [454/600] Step 040/520 Loss 0.147 Prec@(1,5) (96.9%, 100.0%)
[2021-02-22 16:16:15] INFO (nni/MainThread) Train: [454/600] Step 050/520 Loss 0.146 Prec@(1,5) (97.0%, 100.0%)
[2021-02-22 16:16:18] INFO (nni/MainThread) Train: [454/600] Step 060/520 Loss 0.144 Prec@(1,5) (97.0%, 100.0%)
[2021-02-22 16:16:21] INFO (nni/MainThread) Train: [454/600] Step 070/520 Loss 0.145 Prec@(1,5) (97.0%, 100.0%)
[2021-02-22 16:16:23] INFO (nni/MainThread) Train: [454/600] Step 080/520 Loss 0.144 Prec@(1,5) (96.9%, 100.0%)
[2021-02-22 16:16:26] INFO (nni/MainThread) Train: [454/600] Step 090/520 Loss 0.143 Prec@(1,5) (96.9%, 100.0%)
[2021-02-22 16:16:29] INFO (nni/MainThread) Train: [454/600] Step 100/520 Loss 0.141 Prec@(1,5) (97.0%, 100.0%)
[2021-02-22 16:16:31] INFO (nni/MainThread) Train: [454/600] Step 110/520 Loss 0.139 Prec@(1,5) (97.0%, 100.0%)
[2021-02-22 16:16:34] INFO (nni/MainThread) Train: [454/600] Step 120/520 Loss 0.139 Prec@(1,5) (97.1%, 100.0%)
[2021-02-22 16:16:37] INFO (nni/MainThread) Train: [454/600] Step 130/520 Loss 0.138 Prec@(1,5) (97.1%, 100.0%)
[2021-02-22 16:16:40] INFO (nni/MainThread) Train: [454/600] Step 140/520 Loss 0.140 Prec@(1,5) (97.0%, 100.0%)
[2021-02-22 16:16:42] INFO (nni/MainThread) Train: [454/600] Step 150/520 Loss 0.139 Prec@(1,5) (97.0%, 100.0%)
[2021-02-22 16:16:45] INFO (nni/MainThread) Train: [454/600] Step 160/520 Loss 0.142 Prec@(1,5) (97.0%, 100.0%)
[2021-02-22 16:16:48] INFO (nni/MainThread) Train: [454/600] Step 170/520 Loss 0.141 Prec@(1,5) (97.0%, 100.0%)
[2021-02-22 16:16:50] INFO (nni/MainThread) Train: [454/600] Step 180/520 Loss 0.141 Prec@(1,5) (97.0%, 100.0%)
[2021-02-22 16:16:53] INFO (nni/MainThread) Train: [454/600] Step 190/520 Loss 0.141 Prec@(1,5) (97.1%, 100.0%)
[2021-02-22 16:16:56] INFO (nni/MainThread) Train: [454/600] Step 200/520 Loss 0.141 Prec@(1,5) (97.0%, 100.0%)
[2021-02-22 16:16:59] INFO (nni/MainThread) Train: [454/600] Step 210/520 Loss 0.142 Prec@(1,5) (97.0%, 100.0%)
[2021-02-22 16:17:01] INFO (nni/MainThread) Train: [454/600] Step 220/520 Loss 0.142 Prec@(1,5) (97.0%, 100.0%)
[2021-02-22 16:17:04] INFO (nni/MainThread) Train: [454/600] Step 230/520 Loss 0.142 Prec@(1,5) (97.1%, 100.0%)
[2021-02-22 16:17:07] INFO (nni/MainThread) Train: [454/600] Step 240/520 Loss 0.142 Prec@(1,5) (97.0%, 100.0%)
[2021-02-22 16:17:09] INFO (nni/MainThread) Train: [454/600] Step 250/520 Loss 0.141 Prec@(1,5) (97.0%, 100.0%)
[2021-02-22 16:17:12] INFO (nni/MainThread) Train: [454/600] Step 260/520 Loss 0.141 Prec@(1,5) (97.1%, 100.0%)
[2021-02-22 16:17:15] INFO (nni/MainThread) Train: [454/600] Step 270/520 Loss 0.140 Prec@(1,5) (97.1%, 100.0%)
[2021-02-22 16:17:17] INFO (nni/MainThread) Train: [454/600] Step 280/520 Loss 0.141 Prec@(1,5) (97.1%, 100.0%)
[2021-02-22 16:17:20] INFO (nni/MainThread) Train: [454/600] Step 290/520 Loss 0.140 Prec@(1,5) (97.1%, 100.0%)
[2021-02-22 16:17:23] INFO (nni/MainThread) Train: [454/600] Step 300/520 Loss 0.140 Prec@(1,5) (97.1%, 100.0%)
[2021-02-22 16:17:26] INFO (nni/MainThread) Train: [454/600] Step 310/520 Loss 0.140 Prec@(1,5) (97.1%, 100.0%)
[2021-02-22 16:17:28] INFO (nni/MainThread) Train: [454/600] Step 320/520 Loss 0.140 Prec@(1,5) (97.1%, 100.0%)
[2021-02-22 16:17:31] INFO (nni/MainThread) Train: [454/600] Step 330/520 Loss 0.141 Prec@(1,5) (97.1%, 100.0%)
[2021-02-22 16:17:34] INFO (nni/MainThread) Train: [454/600] Step 340/520 Loss 0.141 Prec@(1,5) (97.1%, 100.0%)
[2021-02-22 16:17:36] INFO (nni/MainThread) Train: [454/600] Step 350/520 Loss 0.140 Prec@(1,5) (97.1%, 100.0%)
[2021-02-22 16:17:39] INFO (nni/MainThread) Train: [454/600] Step 360/520 Loss 0.142 Prec@(1,5) (97.1%, 100.0%)
[2021-02-22 16:17:42] INFO (nni/MainThread) Train: [454/600] Step 370/520 Loss 0.142 Prec@(1,5) (97.1%, 100.0%)
[2021-02-22 16:17:45] INFO (nni/MainThread) Train: [454/600] Step 380/520 Loss 0.142 Prec@(1,5) (97.1%, 100.0%)
[2021-02-22 16:17:47] INFO (nni/MainThread) Train: [454/600] Step 390/520 Loss 0.142 Prec@(1,5) (97.0%, 100.0%)
[2021-02-22 16:17:50] INFO (nni/MainThread) Train: [454/600] Step 400/520 Loss 0.143 Prec@(1,5) (97.0%, 100.0%)
[2021-02-22 16:17:53] INFO (nni/MainThread) Train: [454/600] Step 410/520 Loss 0.142 Prec@(1,5) (97.1%, 100.0%)
[2021-02-22 16:17:55] INFO (nni/MainThread) Train: [454/600] Step 420/520 Loss 0.142 Prec@(1,5) (97.1%, 100.0%)
[2021-02-22 16:17:58] INFO (nni/MainThread) Train: [454/600] Step 430/520 Loss 0.141 Prec@(1,5) (97.1%, 100.0%)
[2021-02-22 16:18:01] INFO (nni/MainThread) Train: [454/600] Step 440/520 Loss 0.143 Prec@(1,5) (97.1%, 100.0%)
[2021-02-22 16:18:04] INFO (nni/MainThread) Train: [454/600] Step 450/520 Loss 0.143 Prec@(1,5) (97.1%, 100.0%)
[2021-02-22 16:18:06] INFO (nni/MainThread) Train: [454/600] Step 460/520 Loss 0.143 Prec@(1,5) (97.1%, 100.0%)
[2021-02-22 16:18:09] INFO (nni/MainThread) Train: [454/600] Step 470/520 Loss 0.144 Prec@(1,5) (97.0%, 100.0%)
[2021-02-22 16:18:12] INFO (nni/MainThread) Train: [454/600] Step 480/520 Loss 0.143 Prec@(1,5) (97.1%, 100.0%)
[2021-02-22 16:18:14] INFO (nni/MainThread) Train: [454/600] Step 490/520 Loss 0.142 Prec@(1,5) (97.1%, 100.0%)
[2021-02-22 16:18:17] INFO (nni/MainThread) Train: [454/600] Step 500/520 Loss 0.143 Prec@(1,5) (97.1%, 100.0%)
[2021-02-22 16:18:20] INFO (nni/MainThread) Train: [454/600] Step 510/520 Loss 0.143 Prec@(1,5) (97.1%, 100.0%)
[2021-02-22 16:18:23] INFO (nni/MainThread) Train: [454/600] Step 520/520 Loss 0.143 Prec@(1,5) (97.1%, 100.0%)
[2021-02-22 16:18:23] INFO (nni/MainThread) Train: [454/600] Final Prec@1 97.0640%
...
[2021-02-22 21:37:03] INFO (nni/MainThread) Train: [599/600] Step 000/520 Loss 0.054 Prec@(1,5) (99.0%, 100.0%)
[2021-02-22 21:37:05] INFO (nni/MainThread) Train: [599/600] Step 010/520 Loss 0.060 Prec@(1,5) (99.1%, 100.0%)
[2021-02-22 21:37:08] INFO (nni/MainThread) Train: [599/600] Step 020/520 Loss 0.058 Prec@(1,5) (99.0%, 100.0%)
[2021-02-22 21:37:10] INFO (nni/MainThread) Train: [599/600] Step 030/520 Loss 0.054 Prec@(1,5) (99.1%, 100.0%)
[2021-02-22 21:37:12] INFO (nni/MainThread) Train: [599/600] Step 040/520 Loss 0.059 Prec@(1,5) (98.9%, 100.0%)
[2021-02-22 21:37:15] INFO (nni/MainThread) Train: [599/600] Step 050/520 Loss 0.058 Prec@(1,5) (99.0%, 100.0%)
[2021-02-22 21:37:17] INFO (nni/MainThread) Train: [599/600] Step 060/520 Loss 0.064 Prec@(1,5) (98.8%, 100.0%)
[2021-02-22 21:37:20] INFO (nni/MainThread) Train: [599/600] Step 070/520 Loss 0.066 Prec@(1,5) (98.8%, 100.0%)
[2021-02-22 21:37:22] INFO (nni/MainThread) Train: [599/600] Step 080/520 Loss 0.068 Prec@(1,5) (98.8%, 100.0%)
[2021-02-22 21:37:24] INFO (nni/MainThread) Train: [599/600] Step 090/520 Loss 0.067 Prec@(1,5) (98.8%, 100.0%)
[2021-02-22 21:37:27] INFO (nni/MainThread) Train: [599/600] Step 100/520 Loss 0.069 Prec@(1,5) (98.8%, 100.0%)
[2021-02-22 21:37:29] INFO (nni/MainThread) Train: [599/600] Step 110/520 Loss 0.070 Prec@(1,5) (98.8%, 100.0%)
[2021-02-22 21:37:32] INFO (nni/MainThread) Train: [599/600] Step 120/520 Loss 0.070 Prec@(1,5) (98.8%, 100.0%)
[2021-02-22 21:37:34] INFO (nni/MainThread) Train: [599/600] Step 130/520 Loss 0.070 Prec@(1,5) (98.8%, 100.0%)
[2021-02-22 21:37:37] INFO (nni/MainThread) Train: [599/600] Step 140/520 Loss 0.070 Prec@(1,5) (98.8%, 100.0%)
[2021-02-22 21:37:39] INFO (nni/MainThread) Train: [599/600] Step 150/520 Loss 0.070 Prec@(1,5) (98.8%, 100.0%)
[2021-02-22 21:37:41] INFO (nni/MainThread) Train: [599/600] Step 160/520 Loss 0.070 Prec@(1,5) (98.7%, 100.0%)
[2021-02-22 21:37:44] INFO (nni/MainThread) Train: [599/600] Step 170/520 Loss 0.071 Prec@(1,5) (98.7%, 100.0%)
[2021-02-22 21:37:46] INFO (nni/MainThread) Train: [599/600] Step 180/520 Loss 0.072 Prec@(1,5) (98.6%, 100.0%)
[2021-02-22 21:37:49] INFO (nni/MainThread) Train: [599/600] Step 190/520 Loss 0.072 Prec@(1,5) (98.6%, 100.0%)
[2021-02-22 21:37:51] INFO (nni/MainThread) Train: [599/600] Step 200/520 Loss 0.072 Prec@(1,5) (98.7%, 100.0%)
[2021-02-22 21:37:53] INFO (nni/MainThread) Train: [599/600] Step 210/520 Loss 0.072 Prec@(1,5) (98.6%, 100.0%)
[2021-02-22 21:37:56] INFO (nni/MainThread) Train: [599/600] Step 220/520 Loss 0.071 Prec@(1,5) (98.7%, 100.0%)
[2021-02-22 21:37:58] INFO (nni/MainThread) Train: [599/600] Step 230/520 Loss 0.070 Prec@(1,5) (98.7%, 100.0%)
[2021-02-22 21:38:01] INFO (nni/MainThread) Train: [599/600] Step 240/520 Loss 0.070 Prec@(1,5) (98.7%, 100.0%)
[2021-02-22 21:38:03] INFO (nni/MainThread) Train: [599/600] Step 250/520 Loss 0.072 Prec@(1,5) (98.7%, 100.0%)
[2021-02-22 21:38:06] INFO (nni/MainThread) Train: [599/600] Step 260/520 Loss 0.072 Prec@(1,5) (98.7%, 100.0%)
[2021-02-22 21:38:08] INFO (nni/MainThread) Train: [599/600] Step 270/520 Loss 0.071 Prec@(1,5) (98.7%, 100.0%)
[2021-02-22 21:38:10] INFO (nni/MainThread) Train: [599/600] Step 280/520 Loss 0.071 Prec@(1,5) (98.7%, 100.0%)
[2021-02-22 21:38:13] INFO (nni/MainThread) Train: [599/600] Step 290/520 Loss 0.071 Prec@(1,5) (98.7%, 100.0%)
[2021-02-22 21:38:15] INFO (nni/MainThread) Train: [599/600] Step 300/520 Loss 0.070 Prec@(1,5) (98.7%, 100.0%)
[2021-02-22 21:38:18] INFO (nni/MainThread) Train: [599/600] Step 310/520 Loss 0.069 Prec@(1,5) (98.7%, 100.0%)
[2021-02-22 21:38:20] INFO (nni/MainThread) Train: [599/600] Step 320/520 Loss 0.070 Prec@(1,5) (98.7%, 100.0%)
[2021-02-22 21:38:22] INFO (nni/MainThread) Train: [599/600] Step 330/520 Loss 0.070 Prec@(1,5) (98.7%, 100.0%)
[2021-02-22 21:38:25] INFO (nni/MainThread) Train: [599/600] Step 340/520 Loss 0.069 Prec@(1,5) (98.7%, 100.0%)
[2021-02-22 21:38:27] INFO (nni/MainThread) Train: [599/600] Step 350/520 Loss 0.070 Prec@(1,5) (98.7%, 100.0%)
[2021-02-22 21:38:30] INFO (nni/MainThread) Train: [599/600] Step 360/520 Loss 0.070 Prec@(1,5) (98.7%, 100.0%)
[2021-02-22 21:38:32] INFO (nni/MainThread) Train: [599/600] Step 370/520 Loss 0.070 Prec@(1,5) (98.7%, 100.0%)
[2021-02-22 21:38:34] INFO (nni/MainThread) Train: [599/600] Step 380/520 Loss 0.070 Prec@(1,5) (98.7%, 100.0%)
[2021-02-22 21:38:37] INFO (nni/MainThread) Train: [599/600] Step 390/520 Loss 0.071 Prec@(1,5) (98.7%, 100.0%)
[2021-02-22 21:38:39] INFO (nni/MainThread) Train: [599/600] Step 400/520 Loss 0.071 Prec@(1,5) (98.7%, 100.0%)
[2021-02-22 21:38:42] INFO (nni/MainThread) Train: [599/600] Step 410/520 Loss 0.071 Prec@(1,5) (98.7%, 100.0%)
[2021-02-22 21:38:44] INFO (nni/MainThread) Train: [599/600] Step 420/520 Loss 0.071 Prec@(1,5) (98.7%, 100.0%)
[2021-02-22 21:38:47] INFO (nni/MainThread) Train: [599/600] Step 430/520 Loss 0.071 Prec@(1,5) (98.7%, 100.0%)
[2021-02-22 21:38:49] INFO (nni/MainThread) Train: [599/600] Step 440/520 Loss 0.071 Prec@(1,5) (98.7%, 100.0%)
[2021-02-22 21:38:51] INFO (nni/MainThread) Train: [599/600] Step 450/520 Loss 0.071 Prec@(1,5) (98.7%, 100.0%)
[2021-02-22 21:38:54] INFO (nni/MainThread) Train: [599/600] Step 460/520 Loss 0.070 Prec@(1,5) (98.7%, 100.0%)
[2021-02-22 21:38:56] INFO (nni/MainThread) Train: [599/600] Step 470/520 Loss 0.070 Prec@(1,5) (98.8%, 100.0%)
[2021-02-22 21:38:59] INFO (nni/MainThread) Train: [599/600] Step 480/520 Loss 0.070 Prec@(1,5) (98.8%, 100.0%)
[2021-02-22 21:39:01] INFO (nni/MainThread) Train: [599/600] Step 490/520 Loss 0.070 Prec@(1,5) (98.8%, 100.0%)
[2021-02-22 21:39:03] INFO (nni/MainThread) Train: [599/600] Step 500/520 Loss 0.069 Prec@(1,5) (98.8%, 100.0%)
[2021-02-22 21:39:06] INFO (nni/MainThread) Train: [599/600] Step 510/520 Loss 0.070 Prec@(1,5) (98.8%, 100.0%)
[2021-02-22 21:39:08] INFO (nni/MainThread) Train: [599/600] Step 520/520 Loss 0.070 Prec@(1,5) (98.8%, 100.0%)
[2021-02-22 21:39:08] INFO (nni/MainThread) Train: [599/600] Final Prec@1 98.7600%
[2021-02-22 21:39:09] INFO (nni/MainThread) Valid: [599/600] Step 000/104 Loss 0.146 Prec@(1,5) (96.9%, 100.0%)
[2021-02-22 21:39:09] INFO (nni/MainThread) Valid: [599/600] Step 010/104 Loss 0.113 Prec@(1,5) (96.9%, 100.0%)
[2021-02-22 21:39:10] INFO (nni/MainThread) Valid: [599/600] Step 020/104 Loss 0.135 Prec@(1,5) (97.0%, 100.0%)
[2021-02-22 21:39:10] INFO (nni/MainThread) Valid: [599/600] Step 030/104 Loss 0.163 Prec@(1,5) (96.9%, 99.9%)
[2021-02-22 21:39:11] INFO (nni/MainThread) Valid: [599/600] Step 040/104 Loss 0.159 Prec@(1,5) (97.0%, 99.9%)
[2021-02-22 21:39:11] INFO (nni/MainThread) Valid: [599/600] Step 050/104 Loss 0.155 Prec@(1,5) (97.0%, 99.9%)
[2021-02-22 21:39:12] INFO (nni/MainThread) Valid: [599/600] Step 060/104 Loss 0.150 Prec@(1,5) (97.1%, 99.9%)
[2021-02-22 21:39:12] INFO (nni/MainThread) Valid: [599/600] Step 070/104 Loss 0.140 Prec@(1,5) (97.2%, 100.0%)
[2021-02-22 21:39:13] INFO (nni/MainThread) Valid: [599/600] Step 080/104 Loss 0.143 Prec@(1,5) (97.2%, 99.9%)
[2021-02-22 21:39:13] INFO (nni/MainThread) Valid: [599/600] Step 090/104 Loss 0.137 Prec@(1,5) (97.3%, 99.9%)
[2021-02-22 21:39:14] INFO (nni/MainThread) Valid: [599/600] Step 100/104 Loss 0.139 Prec@(1,5) (97.2%, 99.9%)
[2021-02-22 21:39:14] INFO (nni/MainThread) Valid: [599/600] Step 104/104 Loss 0.140 Prec@(1,5) (97.2%, 100.0%)
[2021-02-22 21:39:14] INFO (nni/MainThread) Valid: [599/600] Final Prec@1 97.2400%
[2021-02-22 21:39:14] INFO (nni/MainThread) Epoch 599 LR 0.000001
[2021-02-22 21:39:14] INFO (nni/MainThread) Train: [600/600] Step 000/520 Loss 0.039 Prec@(1,5) (100.0%, 100.0%)
[2021-02-22 21:39:17] INFO (nni/MainThread) Train: [600/600] Step 010/520 Loss 0.063 Prec@(1,5) (98.9%, 100.0%)
[2021-02-22 21:39:19] INFO (nni/MainThread) Train: [600/600] Step 020/520 Loss 0.061 Prec@(1,5) (98.9%, 100.0%)
[2021-02-22 21:39:22] INFO (nni/MainThread) Train: [600/600] Step 030/520 Loss 0.064 Prec@(1,5) (98.9%, 100.0%)
[2021-02-22 21:39:24] INFO (nni/MainThread) Train: [600/600] Step 040/520 Loss 0.065 Prec@(1,5) (98.8%, 100.0%)
[2021-02-22 21:39:26] INFO (nni/MainThread) Train: [600/600] Step 050/520 Loss 0.065 Prec@(1,5) (98.9%, 100.0%)
[2021-02-22 21:39:29] INFO (nni/MainThread) Train: [600/600] Step 060/520 Loss 0.065 Prec@(1,5) (98.9%, 100.0%)
[2021-02-22 21:39:31] INFO (nni/MainThread) Train: [600/600] Step 070/520 Loss 0.066 Prec@(1,5) (98.8%, 100.0%)
[2021-02-22 21:39:34] INFO (nni/MainThread) Train: [600/600] Step 080/520 Loss 0.066 Prec@(1,5) (98.9%, 100.0%)
[2021-02-22 21:39:36] INFO (nni/MainThread) Train: [600/600] Step 090/520 Loss 0.067 Prec@(1,5) (98.9%, 100.0%)
[2021-02-22 21:39:39] INFO (nni/MainThread) Train: [600/600] Step 100/520 Loss 0.067 Prec@(1,5) (98.9%, 100.0%)
[2021-02-22 21:39:41] INFO (nni/MainThread) Train: [600/600] Step 110/520 Loss 0.069 Prec@(1,5) (98.9%, 100.0%)
[2021-02-22 21:39:43] INFO (nni/MainThread) Train: [600/600] Step 120/520 Loss 0.069 Prec@(1,5) (98.8%, 100.0%)
[2021-02-22 21:39:46] INFO (nni/MainThread) Train: [600/600] Step 130/520 Loss 0.069 Prec@(1,5) (98.8%, 100.0%)
[2021-02-22 21:39:48] INFO (nni/MainThread) Train: [600/600] Step 140/520 Loss 0.070 Prec@(1,5) (98.8%, 100.0%)
[2021-02-22 21:39:51] INFO (nni/MainThread) Train: [600/600] Step 150/520 Loss 0.070 Prec@(1,5) (98.8%, 100.0%)
[2021-02-22 21:39:53] INFO (nni/MainThread) Train: [600/600] Step 160/520 Loss 0.070 Prec@(1,5) (98.8%, 100.0%)
[2021-02-22 21:39:55] INFO (nni/MainThread) Train: [600/600] Step 170/520 Loss 0.071 Prec@(1,5) (98.8%, 100.0%)
[2021-02-22 21:39:58] INFO (nni/MainThread) Train: [600/600] Step 180/520 Loss 0.072 Prec@(1,5) (98.8%, 100.0%)
[2021-02-22 21:40:00] INFO (nni/MainThread) Train: [600/600] Step 190/520 Loss 0.072 Prec@(1,5) (98.8%, 100.0%)
[2021-02-22 21:40:03] INFO (nni/MainThread) Train: [600/600] Step 200/520 Loss 0.072 Prec@(1,5) (98.8%, 100.0%)
[2021-02-22 21:40:05] INFO (nni/MainThread) Train: [600/600] Step 210/520 Loss 0.072 Prec@(1,5) (98.7%, 100.0%)
[2021-02-22 21:40:08] INFO (nni/MainThread) Train: [600/600] Step 220/520 Loss 0.072 Prec@(1,5) (98.7%, 100.0%)
[2021-02-22 21:40:10] INFO (nni/MainThread) Train: [600/600] Step 230/520 Loss 0.072 Prec@(1,5) (98.7%, 100.0%)
[2021-02-22 21:40:12] INFO (nni/MainThread) Train: [600/600] Step 240/520 Loss 0.072 Prec@(1,5) (98.7%, 100.0%)
[2021-02-22 21:40:15] INFO (nni/MainThread) Train: [600/600] Step 250/520 Loss 0.071 Prec@(1,5) (98.8%, 100.0%)
[2021-02-22 21:40:17] INFO (nni/MainThread) Train: [600/600] Step 260/520 Loss 0.071 Prec@(1,5) (98.8%, 100.0%)
[2021-02-22 21:40:20] INFO (nni/MainThread) Train: [600/600] Step 270/520 Loss 0.070 Prec@(1,5) (98.8%, 100.0%)
[2021-02-22 21:40:22] INFO (nni/MainThread) Train: [600/600] Step 280/520 Loss 0.070 Prec@(1,5) (98.8%, 100.0%)
[2021-02-22 21:40:24] INFO (nni/MainThread) Train: [600/600] Step 290/520 Loss 0.071 Prec@(1,5) (98.8%, 100.0%)
[2021-02-22 21:40:27] INFO (nni/MainThread) Train: [600/600] Step 300/520 Loss 0.070 Prec@(1,5) (98.8%, 100.0%)
[2021-02-22 21:40:29] INFO (nni/MainThread) Train: [600/600] Step 310/520 Loss 0.071 Prec@(1,5) (98.8%, 100.0%)
[2021-02-22 21:40:32] INFO (nni/MainThread) Train: [600/600] Step 320/520 Loss 0.070 Prec@(1,5) (98.8%, 100.0%)
[2021-02-22 21:40:34] INFO (nni/MainThread) Train: [600/600] Step 330/520 Loss 0.070 Prec@(1,5) (98.8%, 100.0%)
[2021-02-22 21:40:36] INFO (nni/MainThread) Train: [600/600] Step 340/520 Loss 0.070 Prec@(1,5) (98.8%, 100.0%)
[2021-02-22 21:40:39] INFO (nni/MainThread) Train: [600/600] Step 350/520 Loss 0.070 Prec@(1,5) (98.8%, 100.0%)
[2021-02-22 21:40:41] INFO (nni/MainThread) Train: [600/600] Step 360/520 Loss 0.070 Prec@(1,5) (98.8%, 100.0%)
[2021-02-22 21:40:44] INFO (nni/MainThread) Train: [600/600] Step 370/520 Loss 0.070 Prec@(1,5) (98.8%, 100.0%)
[2021-02-22 21:40:46] INFO (nni/MainThread) Train: [600/600] Step 380/520 Loss 0.070 Prec@(1,5) (98.8%, 100.0%)
[2021-02-22 21:40:49] INFO (nni/MainThread) Train: [600/600] Step 390/520 Loss 0.070 Prec@(1,5) (98.8%, 100.0%)
[2021-02-22 21:40:51] INFO (nni/MainThread) Train: [600/600] Step 400/520 Loss 0.070 Prec@(1,5) (98.8%, 100.0%)
[2021-02-22 21:40:53] INFO (nni/MainThread) Train: [600/600] Step 410/520 Loss 0.071 Prec@(1,5) (98.8%, 100.0%)
[2021-02-22 21:40:56] INFO (nni/MainThread) Train: [600/600] Step 420/520 Loss 0.072 Prec@(1,5) (98.7%, 100.0%)
[2021-02-22 21:40:58] INFO (nni/MainThread) Train: [600/600] Step 430/520 Loss 0.071 Prec@(1,5) (98.8%, 100.0%)
[2021-02-22 21:41:01] INFO (nni/MainThread) Train: [600/600] Step 440/520 Loss 0.071 Prec@(1,5) (98.8%, 100.0%)
[2021-02-22 21:41:03] INFO (nni/MainThread) Train: [600/600] Step 450/520 Loss 0.071 Prec@(1,5) (98.8%, 100.0%)
[2021-02-22 21:41:05] INFO (nni/MainThread) Train: [600/600] Step 460/520 Loss 0.071 Prec@(1,5) (98.8%, 100.0%)
[2021-02-22 21:41:08] INFO (nni/MainThread) Train: [600/600] Step 470/520 Loss 0.071 Prec@(1,5) (98.8%, 100.0%)
[2021-02-22 21:41:10] INFO (nni/MainThread) Train: [600/600] Step 480/520 Loss 0.071 Prec@(1,5) (98.8%, 100.0%)
[2021-02-22 21:41:13] INFO (nni/MainThread) Train: [600/600] Step 490/520 Loss 0.070 Prec@(1,5) (98.8%, 100.0%)
[2021-02-22 21:41:15] INFO (nni/MainThread) Train: [600/600] Step 500/520 Loss 0.070 Prec@(1,5) (98.8%, 100.0%)
[2021-02-22 21:41:18] INFO (nni/MainThread) Train: [600/600] Step 510/520 Loss 0.070 Prec@(1,5) (98.8%, 100.0%)
[2021-02-22 21:41:20] INFO (nni/MainThread) Train: [600/600] Step 520/520 Loss 0.071 Prec@(1,5) (98.8%, 100.0%)
[2021-02-22 21:41:20] INFO (nni/MainThread) Train: [600/600] Final Prec@1 98.7720%
[2021-02-22 21:41:20] INFO (nni/MainThread) Valid: [600/600] Step 000/104 Loss 0.150 Prec@(1,5) (96.9%, 100.0%)
[2021-02-22 21:41:21] INFO (nni/MainThread) Valid: [600/600] Step 010/104 Loss 0.114 Prec@(1,5) (97.0%, 100.0%)
[2021-02-22 21:41:21] INFO (nni/MainThread) Valid: [600/600] Step 020/104 Loss 0.136 Prec@(1,5) (97.0%, 100.0%)
[2021-02-22 21:41:22] INFO (nni/MainThread) Valid: [600/600] Step 030/104 Loss 0.164 Prec@(1,5) (96.8%, 99.9%)
[2021-02-22 21:41:22] INFO (nni/MainThread) Valid: [600/600] Step 040/104 Loss 0.160 Prec@(1,5) (96.9%, 99.9%)
[2021-02-22 21:41:23] INFO (nni/MainThread) Valid: [600/600] Step 050/104 Loss 0.156 Prec@(1,5) (97.0%, 99.9%)
[2021-02-22 21:41:24] INFO (nni/MainThread) Valid: [600/600] Step 060/104 Loss 0.151 Prec@(1,5) (97.1%, 99.9%)
[2021-02-22 21:41:24] INFO (nni/MainThread) Valid: [600/600] Step 070/104 Loss 0.140 Prec@(1,5) (97.2%, 100.0%)
[2021-02-22 21:41:25] INFO (nni/MainThread) Valid: [600/600] Step 080/104 Loss 0.143 Prec@(1,5) (97.2%, 99.9%)
[2021-02-22 21:41:25] INFO (nni/MainThread) Valid: [600/600] Step 090/104 Loss 0.137 Prec@(1,5) (97.3%, 99.9%)
[2021-02-22 21:41:26] INFO (nni/MainThread) Valid: [600/600] Step 100/104 Loss 0.139 Prec@(1,5) (97.3%, 99.9%)
[2021-02-22 21:41:26] INFO (nni/MainThread) Valid: [600/600] Step 104/104 Loss 0.139 Prec@(1,5) (97.3%, 100.0%)
[2021-02-22 21:41:26] INFO (nni/MainThread) Valid: [600/600] Final Prec@1 97.2500%
```

调用`nni.algorithms.nas.pytorch.darts.DartsTrainer`可进行DARTS的架构搜索。再通过`retrain.py`再次进行训练，最终在第454个epoch达到了97%的准确率，重训练过程中精确度最高能够达到98%。

由于DARTS的候选边在MobileNet中并没有出现，所以最终生成的模型相对变化较大，但与此同时，性能上的提升也十分明显。

## 实验总结

- 我们使用了Google Colab上的GPU进行训练，通过NNI的官方文档提供的指导说明，通过反向代理访问到了NNI的Web UI，使得我们能够在有限的算力下完成NNI的有关实验。

- 在算力允许的条件下，每一次trial的epoch最好设置得足够大，这样所得的模型最终结果相对能够更加精确。

- NNI在超参调优和神经网络架构搜索方面真正解决了用户痛点所在，省下了繁琐的人工调参以及模型优化时间，以更低的时间成本，更高的工作效率为相关学习研究提供很大的方便。
