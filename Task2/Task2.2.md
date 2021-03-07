# Task2.2

## 超参调优

NNI支持通过配置搜索空间自定义搜索结构，不仅能够运用高效率的算法进行自动超参调优，更能够在多个模型与超参中选择出性能更优的组合，从而提高模型准确率。

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


## Classic NAS

通过上一步得到的模型与参数的组合，我们尝试在搜索空间上定义随机结构，测试模型的性能。

神经网络的随机结构可以借助NNI的经典NAS算法来实现，tuner在NNI的example中可以找到。

编写随机结构的搜索空间时，可以使用`nni.nas.pytorch.mutables`中的`LayerChoice`和`Inputchoice`来进行实现。

这两种定义待选连接的方式都很方便。`LayerChoice`在代码使用上可视作与其他普通神经网络等同，而要实现`InputChoice`所代表的跳过连接，则与`InputChoice`的输出进行concat操作即可。

这里定义了MobileNet的随机结构，代码如下：

```python

```

### 最终结果



## One-Shot NAS

上一步的经典NAS算法对原始模型的改动较小，也得不到较大的优化效果，准确率不仅没有升高，反而还更低了。

在这一步，我们尝试One-Shot NAS，通过定义DARTS的搜索空间，大幅度改变原有模型，尝试真正意义上提高预测的精确度至97%以上。

### DARTS原理简要分析

DARTS全称Differentiable Architecture Search，是NAS中著名的算法之一。该算法的特色是将若干个待搜索的架构从互不关联的“黑箱优化”问题变成可松弛的连续优化问题，通过梯度下降来进行更新。

由于需求只是构造MobileNet的搜索空间，这里只对CNN的DARTS进行分析。

首先，如果将一个状态看作一个节点，把一种操作看作一条边，那么CNN的网络模型就可以抽象成一个有向无环图（DAG）。

而在我们进行搜索的过程中，两点之间其实包含有“重边”。这些“重边”虽然两端节点相同，但各代表着不同的操作。我们需要做的，就是在这些待选边中找出整体最适合的一条边来成为DAG的一部分，实现架构的搜索。

我们首先给cell下定义。一个cell是一个包含了$N$个节点的有向无环图。其中编号为$i$的节点$x^{(i)}$代表着特征所存在着的状态，而从$i$到$j$的一条有向边就代表着一种操作，这种操作记为$o^{(i,j)}$。

接下来定义cell的输入与输出。一个cell会有两个输入，而只会有一个输出。这个cell的输出是将所有前面节点的操作concat起来的结果。用公式写出来就是：

$$x^{(j)} = \int_{i<j} o^{(i, j)}(x^{(i)})$$

这里$o^{(i,j)}(x)$代表着将$x$所代表的状态经过$(i,j)$这条有向边所代表的操作后所得到的新状态。正如直观感觉一般，也就是可以抽象成一个函数。

一条边所代表的，可以是一个池化层，可以是标准的conv+bn，也可以是skipconnect等其他的子模型。

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

DARTS算法将动辄耗费上千个GPU天的神经网络架构搜索缩短至1.5或4个GPU天，使得NAS应用的门槛和成本大幅度降低。

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

调用`nni.algorithms.nas.pytorch.darts.DartsTrainer`可进行DARTS的架构搜索。再通过`retrain.py`再次进行训练，最终在第4xx个epoch达到了97%的准确率，并最终达到了98%。

由于DARTS的侯选边在MobileNet中并没有出现，所以最终生成的模型相对变化较大，但提升也十分明显。
