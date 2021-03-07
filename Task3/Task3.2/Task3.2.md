# Task 3.2 进阶任务实验报告

## Task 3.2.1 表格型数据的进阶任务

### 电影票房预测：TMDB Box Office Prediction

#### 特征处理

首先，可以发现`belongs-to-collection`，`homepage`等特征存在缺失值，可根据具体数据类型使用不同策略填补缺失值，例如json数据填补`'[]'`，`runtime`用中位数填补等等。

其次，数据中存在异常值，需要人工删去异常数据或对异常数据进行填补。`budget`特征中，前25%的数据均是0，必然是异常数据，解决方案之一是用剩余`budget`数据的中位数来对其进行填补。

`poster-path`、`imdb-id`特征一个是图片链接，一个是id，对模型没有任何直接关系，直接删去。

在两个数据中，`status`特征99.8%的数据都是Released，那么这个特征根本不能提高分类精确度，因此可以直接删去。

`release-date`为常见的时间序列特征，通过`pd.DatetimeIndex`可以方便地提取年、月、日等新特征。

在直方图中可以观察出`revenue`和`budget`数据的分布不均匀，可以通过`np.log1p`进行平滑处理，这样的预测结果比较符合正态分布，预测准确度能够提高。最终预测的`log-revenue`通过`np.exp1m`可还原回正常数据。

`cast`，`crew`，`belongs-to-collection`等特征是以json格式存储的，对数据中json的解析，`ast.literal_eval`能比较方便地完成解析任务。[参考链接](https://www.kaggle.com/c/tmdb-box-office-prediction/discussion/80045)

#### 人工构造特征

基于先验知识，大制片厂参与制作，有明星演员，有明星导演的电影，票房一般会比较高。我们可以预先统计出演员、导演、制片厂的排名榜，然后通过这些关键要素的计数多少或存在与否构建新特征。

很多特征中，特征中所包含的元素越多，票房有比较大的几率倾向于越高，如`spoken-languages`，`genres`等特征。我们便可以构造count类型的新特征。

更多详细特征工程内容可查看[该题目的jupyter notebook](tmdb-box-office-prediction.ipynb)。

最终提交结果的loss是2.11(656/1395)，误差可接受，但有待进一步优化。

#### 自动特征工程

#### 最终结果

### 旧金山犯罪分类：San Francisco Crime Classification

#### 原始特征

该数据的特征相对较少，特征的处理也相对比较方便。

#### 特征处理

`X`，`Y`两个特征中，在测试集发现经度为-120.5，纬度为90的异常特征，我们分别使用经纬度的中位数进行替换。替换后可对数据进行标准化。

#### 人工构造特征

对时间序列特征，提取出年、月、日、季度、小时、星期等子特征。并使用onehot编码展开。

发现地址中具有规律，可根据地址最后两个字母的不同，构造特征。

最终使用随机森林进行回归，准确率为27.5%左右，有待提高。

#### 使用NNI


#### 最终结果

使用LightGBM算法，在验证集上的结果是2.35，达到了top25%左右的水平。

在测试集上的结果是



### 土壤属性预测：Africa Soil Property Prediction Challenge

multi-label的回归任务其实可以拆分为多个单label的回归任务，通过多次构建模型进行回归来预测各个label的值。

#### 原始特征

原始特征均进行过标准化处理，除了`Depth`特征可以将字符串变化为01编码，其他特征无需进一步处理。

#### 手动选择建立模型

使用了sklearn中贝叶斯线性回归模型（BayesianRidge），通过做5次模型的回归，最终kaggle上的loss为0.46489，拟合效果很好。

#### 使用NNI AutoFE工具

由于原代码只支持二分类任务，这里为了实现multi-label的回归任务，使用了以`lightgbm.LGBMRegressor`为内置结构的`sklearn.multioutput.MultiOutputRegressor`。[参考链接](https://stackoverflow.com/questions/52648383/how-to-get-coefficients-and-feature-importances-from-multioutputregressor)

由于特征数较多，特征的搜索空间较难填写。


## Task 3.2.2 复杂型数据的探究任务

