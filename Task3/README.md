# Task 3 进阶任务

## 特征工程简介

有这么一句话在业界广泛流传：

> 数据和特征决定了机器学习的上限，而模型和算法只是逼近这个上限而已。

数据是特征的来源，特征是给定算法下模型精确度的最大决定因素，可见提升特征质量意义重大。

特征工程(Feature Engineering)是机器学习的一个重要分支，指的是通过多种数据处理方法，从原始数据提取出若干个能优秀反映问题的特征，以提升最终算法与模型准确率的过程。

## 自动特征工程

自动特征工程是一种新技术，是机器学习发展的一大步。自动特征工程能够在降低时间成本的同时，生成更优秀的特征，从而构建出准确率更高的模型。

利用NNI的自动特征工程实现，我们通过简单调用函数便可实现特征工程的自动调优。

## 环境准备

- nni
- numpy
- lightgbm: 微软开源算法
- pandas: 基于python的数据分析强力工具
- sklearn: 集成了特征工程相关的常用函数

建议在conda环境下部署自动特征工程python环境。

此外，由于pandas版本更新，直接运行自带项目会报错，实际上只需修改`fe_util.py`中的`agg`参数类型即可，大致修改如下：

```diff
def aggregate(df, num_col, col, stat_list = AGGREGATE_TYPE):
-   agg_dict = {}
+   agg_list = []
    for i in stat_list:
-       agg_dict[('AGG_{}_{}_{}'.format(i, num_col, col)] = i
+       agg_list.append(('AGG_{}_{}_{}'.format(i, num_col, col), i))
-   agg_result = df.groupby([col])[num_col].agg(agg_dict)
+   agg.result = df.groupby([col])[num_col].agg(agg_list)
    r = left_merge(df, agg_result, on = [col])
    df = concat([df, r])
    return df
```

该修改已提交pull request至原项目。

## 配置文件

### 配置搜索空间

NNI的自动特征工程支持count、crosscount、aggregate等一阶与二阶特征运算，配置搜索空间时只需按json格式填写搜索范围。具体填写方法以项目示例搜索空间为例：

```json
{
    "count":[
        "C1","C2","C3","C4","C5","C6","C7","C8","C9","C10",
        "C11","C12","C13","C14","C15","C16","C17","C18","C19",
        "C20","C21","C22","C23","C24","C25","C26"
    ],
    "aggregate":[
        ["I9","I10","I11","I12"],
        [
            "C1","C2","C3","C4","C5","C6","C7","C8","C9","C10",
            "C11","C12","C13","C14","C15","C16","C17","C18","C19",
            "C20","C21","C22","C23","C24","C25","C26"
        ]
    ],
    "crosscount":[
        [
            "C1","C2","C3","C4","C5","C6","C7","C8","C9","C10",
            "C11","C12","C13","C14","C15","C16","C17","C18","C19",
            "C20","C21","C22","C23","C24","C25","C26"
        ],
        [
            "C1","C2","C3","C4","C5","C6","C7","C8","C9","C10",
            "C11","C12","C13","C14","C15","C16","C17","C18","C19",
            "C20","C21","C22","C23","C24","C25","C26"
        ]
    ]
}
```

### 导入tuner

导入自动特征工程的tuner时，需要在`config.yml`中的`tuner`项添加相关信息，其他地方正常填写即可。

```yaml
tuner:
  codeDir: .
  classFileName: autofe_tuner.py
  className: AutoFETuner
  classArgs:
    optimize_mode: maximize
```

## 代码

Tuner在生成的搜索空间中随机选取一定数量的feature组合，通过`nni.get_next_parameter()`的接口，以dict的形式返回给单次trial。经一系列处理后运行lightGBM算法，得到最终以AUC形式呈现的结果。

调用代码主体部分如下：

```python
# get parameter from tuner
RECEIVED_PARAMS = nni.get_next_parameter()
logger.info("Received params:\n", RECEIVED_PARAMS)

# get sample column from parameter
df = pd.read_csv(file_name)
if 'sample_feature' in RECEIVED_PARAMS.keys():
    sample_col = RECEIVED_PARAMS['sample_feature']
else:
    sample_col = []

# df: raw feaure + sample_feature
df = name2feature(df, sample_col, target_name)
feature_imp, val_score = lgb_model_train(df, _epoch=1000, target_name=target_name,id_index=id_index)

# report result to nni
nni.report_final_result({
    "default":val_score, 
    "feature_importance":feature_imp
})
```

## 项目示例运行结果

### overview

![](./images/task3_1.png)

### Top 10 trials

![](images/task3_2.png)

### part of sample features

![](images/task3_3.png)

### plot

![](images/task3_4.png)

### hyper parameter

![](images/task3_6.png)

### feature importance of top 1 trial

```
           feature_name  split  ...  split_percent  feature_score
5                    I6     43  ...       7.570423       0.108238
4                    I5     18  ...       3.169014       0.037422
29                  C17     17  ...       2.992958       0.037006
161  AGG_median_I11_C15     17  ...       2.992958       0.036113
48        count_C18_C23     18  ...       3.169014       0.030764
..                  ...    ...  ...            ...            ...
77      AGG_min_I11_C17      0  ...       0.000000       0.000000
76      AGG_min_I12_C20      0  ...       0.000000       0.000000
127       count_C19_C23      0  ...       0.000000       0.000000
128       count_C17_C22      0  ...       0.000000       0.000000
112        count_C26_C7      0  ...       0.000000       0.000000
```

## 使用NNI跑一跑heart数据集

[数据集地址](http://archive.ics.uci.edu/ml/datasets/Statlog+%28Heart%29)

## 结果

