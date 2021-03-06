# Task 4 项目启动报告

## 项目简介

本小组的自主项目主题为目标检测。

针对一张图像，从浅入深的理解分为四步：

![](https://miro.medium.com/max/5856/1*Hz6t-tokG1niaUfmcysusw.jpeg)

1. 分类(Classification)，即对一张单物体的图像进行物体的分类。
2. 定位(Localization)，即在分类的基础上，划分出物体的出现位置。
3. 检测(Detection)，即在多物体的图像中做到不同物体的定位和分类工作。
4. 分割(Segmentation)，即在像素水平上解决检测任务。

目标检测(Object Detection)是计算机视觉领域的经典任务，旨在检测出一副图像中的若干个物体及其类别。

目标检测任务比简单的图像分类任务更加复杂，它不仅需要做到分类，更重要的是，需要在包含多个物体的一张图像中，划定各个物体的边界框(bounding boxes)，并且给出各个预测子任务的成功概率。

目标检测的相关研究成果已经落地投入应用，但本小组计划深入学习目标检测的技术原理，实现经典的目标检测模型，同时借助NNI调优模型，在经典模型的基础上做出新结果。

## 项目目标

本项目计划从技术原理的学习开始，逐步实现经典的one-stage目标检测项目，并通过使用NNI的功能，调优模型的特征提取器网络，争取训练出达到目前水平的模型。

## 项目规划

暂将项目规划分为如下五步：

1. 学习目标检测的技术原理，阅读相关论文，根据已有资料，整理技术细节。
2. 用PyTorch构建出目标检测模型代码。
3. 在Pascal VOC数据集上进行训练和测试，得到初始结果。
4. 使用NNI的超参调优和神经架构搜索等功能，对特征提取器等模型进行调整，得到多组调优后的实验结果。
5. 使用COCO等其他数据集，对比经典模型，评估实际应用能力。

## 实施方案

大体的实施方案是将NNI运用于特征提取器的结构优化和全局参数选择中，在经典模型的基础上提升模型实际性能表现。

## 目前进展

目前正在进行目标检测技术原理的学习，后续的实现代码将会保存在个人仓库中。
