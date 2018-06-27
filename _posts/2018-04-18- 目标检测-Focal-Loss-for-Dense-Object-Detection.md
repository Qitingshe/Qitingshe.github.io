---
layout:     post
title:      Focal Loss for Dense Object Detection
subtitle:   RetinaNet
date:       2018-04-18
author:     QITINGSHE
header-img: img/post-bg-debug.png
catalog: true
tags:
    - Object Detection
    - DeepLearning
---

# Why: Focal Loss
当前主流检测算法：
- One-stage: YOLO and SSD
- Two-stage: RCNN 系列
One-stage的特点：输出一个稠密的proposal，然后丢进分类器中，直接进行类别分类。方法简单，速度快，但精度不高。

Two-stage的特点：分类器在一个稀疏的候选目标中进行分类(背景和对应类别)，通过前面的proposal过程实现的：Seletive or RPN —— 稀疏集合内做分类。

稠密分类精度不高的原因：
核心问题：稠密proposal中前景和背景极度不平衡，以YOLO为例，在PASCAL VOC数据集中，每张图片的目标可能只有几个，但YOLO V2最后一层输出是13×13×5,即845个目标，大量(简单易区分)的负样本在loss中占据了很大比重，使得有用的loss不能回传。

由此提出了Focal loss，给易分类的简单样本较小的权重，给不易区分的难样本更大的权重。作者提出新的one-stage的检测器RetinaNet，达到了速度和精度很好的trade-off
$$
FL(p_t)=-(1-p_t)^\gamma log(p_t)
$$

# 物体检测的两种主流方法

深度学习之前，经典的物体检测方法为滑动窗口，并使用人工设计的特征。HoG和DPM等方法是其中比较有名的。

R-CNN系的方法是目前精度最高的方法。在R-CNN方法中，正负类别不平衡问题通过前面的proposal解决了。通过EdgeBoxes, Selective Search, DeepMask, RPN等方法过滤掉大多数的背景，实际传给后续网络的proposal的数量是比较少的（1-2k）。

在YOLO ，SSD等方法中，需要直接对feature map的大量proposal（1000k）进行检测，而且这些feature map上重叠。大量的负样本带来两个问题：

- 过于简单，有效信息过少，使得训练效率低
- 简单的负样本在训练过程中压倒性优势，使得模型发生退化。

在Faster-RCNN中，Huber  Loss被用来降低outlier的影响（较大的样本是难例，传回来的梯度做了clipping，也只能是1）。而Focal Loss 是对inner中简单的那些样本对loss的贡献进行限制。即使这些简单样本数量很多，也不让他们在训练中占到优势

# Focal Loss

Focal Loss 从交叉熵损失而来，二分类的交叉熵损失如下：

$$
\text{CE}(p, y) = \begin{cases}-\log(p) \quad &\text{if}\quad y = 1\\ -\log(1-p) &\text{otherwise}\end{cases}
$$

对应的，多分类的交叉熵损失是：

$$
\text{CE}(p,y)=-log(p_y)
$$

如下图所示，蓝色线为交叉熵损失函数随着$p_t$ 的变化曲线。当概率大于0.5,即认为是易分类的简单样本时，值仍然较大。这样，很多简单样本累加起来，就很可能盖住那些稀少的不易正确分类的类别。

![2](https://github.com/Qitingshe/Qitingshe.github.io/raw/master/pic/focal_loss_vs_ce_loss.jpg)

为了改善类别样本分布不均衡的问题，已经有人提出使用加上权值的交叉熵损失，如下（即用参数$\alpha_t$ 来平衡，这组参数可以是超参数，也可以由类别的比例倒数决定）。作者将其作为比较的baseline。

$$
\text{CE}(p)=-\alpha_t log(p_t)
$$

作者提出的则是一个自适应调节的权重，即Focal Loss，定义如下。由上图可以看到$\gamma$ 取值不同的时候的函数值变化。作者发现，$\gamma=2$ 时能够获得最佳的效果提升。

$$
\text{FL}(p_t)=-(1-p_t)^\gamma log(p_t)
$$

在实际实验中，作者使用的是加权之后的Focal Loss，作者发现这样能够带来些许的性能提升。

# RetinaNet

基于ResNet 和 Feature Pyramid Net (FPN) 设计了一种新的oner-stage检测框架，命名为RetinaNet。



 





















# Link

[Focal Loss论文阅读 - Focal Loss for Dense Object Detection](https://xmfbit.github.io/2017/08/14/focal-loss-paper/)















