---
layout:     post
title:      Diffusion模型介绍
subtitle:   概率生成模型
date:       2023-03-22
author:     QITINGSHE
header-img: img/post-bg-debug.png
catalog: true
tags:
    - Conditional Generative Model
    - Probabilistic Model
---

最近在阅读Diffusion模型，发现这真是一个非常有趣的方向，利用预测概率分布的方式进行生成任务，相比较以往确定性任务而言，这种方式天然具有随机性，有着天马行空的想象力，当我们需要进行更加精确的生成时，只需要增加条件约束就可以了，比如增加文本约束，线稿信息约束等。通过添加约束不断缩小其预测空间，但是无论增加多少约束，其随机性总是无法消除掉，从某种程度上讲，我们可以对它生成的结果抱有期待。谁又能抵御对未知的好奇心呢？

## 一段话描述Diffusion模型

Diffusion模型又称为扩散模型，它涉及到几个概念，扩散过程、反向过程。

**扩散过程**：对一张图像逐步添加高斯噪声得到$\mathbf{x}_t$，经过$T$步之后，得到$\mathbf{x}_T$是一个符合标高斯分布的噪声(你选择其他分布的噪声也不是不可以，这里不是强制约束)
![](https://github.com/Qitingshe/Qitingshe.github.io/raw/master/_posts/assets/diffusion1.png)
**反向过程**：扩散过程是数据噪声化，那么反向过程就是其逆过程，一个去噪的过程，从一个随机高斯噪声逐渐去噪最终生成一张图片，所以反向过程也是一个数据**生成过程**

Diffusion模型就是构建一个预测噪声的模型$M$，假定从标准高斯分布中采样的噪声为$\epsilon$，将这个噪声按照一定方法添加到图像数据$\mathbf{x}_0$中，得到$\mathbf{x}_t$，将$\mathbf{x}_t$输入到模型$M(\mathbf{x}_t)$，模型输出噪声为$\^\epsilon$，**期望这个输出噪声所在的分布与$\epsilon$所对应的标准高斯分布一致**，可以通过KL散度来计算分布的相似度（可以转换为求两个噪声的L2损失）。

从上面描述我们可以发现，模型的任务其实是希望从输入$\mathbf{x}_t$中恢复噪声$\epsilon$，但是模型并不知道原始图像$\mathbf{x}_0$具体是什么，这个时候模型就需要在训练过程中，不断**学习整个训练集的数据分布**，借此来估计原始输入$\mathbf{x}_0$，也就是隐式地学习了图像数据的构建信息。

在Inference阶段，其实是一个$T$次循环的**去噪过程**，首先从标准高斯分布中采样一个噪声$\mathbf{x}_T$，然后利用这个噪声$\mathbf{x}_T$计算出下一个采样$\mathbf{x}_{T-1}$的分布，从中采样出$\mathbf{x}_{T-1}$，然后送入模型预测一个噪声$\epsilon_{T-1}$，利用这个噪声对$\mathbf{x}_{T-1}$进行去噪，生成$\mathbf{x}_{T-2}$，$\mathbf{x}_{T-2}$理论上讲会更加**趋近于图像数据的分布**。
以此循环，最后生成$\mathbf{x}_0$，即完成图像的生成。这就是目前Diffusion模型的工作流程。

## 原理分析
