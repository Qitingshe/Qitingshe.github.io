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

### 扩散过程

给定一个从真实数据分布中采样的数据点$\mathbf{x}_0\sim q(\mathbf{x})$，在$T$步中逐渐给数据样本添加少量的高斯噪声，生成带噪声的样本序列$\mathbf{x}_1,...,\mathbf{x}_T$

$$
q\left(\mathbf{x}_t \mid \mathbf{x}_{t-1}\right)=\mathcal{N}\left(\mathbf{x}_t ; \sqrt{1-\beta_t} \mathbf{x}_{t-1}, \beta_t \mathbf{I}\right) \quad q\left(\mathbf{x}_{1: T} \mid \mathbf{x}_0\right)=\prod_{t=1}^T q\left(\mathbf{x}_t \mid \mathbf{x}_{t-1}\right)
$$

其中$\left\{\beta_t\right\}_{t=1}^T$表示每步添加的噪声的方差，取值区间为$0\sim 1$，称为variance schedule或noise schedule，通常$t$越大方差越大，即$\beta_1<\beta_2<...<\beta_T$

上述过程有一个重要性质：可以直接基于原始数据$\mathbf{x}_0$来对任意$t$步的$\mathbf{x}_t$进行采样
令$\alpha_t=1-\beta_t$，且$\bar{\alpha}_t=\prod_{i=1}^t\alpha_i$，可得

$$
\begin{array}{rlr}
\mathbf{x}_t & =\sqrt{\alpha_t} \mathbf{x}_{t-1}+\sqrt{1-\alpha_t} \epsilon_{t-1} \quad \quad\quad\quad\quad; \text { where } \boldsymbol{\epsilon}_{t-1}, \boldsymbol{\epsilon}_{t-2}, \cdots \sim \mathcal{N}(\mathbf{0}, \mathbf{I}) \\
& =\sqrt{\alpha_t \alpha_{t-1}} \mathbf{x}_{t-2}+\sqrt{1-\alpha_t \alpha_{t-1}} \overline{\boldsymbol{\epsilon}}_{t-2} \quad ; \text { where } \overline{\boldsymbol{\epsilon}}_{t-2} \text { merges two Gaussians }\left({ }^*\right) . \\
& =\ldots \\
& =\sqrt{\bar{\alpha}_t} \mathbf{x}_0+\sqrt{1-\bar{\alpha}_t}\epsilon \\
q\left(\mathbf{x}_t \mid \mathbf{x}_0\right) & =\mathcal{N}\left(\mathbf{x}_t ; \sqrt{\bar{\alpha}_t} \mathbf{x}_0,\left(1-\bar{\alpha}_t\right) \mathbf{I}\right)
\end{array}
$$

这样，只要设定好$\beta_t$的取值，就可以快速得到任何第$t$步的扩散结果，即

$$
\mathbf{x}_{\mathbf{t}}\left(\mathbf{x}_{\mathbf{0}}, \epsilon\right)=\sqrt{\bar{\alpha}_t} \mathbf{x}_0+\sqrt{1-\bar{\alpha}_t} \epsilon \quad \text { where } \epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})
$$

DDPM论文3.2节所提到的算法1就是基于该公式得到的
![](https://github.com/Qitingshe/Qitingshe.github.io/raw/master/_posts/assets/diffusion2.png)

### 反向过程

反向过程就是上述扩散过程的逆过程，即要构建$q(\mathbf{x}_{t-1}\mid \mathbf{x}_t)$，这样我们就可以从随机噪声$\mathbf{x}_T\sim \mathcal{N}(\mathbf{0},\mathbf{I})$中重建真实数据样本——生成图片了。

但是想要估计出$q(\mathbf{x}_{t-1}\mid \mathbf{x}_t)$并不容易，因为这需要全量数据集，因此我们需要学习一个模型$p_\theta$来**近似之前的条件概率分布**，这样就可以执行之前说的反向过程了

$$
p_\theta\left(\mathbf{x}_{0: T}\right)=p\left(\mathbf{x}_T\right) \prod_{t=1}^T p_\theta\left(\mathbf{x}_{t-1} \mid \mathbf{x}_t\right) \quad p_\theta\left(\mathbf{x}_{t-1} \mid \mathbf{x}_t\right)=\mathcal{N}\left(\mathbf{x}_{t-1} ; \boldsymbol{\mu}_\theta\left(\mathbf{x}_t, t\right), \mathbf{\Sigma}_\theta\left(\mathbf{x}_t, t\right)\right)
$$

其中，$p(\mathbf{x}_T)=\mathcal{N}(\mathbf{x}_T;\mathbf{0},\mathbf{I})$，
$p_\theta(\mathbf{x}_{t-1}\mid \mathbf{x}_t)$是一个参数化的高斯分布，其均值和方差为$\boldsymbol{\mu}_\theta\left(\mathbf{x}_t, t\right)$和$\mathbf{\Sigma}_\theta\left(\mathbf{x}_t, t\right)$

建模成功后，就要考虑如何获得真实的条件分布了，我们无法直接处理$q(\mathbf{x}_{t-1}\mid \mathbf{x}_t)$，因为不知道需要恢复到哪个样本，所以需要加上$\mathbf{x}_0$的后验分布$q(\mathbf{x}_{t-1}\mid \mathbf{x}_t,\mathbf{x}_0)$

根据贝叶斯公式，得到

$$
q\left(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0\right)=q\left(\mathbf{x}_t \mid \mathbf{x}_{t-1}, \mathbf{x}_0\right) \frac{q\left(\mathbf{x}_{t-1} \mid \mathbf{x}_0\right)}{q\left(\mathbf{x}_t \mid \mathbf{x}_0\right)}
$$

其中

$$q\left(\mathbf{x}_t \mid \mathbf{x}_{t-1}, \mathbf{x}_0\right)=q\left(\mathbf{x}_t \mid \mathbf{x}_{t-1}\right)=\mathcal{N}\left(\mathbf{x}_t ; \sqrt{1-\beta_t} \mathbf{x}_{t-1}, \beta_t \mathbf{I}\right)$$

这里的$\mathbf{x}_0$是一个多余条件，现在可以发现等式右边都是扩散模型的扩散过程中的某一步，这个是可以获得的。

经过推导可以得到后验概率分布$q\left(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0\right)$的均值和方差

$$
\bar{\beta}_t=\frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\cdot \beta_t
$$

$$
\begin{array}{rlr}
\boldsymbol{\mu}_t \left(\mathbf{x}_t, \mathbf{x}_0\right) &=\frac{\sqrt{\alpha_t}\left(1-\bar{\alpha}_{t-1}\right)}{1-\bar{\alpha}_t} \mathbf{x}_t+\frac{\sqrt{\bar{\alpha}_{t-1}} \beta_t}{1-\bar{\alpha}_t} \mathbf{x}_0 \\
&=\frac{1}{\sqrt{\alpha_t}}\left(\mathbf{x}_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon_t \right)
\end{array}
$$

可以发现方差是一个定量（扩散过程参数固定），而均值是一个依赖$\mathbf{x}_t$的函数
具体推导过程参考[What are Diffusion Models?](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)

上面这个反向过程的公式推导很重要，是扩散模型能够重建图像的理论依据，这样就得到DDPM论文所述的算法2
![](https://github.com/Qitingshe/Qitingshe.github.io/raw/master/_posts/assets/diffusion3.png)

上述扩散过程和反向过程可以看作是有$T$个隐变量的VAE模型，我们可以利用变分下限来构建优化目标

$$
\begin{aligned}
-\log p_\theta\left(\mathbf{x}_0\right) & \leq-\log p_\theta\left(\mathbf{x}_0\right)+D_{\mathrm{KL}}\left(q\left(\mathbf{x}_{1: T} \mid \mathbf{x}_0\right) \| p_\theta\left(\mathbf{x}_{1: T} \mid \mathbf{x}_0\right)\right) \\
& =-\log p_\theta\left(\mathbf{x}_0\right)+\mathbb{E}_{\mathbf{x}_{1: T} \sim\left(\mathbf{x}_{1: T} \mid \mathbf{x}_0\right)}\left[\log \frac{q\left(\mathbf{x}_{1: T} \mid \mathbf{x}_0\right)}{p_\theta\left(\mathbf{x}_{0: T}\right) / p_\theta\left(\mathbf{x}_0\right)}\right] \\
& =-\log p_\theta\left(\mathbf{x}_0\right)+\mathbb{E}_q\left[\log \frac{q\left(\mathbf{x}_{1: T} \mid \mathbf{x}_0\right)}{p_\theta\left(\mathbf{x}_{0: T}\right)}+\log p_\theta\left(\mathbf{x}_0\right)\right] \\
& =\mathbb{E}_q\left[\log \frac{q\left(\mathbf{x}_{1: T} \mid \mathbf{x}_0\right)}{p_\theta\left(\mathbf{x}_{0: T}\right)}\right] \\
\text { Let } L_{\mathrm{VLB}} & =\mathbb{E}_{q\left(\mathbf{x}_{0 T T}\right)}\left[\log \frac{q\left(\mathbf{x}_{1: T} \mid \mathbf{x}_0\right)}{p_\theta\left(\mathbf{x}_{0: T}\right)}\right] \geq-\mathbb{E}_{q\left(\mathbf{x}_0\right)} \log p_\theta\left(\mathbf{x}_0\right)
\end{aligned}
$$

进一步对训练目标分解得到

$$
\begin{aligned}
L_{\mathrm{VLB}} &=\mathbb{E}_q[\underbrace{D_{\mathrm{KL}}\left(q\left(\mathbf{x}_T \mid \mathbf{x}_0\right) \| p_\theta\left(\mathbf{x}_T\right)\right)}_{L_T}+\sum_{t=2}^T \underbrace{D_{\mathrm{KL}}\left(q\left(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0\right) \| p_\theta\left(\mathbf{x}_{t-1} \mid \mathbf{x}_t\right)\right)}_{L_{t-1}}\underbrace{-\log p_\theta\left(\mathbf{x}_0 \mid \mathbf{x}_1\right)}_{L_0}] \\
&  =L_T+L_{T-1}+\cdots+L_0 
\end{aligned}
$$

其中

$$
\begin{aligned}
\text { where } L_T & =D_{\mathrm{KL}}\left(q\left(\mathbf{x}_T \mid \mathbf{x}_0\right) \| p_\theta\left(\mathbf{x}_T\right)\right) \\
L_t & =D_{\mathrm{KL}}\left(q\left(\mathbf{x}_t \mid \mathbf{x}_{t+1}, \mathbf{x}_0\right) \| p_\theta\left(\mathbf{x}_t \mid \mathbf{x}_{t+1}\right)\right) \text { for } 1 \leq t \leq T-1 \\
L_0 & =-\log p_\theta\left(\mathbf{x}_0 \mid \mathbf{x}_1\right)
\end{aligned}
$$

每个KL散度（除了$L_0$）都表示两个高斯分布的相似度，训练阶段$L_T$可以被忽略，因为$p$没有可学习参数，而$x_T$是一个高斯随机噪声，$L_0$单独采用一个解码器进行图像重建。

之前定义了$p_{\theta}(\mathbf{x}_{t-1}\mid \mathbf{x}_t)$为一个参数化的高斯分布$\mathcal{N}\left(\mathbf{x}_{t-1} ; \boldsymbol{\mu}_\theta\left(\mathbf{x}_t, t\right), \mathbf{\Sigma}_\theta\left(\mathbf{x}_t, t\right)\right)$
对于两个高斯分布的KL散度，其计算公式为

$$
\mathrm{KL}\left(p_1 \| p_2\right)=\frac{1}{2}\left(\operatorname{tr}\left(\boldsymbol{\Sigma}_2^{-1} \boldsymbol{\Sigma}_1\right)+\left(\boldsymbol{\mu}_2-\boldsymbol{\mu}_1\right)^{\top} \boldsymbol{\Sigma}_2^{-1}\left(\boldsymbol{\mu}_2-\boldsymbol{\mu}_1\right)-n+\log \frac{\operatorname{det}\left(\boldsymbol{\Sigma}_{\mathbf{2}}\right)}{\operatorname{det}\left(\boldsymbol{\Sigma}_{\mathbf{1}}\right)}\right)
$$

在DDPM中，对该模型做了进一步简化，采用固定方差$\boldsymbol{\Sigma}_\theta\left(\mathbf{x}_t, t\right)=\sigma_t^2 \mathbf{I}$
就有

$$
\begin{aligned}
D_{\mathrm{KL}}\left(q\left(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0\right) \| p_\theta\left(\mathbf{x}_{t-1} \mid \mathbf{x}_t\right)\right) & =D_{\mathrm{KL}}\left(\mathcal{N}\left(\mathbf{x}_{t-1} ; \tilde{\boldsymbol{\mu}}\left(\mathbf{x}_t, \mathbf{x}_0\right), \sigma_t^2 \mathbf{I}\right) \| \mathcal{N}\left(\mathbf{x}_{t-1} ; \boldsymbol{\mu}_\theta\left(\mathbf{x}_t, t\right), \sigma_t^2 \mathbf{I}\right)\right) \\
& =\frac{1}{2}\left(n+\frac{1}{\sigma_t^2}\left\|\tilde{\boldsymbol{\mu}}_t\left(\mathbf{x}_t, \mathbf{x}_0\right)-\boldsymbol{\mu}_\theta\left(\mathbf{x}_t, t\right)\right\|^2-n+\log 1\right) \\
& =\frac{1}{2 \sigma_t^2}\left\|\tilde{\boldsymbol{\mu}}_t\left(\mathbf{x}_t, \mathbf{x}_0\right)-\boldsymbol{\mu}_\theta\left(\mathbf{x}_t, t\right)\right\|^2
\end{aligned}
$$

那么优化目标$L_{t-1}$就变成

$$
L_{t-1}=\mathbb{E}_{q\left(\mathbf{x}_t \mid \mathbf{x}_0\right)}\left[\frac{1}{2 \sigma_t^2}\left\|\tilde{\boldsymbol{\mu}}_t\left(\mathbf{x}_t, \mathbf{x}_0\right)-\boldsymbol{\mu}_\theta\left(\mathbf{x}_t, t\right)\right\|^2\right]
$$

即，希望网络学习到的均值$\mathbf{\mu}_{\theta}(\mathbf{x}_t,t)$和后验分布的均值$\mathbf{\bar{\mu}}(\mathbf{x}_t,\mathbf{x}_0)$一致。

不过DDPM发现预测均值并不好，根据

$$
\mathbf{x}_{\mathbf{t}}\left(\mathbf{x}_{\mathbf{0}}, \epsilon\right)=\sqrt{\bar{\alpha}_t} \mathbf{x}_0+\sqrt{1-\bar{\alpha}_t} \epsilon \quad \text { where } \epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})
$$

可将上述优化目标简化为

$$
L_{t-1}^{\text {simple }}=\mathbb{E}_{\mathbf{x}_0, \epsilon \sim \mathcal{N}(0, \mathbf{I})}\left[\left\|\epsilon-\epsilon_\theta\left(\sqrt{\bar{\alpha}_t} \mathbf{x}_0+\sqrt{1-\bar{\alpha}_t} \epsilon, t\right)\right\|^2\right]
$$

这样就由原来**预测均值转换成预测噪音**$\epsilon$

对应优化目标变为**随机生成噪音数据**$\epsilon$与**模型预测的噪音**$\epsilon_\theta\left(\sqrt{\bar{\alpha}_t} \mathbf{x}_0+\sqrt{1-\bar{\alpha}_t} \epsilon, t\right)$的L2损失。

## 总结

Diffusion就是构建一个模型来预测噪声，训练阶段该模型输入是一张带噪声的图片，希望模型经过训练可以估计出图片分布，从而将图片与噪声能够剥离开来；当模型训练完成后，给定一个噪声，可以利用Diffusion的反向过程对该噪声进行去噪操作，每次去噪之后生成的结果在分布上都会更加趋近于真实图像分布，达到了生成图像的目的。

最后，做个不成熟的联想，以前有人说努力也不一定能成功，现在可以用更专业的工具反思这句话了，虽然努力不一定能成功，但是努力一定会改变概率分布，让自己所处的概率分布更接近成功的概率分布。这就是Diffusion模型能够效果这么好的底层逻辑。
