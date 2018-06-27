---
layout:     post
title:      A Two-Stage Contrast Enhancement Algorithm for Digital Images
subtitle:   Image enhancement
date:       2018-06-08
author:     QITINGSHE
header-img: img/post-bg-debug.png
catalog: true
tags:
    - post
    - DIP
---
# A Two-Stage Contrast Enhancement Algorithm for Digital Images


## 简介
在对人眼对比度的敏感性研究表明，对数曲线服从人类感知中的韦伯定律，本文对HDRI和LDRI提出了一种基于对数曲线的局部对比度增强算法，该算法可以根据局部信息自适应第改变曲率。对细节增强的程度控制，本文给出了两个参数来协调。另外还提出了two-stage流程来解决halo artifact问题。


## 关注点
- Local contrast enhancement
- The effect of halo artifact


## Logarithmic Curve
该曲线用于图像对比度增强
$$
v(x,y)=\frac{\log(w(x,y)*(\beta-1)+1)}{\log(\beta)}
$$

$w(x,y)$是增强前的亮度级别，$v(x,y)$ 是增强后的亮度级别，$\beta$是对数曲线的一个参数。该曲线是对全局图像的对比度增强，因为每个像素的$\beta$都一样


## beta映射
为了完成局部细节对比度的增强，我们需要建立一个beta映射，为每个像素值给定一个beta
建立beta映射需要两步，在这两步之前我们先要得到边缘信息，这里文中使用sobel算子得到边缘信息。


### 第一步
对边缘像素的$\beta$暂不做处理，对每一个非边缘像素，我们使用一个mask，先计算mask内的均值，为了防止halo artifact现象，mask中一些像素不会参与均值计算，我们设置一个阈值来区分这些像素
$$
\beta = \left\{
\begin{array}{cl}
10*\frac{w(x,y)-mean}{w(x,y)}+2 & \text{if } w(x,y)\geq mean \text { and } Sobel(x,y) <th,\\
2 & \text{if } w(x,y)< mean \text { and } Sobel(x,y) <th,\\
undefine & \text{otherwise}.
\end{array} \right.
$$

其中$mean$是mask中有效像素的均值，$th$决定哪些像素是有效的
该公式表示了人类的视觉系统，在不同背景下，即便相同的亮度也会造成不同的感觉


![](https://github.com/Qitingshe/Qitingshe.github.io/raw/master/pic/center.png)
从这幅图可以看到，，$\beta$值与不同局部均值的关系，越暗的背景，人们观察到的前景越亮


### 第二步
确定第一步未处理的$\beta$
当像素的$\beta$值是undefinde，我们将在该点为中心的mask中找到与该点亮度最接近的像素，以该点的$\beta$作为中心点的$\beta$值。这样我们的beta映射就完成了，然后就可以按之前给出的公式完成局部对比度增强


**valley region**
在计算$\beta$的时候我们将一部分$\beta$值设置为2,这些像素在局部区域是低谷部分，为了增强这部分，我们将亮度反转，然后再次使用之前的第一第二步来计算$\beta$，这样就可以计算出Valley region的$\beta$


## For HDRI
对HDR图像而言我们还需要做Tone mapping，本文借鉴了Monobe提出的tone mapping算法，本文对其进行了一些修改以符合我们的要求，下面给出了Monobe的公式
$$
v(x,y)=Tone(w(x,y))*E(w(x,y),w_{avg}(x,y))
$$

$$
E(w(x,y),w_{avg}(x,y))=\bigg(\frac{w(x,y)}{w_{avg}(x,y)}\bigg)^{\alpha\big(1-\frac{w(x,y)}{Tone(w(x,y))}*\frac{dTone(w(x,y))}{dw(x,y)}\big)}
$$

其中最重要的参数是$\alpha$，它可以调节对比度


在Monobe的方法中，$\alpha$定义如下
$$
\alpha = \left\{
\begin{array}{cl}
0.25 &\text{if }w(x,y)>w_{avg}(x,y)\\
1.75 &\text{otherwise}
\end{array} \right.
$$
如果中心点的亮度值比周围的要高，$\alpha$设为0.25,这样可以避免过饱和，否则$\alpha$=1.75 ，以补偿视觉对比度退化的问题，这样可以做到保持对比度的效果，但本文不仅需要做到保持对比度，还要对其进行增强，所以对$\alpha$的取值做了一些调整


## 更新的$\alpha$计算公式
$$
\alpha = \left\{
\begin{array}{cl}
1+(1-kweight)*0.75 &\text{if }w(x,y)>w_{avg}(x,y)\\
kweight &\text{otherwise}
\end{array} 
\right.
$$

$$
kweight=\frac{1}{exp(Sobel(x,y))}
$$

$kweight$用来定义像素值是否在平滑区域，如果sobel算子的值接近0,意味着像素在平滑区域，为了避免过度增强，$\alpha$设置的接近1,这样局部对比度就会保持，不会增强，相反，在边缘区域，$\alpha$接近1.75,局部对比度就会增强


## 最终的公式
$$
v(x,y)=\frac{\log(Tone(w(x,y))*(\beta-1)+1)}{\log(\beta)}*E(w(x,y),w_{avg}(x,y))
$$

其中

$$
E(w(x,y),w_{avg}(x,y))=\bigg(\frac{w(x,y)}{w_{avg}(x,y)}^{\alpha\big(1-\frac{w(x,y)}{Tone(w(x,y))}*\frac{dTone(w(x,y))}{dw(x,y)}\big)}\bigg)
$$


**流程图**
![](https://github.com/Qitingshe/Qitingshe.github.io/raw/master/pic/flowchart.png)


## 优点和总结
- Two-stage 步骤降低了人为造成的光晕效果(Halo Artifact)
- 在压缩动态范围时，增强了图像的细节信息
- 对数曲率可以由局部信息适应性地调整
- 本文算法在LDRI上效果也很不错

本文方法有两个参数，$\alpha$用于在Tone Mapping时决定细节增强的程度，$\beta$用于在Tone Mapping后决定增强的级别，通过灵活的调整，输出图像可以保留诸多细节信息
