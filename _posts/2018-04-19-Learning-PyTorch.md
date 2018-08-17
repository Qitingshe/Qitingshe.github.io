---
layout:     post
title:      model
subtitle:   subtitle
date:       2018-04-14
author:     QITINGSHE
header-img: img/post-bg-debug.png
catalog: true
tags:
    - post
    - DeepLearning
---

# Tensor

A PyTorch Tensor is conceptually identical to a **numpy array**: a Tensor is an n-dimensional array.

PyTorch do not know anything about deep learning or computational graphs or gradients; they are a genertic tool scientific computing.

However unlike numpy, PyTorch Tensors can utilize GPUs to accelerate their numeric computations. To run a PyTorch Tensor on GPU, you simply need to cast it to a new datatype.

# Variables and autograd

The **autograd** package in PyTorch provides function to compute the backward passes in neural networks. When using autograd, the forward pass of your network will define a **compiutational graph**; 

- nodes in the graph will be Tensors
- edges will be functions that produce output Tensors from input Tensors.


Backpropagating through this graph then allows you to easily compute gradients.

## In practice:

We wrap our PyTorch Tensors in **Variable** objects; a Variable represents a node in computational graph.

- If $x$ is a Variable 
- $x.data$ is a Tensor
- $x.grad$ is **another** Variable holding the gradient of x with respect to some scalar value(标量).

***Same API for  Variables as Tensor*** 

Almost any operation  that you can perform on a Tensor also works  on Variables
**The difference:** is that using Variables defines a computational graph, allowing you to automatically compute gradients.

```python
import torch
from torch.autograd import Variable

dtype = torch.FloatTensor
# dtype = torch.cuda.FloatTensor
# above run on GPU

N, D_in, H, D_out = 64, 1000, 100, 10

x = Variable(torch.randn(N, D_in).type(dtype), requires_grad=False)
y = Variable(torch.randn(N, D_out).type(dtype), requires_grad=False)

w1 = Variable(torch.randn(D_in, H).type(dtype), requires_grad=True)
w2 = Variable(torch.randn(H, D_out).type(dtype), requires_grad=True)

learning_rate = 1e-6
for t in range(500):
    y_pred = x.mm(w1).clamp(min=0).mm(w2)
    # Caculate the loss
    loss = (y_pred-y).pow(2).sum()
    print(t, loss.data[0])
    # Use autograd to compute the backward pass. 
    # This call will compute the gradient of loss with respect to all Variable with requires_grad=True.
    # After this call w1.grad amd w2.grad will be Variables holding the gradient of the loss with respect to w1 and w2 respectively.
    loss.backward()
    # Update weights
    # w1.grad and w2.grad are Variables and w1.grad.data and w2.grad.data are Tensors.
    w1.data -= learning_rate*w1.grad.data
    w2.data -= learning_rate*w2.grad.data
    w1.grad.data.zero_()
    w2.grad.data.zero_()

```



# Defining new autograd functions

Under the hood, each primitive autograd operator realizes two functions that operate on Tensor. 

- **forward function**: computes output Tensors from input Tensors.
- **backward function**: receives the gradient of the output Tensors with respect to some scalar value, and computes the gradient of the input Tensors with respect to that same scalar value.

In PyTorch we can easily define our own autograd operator by defining a subclass(子类) of $torch.autograd.Function$ and implementing the $forward$ and $backward$ functions. 

```python
import torch
from torch.autograd import Variable

class MyReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we receive a Tenser containing the input and return a Tensor containing the output. ctx is a context object that can be used to stash information for backward computation. You can cache arbitrary objeccts for use in the backward pass using the ctx.save_for_backward method.
        """
        ctx.save_for_backward(input)
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss with respect to the output, and we need to compute the gradient of the loss with respect to the input.
        """
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        return grad_input


dtype = torch.FloatTensor
# dtype = torch.cuda.FloatTensor
# above run on GPU

N, D_in, H, D_out = 64, 1000, 100, 10

x = Variable(torch.randn(N, D_in).type(dtype), requires_grad=False)
y = Variable(torch.randn(N, D_out).type(dtype), requires_grad=False)

w1 = Variable(torch.randn(D_in, H).type(dtype), requires_grad=True)
w2 = Variable(torch.randn(H, D_out).type(dtype), requires_grad=True)

learning_rate = 1e-6
for t in range(500):
    relu = MyReLU.apply

    y_pred = relu(x.mm(w1)).mm(w2)
    # Caculate the loss
    loss = (y_pred-y).pow(2).sum()
    print(t, loss.data[0])
    # Use autograd to compute the backward pass.
    # This call will compute the gradient of loss with respect to all Variable with requires_grad=True.
    # After this call w1.grad amd w2.grad will be Variables holding the gradient of the loss with respect to w1 and w2 respectively.
    loss.backward()
    # Update weights
    # w1.grad and w2.grad are Variables and w1.grad.data and w2.grad.data are Tensors.
    w1.data -= learning_rate*w1.grad.data
    w2.data -= learning_rate*w2.grad.data
    w1.grad.data.zero_()
    w2.grad.data.zero_()
```

# TensorFlow: Static Graphs

PyTorch autograd looks a lot like Tensorflow:

- Frameworks we  define a computational graph
- Use automatic differentitation to compute gradients.

Difference about computational graphs: 

- Tensorflow: **static** 
- PyTorch: **dynamic** 


Static graphs are nice because you can optimize the graph up front

In Tensorflow the act of updating the value of the weights is part of the computational graph; in PyTorch this happens outside the computational graph

# PyTorch: nn

Computational graphs and autograd are a very powerful paradigm for defining complex operators and automatcally taking derivatives; however for large neural networks raw autograd can be a bit too low-level. 

When building neural networks we frequently think of arranging the computation into layers, some of which have **learnable parameters** which will be optimized during learing.

In PyTorch the *nn* package serves this purpose. The *nn* package defines a set of **Modules**, which are roughly equivalent to neural network layers. A Mudule receives input Variables and computes output Variables, but may also hold internal state such as Variables containing learnable parameters. The *nn* package also dedfines a set of useful loss functions that are commonly used when training neural networks.

# PyTorch: optim

Up to this point we have updated the weights of our models by manually mutating the $.data$ member for Variables holding learnable parameters. 

The $optim$ package in PyTorch  abstracts the idea of an optimization  algorithm and provides implementations of commonly used optimization algorithms.

```python
import torch 
from torch.autograd import Variable

N,D_in,H,D_out=64,1000,100,10

x=Variable(torch.randn(N,D_in))
y=Variable(torch.randn(N,D_out),requires_grad=False)

model=torch.nn.Sequential(
    torch.nn.Linear(D_in,H),
    torch.nn.ReLU(),
    torch.nn.Linear(H,D_out),
)
loss_fn=torch.nn.MSELoss(size_average=False)
# Use the optim package to define an Optimizer that will update the weights of the model for us. In Adam: The first argument to the Adam constructor tells the optimizer which Variables it should opdate.
learning_rate=1e-4
optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate)

for t in range(500):
    # Forward pass: compute predicted y by passing x to the model.
    y_pred=model(x)
    
    loss=loss_fn(y_pred,y)
    print(t,loss.data[0])
	# Before the backward pass, use the optimizer object to zero all of the gradients for the variables it will update(learnable weights of the model). This is because by default, gradients are accumulated in buffers whenever .backward() is called. Checkout docs of torch.autograd.backward for more details.
    optimizer.zero_grad()
    loss.backward()
	# Calling the step function on an Optimezer makes an update to its parameters
    optimizer.step()
```

# PyTorch: Custom nn Modules

If you want to specify models that are more complex than a sequence of existing Modules; for these cases you can defiine your own Modules by subclassing $ nn.Module$ and defining a $ forward$ which receives input Variables and produces output Variables using other modules or other autograd operations on Variables.

```python
import torch 
from torch.autograd import Variable

class TwoLayerNet(torch.nn.Module):
    def __init__(self,D_in,H,D_out):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as member variables.
        """
        super(TwoLayerNet,self).__init__()
        self.linear1=torch.nn.Linear(D_in,H)
        self.linear2=torch.nn.Linear(H,D_out)

    def forward(self,x):
        """
        In the forward function we accept a Variable of input data and we must return a Variable of output data. We can use Modules defined in the constructor as well as arbitrary operators on Variables.
        """
        h_relu=self.linear1(x).clamp(min=0)
        y_pred=self.linear2(h_relu)
        return y_pred


N,D_in,H,D_out=64,1000,100,10

x=Variable(torch.randn(N,D_in))
y=Variable(torch.randn(N,D_out),requires_grad=False)

model=TwoLayerNet(D_in,H,D_out)

criterion=torch.nn.MSELoss(size_average=False)
optimizer=torch.optim.SGD(model.parameters(),lr=1e-4)

for t in range(500):
    y_pred=model(x)
    loss=criterion(y_pred,y)
    print(t,loss.data[0])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


```

# PyTorch: Control Flow + Weight Sharing

As  an example of dynamic graphs and weight sharing, we implement a very strange model: a fully-connected ReLU network that on each forward  pass chooses a random number between 1 and 4 and uses that many layers, reusing the same weights multiple times to compute the innermost hidden layers.

For this modle we can use normal Python flow control to implement the loop, and we can implement weight sharing among the innermost layers by simply reusing the same Module multiple times when dedining the forward pass.

```python
import torch
import random
from torch.autograd import Variable
import matplotlib.pyplot as plt


class DynamicNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(DynamicNet, self).__init__()
        self.input_linear = torch.nn.Linear(D_in, H)
        self.middle_linear = torch.nn.Linear(H, H)
        self.output_linear = torch.nn.Linear(H, D_out)

    def forward(self, x):
        h_relu = self.input_linear(x).clamp(min=0)
        for _ in range(random.randint(0, 3)):
            h_relu = self.middle_linear(h_relu).clamp(min=0)
        y_pred = self.output_linear(h_relu)
        return y_pred


N, D_in, H, D_out = 64, 1000, 100, 10

x = Variable(torch.randn(N, D_in))
y = Variable(torch.randn(N, D_out), requires_grad=False)

model = DynamicNet(D_in, H, D_out)

criterion = torch.nn.MSELoss(size_average=False)


optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
#optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
t_list = []
loss_list = []

for t in range(1500):
    y_pred = model(x)
    loss = criterion(y_pred, y)
    t_list.append(t)
   # print(t)
    loss_list.append(loss.data[0])
    print(t,loss.data[0])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

plt.plot(t_list, loss_list)
plt.xlabel("t")
plt.ylabel("mseloss")
plt.show()
```

# Training a classifier

You have seen how to define neural networks, compute loss and make updates to the weights of the network.

Now you might be thinking,

## What about data?

Generally, when you have to deal with image, text, audio or video data, you can use standard python packages that load data into a numpy array.Then you can convert this array into a $torch.*Tensor$.

- For images, packages such as Pillow, OpenCV are useful
- For audio, packages such as scipy and librosa
- For text, either raw Python or Cython based loading, or NLTK and SpaCy are useful 

Specifically for vision, we have created a package called $ torchvision $, that has  data loaders for common datasets such as Imagenet, CIFAR10,MNIST, etc. and data transformers for images, viz, $ torchvision.datasets$ and $ torch.utils.data.DataLoader$.

## Training an image classifier

We will do the following steps in order:

1. Load and normalizing the CIFAR10 training and test datasets using $torchvision$ 
2. Define a Convolution Neural Network
3. Define a loss function 
4. Train the network on the training data
5. Test the network on the test data




---

# PyTorch

- 2002年：torch——lua
- 2017年：pytorch——python，对Tensor之上的所有模块全部重构，新增自动求导，动态图框架

## 深度学习框架

- Theano

  2008, 一个Python库，用于*定义、优化和计算* 数学表达式，可用GPU加速，结合了计算机代数系统CAS和优化编译器，可定制C语言代码。Theano缺点明显，Keras是基于Theana基础之上的的第三方框架，它提供了更好的封装接口。Theano在2017年被宣布终止开发

  贡献：——计算图为框架核心

  ​	   ——GPU加速计算

- TensorFlow

  2015, 基于计算图实现自动微分求导，使用数据流图进行数值计算，节点为数学运算，边为节点之间传递的多维数组——Tensor。TensorFlow API 支持众多语言和系统，使用C++ Eigen库，可在ARM架构上编译和优化

  缺点：

  - 过于复杂的系统设计

  - 频繁变动的API

  - API接口设计过于晦涩难懂

    ——创建了图、会话、命名空间、PlaceHolder等诸多抽象概念

    ——同一功能有多种实现

  - 文档混乱脱节

  TensorFlow虽不完美，但社区强大，适合生产环境

- Keras

  一个高层神经网络API，纯Python编写，使用TensorFlow、Theano及CNTK作为后端。最容易上手的框架之一

  缺点：过度封装导致丧失灵活性和运行缓慢，学习者只是调用接口，无法深入底层细节

- Caffe/Caffe2(Convolutional Architecture for Fast Embedding)

  C++，支持命令行。优点：简洁快速，缺点：缺乏灵活性，难扩展，难配置，Caffe2是之后的改良版，速度快，性能优异，适合生产环境

- MXNet

  强大的分布式支持、内存显存优化明显，由于MXNet长期处于快速迭代的过程中，但文档却长期未更新，新用户难以掌握，老用户需要查阅源码才能使用。Gluon是MXNet 的接口之一，它模仿PyTorch的接口设计。

- CNTK

  微软开源，支持CPU和GPU模式，计算图结构，叶子节点代表输入或网络参数，其他节点代表计算步骤，CNTK是一个非常强大的命令行系统，擅长语音方面的研究

- 其他框架

  百度：PaddlePaddle

  CMU：DyNet

  英特尔：Nervana

  Amazon：DSSTNE

  C++：tiny-dnn

  Java：Deeplearning4J

  针对移动设备：CoreML、MDL


## 计算图

- 动态图：define and run，一次定义多次运行，一旦创建不可修改，定义时使用特殊语法，同时无法使用if、while和for-loop。静态图一般比较庞大，占用过高的显存，静态图可以预先优化。
- 静态图：define by run，多次定义多次运行，每次前向传播(每次调用代码运行)都会创建一幅新的计算图

## Tensor

PyTorch中重要的数据结构，类似于高维数组

- 一个数——标量
- 一维数组——向量
- 二维数组——矩阵
- 更高维数组

Tensor很接近numpy的ndarrys，但Tensor可以使用GPU加速。

