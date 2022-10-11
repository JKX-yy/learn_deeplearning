# 深度学习-Pytorch

## 一   序言

### 1. 全书内容

![image-20221011111914982](C:\Users\86183\AppData\Roaming\Typora\typora-user-images\image-20221011111914982.png)

 [1节](https://zh.d2l.ai/chapter_introduction/index.html#chap-introduction) 提供深度学习的入门课程

 [2节](https://zh.d2l.ai/chapter_preliminaries/index.html#chap-preliminaries) 中，我们将快速向你介绍实践深度学习所需的前提条件，例如如何存储和处理数据，以及如何应用基于线性代数、微积分和概率基本概念的各种数值运算。

[3节](https://zh.d2l.ai/chapter_linear-networks/index.html#chap-linear) 和 [4节](https://zh.d2l.ai/chapter_multilayer-perceptrons/index.html#chap-perceptrons) 涵盖了深度学习的最基本概念和技术，例如线性回归、多层感知机和正则化。

 [5节](https://zh.d2l.ai/chapter_deep-learning-computation/index.html#chap-computation) 描述了深度学习计算的各种关键组件，并为我们随后实现更复杂的模型奠定了基础。

 [6节](https://zh.d2l.ai/chapter_convolutional-neural-networks/index.html#chap-cnn) 和 [7节](https://zh.d2l.ai/chapter_convolutional-modern/index.html#chap-modern-cnn) 中，我们介绍了卷积神经网络（convolutional neural network，CNN）

 [8节](https://zh.d2l.ai/chapter_recurrent-neural-networks/index.html#chap-rnn) 和 [9节](https://zh.d2l.ai/chapter_recurrent-modern/index.html#chap-modern-rnn) 中，我们引入了循环神经网络(recurrent neural network，RNN)

 [10节](https://zh.d2l.ai/chapter_attention-mechanisms/index.html#chap-attention) 中，我们介绍了一类新的模型，它采用了一种称为注意力机制的技术，

[11节](https://zh.d2l.ai/chapter_optimization/index.html#chap-optimization) 中，我们讨论了用于训练深度学习模型的几种常用优化算法

 [12节](https://zh.d2l.ai/chapter_computational-performance/index.html#chap-performance) 将探讨影响深度学习代码计算性能的几个关键因素

[13节](https://zh.d2l.ai/chapter_computer-vision/index.html#chap-cv) 中，我们展示了深度学习在计算机视觉中的主要应用

 [14节](https://zh.d2l.ai/chapter_natural-language-processing-pretraining/index.html#chap-nlp-pretrain) 和 [15节](https://zh.d2l.ai/chapter_natural-language-processing-applications/index.html#chap-nlp-app) 中，我们展示了如何预训练语言表示模型并将其应用于自然语言处理任务。

### 2. 代码

有时，为了避免不必要的重复，我们将本书中经常导入和引用的函数、类等封装在`d2l`包中。对于要保存到包中的任何代码块，比如一个函数、一个类或者多个导入，我们都会标记为`#@save`。我们在 [16.6节](https://zh.d2l.ai/chapter_appendix-tools-for-deep-learning/d2l.html#sec-d2l) 中提供了这些函数和类的详细描述。`d2l`软件包是轻量级的，仅需要以下软件包和模块作为依赖项：

```python
#@save
import collections
import hashlib
import math
import os
import random
import re
import shutil
import sys
import tarfile
import time
import zipfile
from collections import defaultdict
import pandas as pd
import requests
from IPython import display
from matplotlib import pyplot as plt
from matplotlib_inline import backend_inline

d2l = sys.modules[__name__]
```

下面是我们如何从PyTorch导入模块。

```python
#@save
import numpy as np
import torch
import torchvision
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils import data
from torchvision import transforms
```

### 3.小结

1.深度学习已经向彻底改变了模式识别，引入了一些列技术，包括计算机视觉，自然语言处理，自动语音识别。

2.要成功地应用深度学习，你必须知道如何抛出一个问题、建模的数学方法、将模型与数据拟合的算法，以及实现所有这些的工程技术。

3.这本书提供了一个全面的资源，包括文本、图表、数学和代码，都集中在一个地方。

4.要回答与本书相关的问题，请访问我们的论坛[discuss.d2l.ai](https://discuss.d2l.ai/).

5.所有Jupyter记事本都可以在GitHub上下载。

### 4. 安装

我们需要配置一个环境来运行 Python、Jupyter Notebook、相关库以及运行本书所需的代码，以快速入门并获得动手学习经验。

### 安装miniconda

如果已安装conda，则可以跳过以下步骤。

conda、miniconda、anaconda的区别以及在pycharm中选择conda的虚拟环境

conda是一种通用包管理系统，旨在构建和管理任何语言和任何类型的软件。举个例子：包管理与pip的使用类似，环境管理则允许用户方便地安装不同版本的python并可以快速切换。

Anaconda则是一个打包的集合，里面预装好了conda、某个版本的python、众多packages、科学计算工具等等，就是把很多常用的不常用的库都给你装好了。

Miniconda，顾名思义，它只包含最基本的内容——python与conda，以及相关的必须依赖项，对于空间要求严格的用户，Miniconda是一种选择。就只包含最基本的东西，其他的库得自己装。

## 二  前言

## 1.1. 日常生活中的机器学习

## 1.2. 关键组件

1. 我们可以学习的*数据*（data）。
2. 如何转换数据的*模型*（model）。
3. 一个*目标函数*（objective function），用来量化模型的有效性。
4. 调整模型参数以优化目标函数的*算法*（algorithm）。

### 1.2.1. 数据

 每个数据集由一个个*样本*（example, sample）组成，大多时候，它们遵循独立同分布(independently and identically distributed, i.i.d.)。

通常每个样本由一组称为*特征*（features，或*协变量*（covariates））的属性组成。

机器学习模型会根据这些属性进行预测。 在上面的监督学习问题中，要预测的是一个特殊的属性，它被称为*标签*（label，或*目标*（target））。

当每个样本的特征类别数量都是相同的时候，其特征向量是固定长度的，这个长度被称为数据的*维数*（dimensionality）。 固定长度的特征向量是一个方便的属性，它有助于我们量化学习大量样本。

**与传统机器学习方法相比，深度学习的一个主要优势是可以处理不同长度的数据。**

**机器学习\*（ML）\***

使用标准的机器学习的方法，我们需要手动选择图像的相关特征，以训练机器学习模型。然后，模型在对新对象进行分析和分类时引用这些特征。

通过深度学习的工作流程，可以从图像中自动提取相关功能。另外，深度学习是一种端到端的学习，网络被赋予原始数据和分类等任务，并且可以自动完成。

### 1.2.2. 模型

深度学习与经典方法的区别主要在于：前者关注的功能强大的模型，这些模型由神经网络错综复杂的交织在一起，包含层层数据转换，因此被称为*深度学习*（deep learning）。 在讨论深度模型的过程中，我们也将提及一些传统方法。

### 1.2.3. 目标函数

*目标函数*（objective function）。 我们通常定义一个目标函数，并希望优化它到最低点。 因为越低越好，所以这些函数有时被称为*损失函数*（loss function，或cost function）。

通常，损失函数是根据模型参数定义的，并取决于数据集。

 该数据集由一些为训练而收集的样本组成，称为*训练数据集*（training dataset，或称为*训练集*（training set））

 然而，在训练数据上表现良好的模型，并不一定在“新数据集”上有同样的效能，这里的“新数据集”通常称为*测试数据集*（test dataset，或称为*测试集*（test set））。

### 1.2.4. 优化算法

我们接下来就需要一种算法，它能够搜索出最佳参数，以最小化损失函数。通常基于一种基本方法–*梯度下降*（gradient descent）

 然后，它在可以减少损失的方向上优化参数。

## 1.3. 各种机器学习问题

### 1.3.1. 监督学习

*监督学习*（supervised learning）擅长在“给定输入特征”的情况下预测标签。 每个“特征-标签”对都称为一个*样本*（example）

 **我们的目标是生成一个模型，能够将任何输入特征映射到标签，即预测。**

监督学习的应用：许多重要的任务可以清晰地描述为：**在给定一组特定的可用数据的情况下，估计未知事物的概率。**

#### 1.3.1.1. 回归

*回归*（regression）是最简单的监督学习任务之一

假设你在市场上寻找新房子，你可能需要估计一栋房子的公平市场价值。 销售价格，即标签，是一个数值。 当标签取任意数值时，我们称之为*回归*问题。 **我们的目标是生成一个模型，它的预测非常接近实际标签值。**

总而言之，判断回归问题的一个很好的经验法则是，任何有关“**多少**”的问题很可能就是回归问题。比如：

- 这个手术需要多少小时？

- 在未来六小时，这个镇会有多少降雨量？

  

#### 1.3.1.2. 分类

这种“哪一个？”的问题叫做*分类*（classification）问题。 在*分类*问题中，我们希望模型能够预测样本属于哪个*类别*（category，正式称为*类*（class））

**分类问题的常见损失函数被称为*交叉熵*（cross-entropy）**

#### 1.3.1.3. 标记问题

学习预测不相互排斥的类别的问题称为*多标签分类*（multi-label classification）。 举个例子，人们在技术博客上贴的标签，比如“机器学习”、“技术”、“小工具”、“编程语言”、“Linux”、“云计算”、“AWS”。 一篇典型的文章可能会用5-10个标签，因为这些概念是相互关联的。

#### 1.3.1.4. 搜索

有时，我们不仅仅希望输出为一个类别或一个实值。 在信息检索领域，我们希望对一组项目进行排序。

#### 1.3.1.5. 推荐系统

另一类与搜索和排名相关的问题是*推荐系统*（recommender system），它的目标是向特定用户进行“个性化”推荐。 例如，对于电影推荐，科幻迷和喜剧爱好者的推荐结果页面可能会有很大不同。 类似的应用也会出现在零售产品、音乐和新闻推荐等等。

#### 1.3.1.6. 序列学习

这些问题是序列学习的实例，是机器学习最令人兴奋的应用之一。 序列学习需要摄取输入序列或预测输出序列，或两者兼而有之。 具体来说，输入和输出都是可变长度的序列，例如机器翻译和从语音中转录文本。 虽然不可能考虑所有类型的序列转换，但以下特殊情况值得一提。

**自动语音识别**。**文本到语音**。**机器翻译**。

### 1.3.2. 无监督学习

到目前为止，所有的例子都与**监督学习**有关，即我们向模型提供巨大数据集：每个样本包含特征和相应标签值。 打趣一下，“监督学习”模型像一个打工仔，有一份极其专业的工作和一位极其平庸的老板。 老板站在身后，准确地告诉模型在每种情况下应该做什么，直到模型学会从情况到行动的映射。 取悦这位老板很容易，只需尽快识别出模式并模仿他们的行为即可。

相反，如果你的工作没有十分具体的目标，你就需要“自发”地去学习了。 （如果你打算成为一名数据科学家，你最好培养这个习惯。） 比如，你的老板可能会给你一大堆数据，然后让你用它做一些数据科学研究，却没有对结果有要求。 我们称这类数据中不含有“目标”的机器学习问题为***无监督学习*（unsupervised learning），**

- *聚类*（clustering）问题：没有标签的情况下，我们是否能给数据分类呢？比如，给定一组照片，我们能把它们分成风景照片、狗、婴儿、猫和山峰的照片吗？同样，给定一组用户的网页浏览记录，我们能否将具有相似行为的用户聚类呢？
- *主成分分析*（principal component analysis）问题：我们能否找到少量的参数来准确地捕捉数据的线性相关属性？比如，一个球的运动轨迹可以用球的速度、直径和质量来描述。再比如，裁缝们已经开发出了一小部分参数，这些参数相当准确地描述了人体的形状，以适应衣服的需要。另一个例子：在欧几里得空间中是否存在一种（任意结构的）对象的表示，使其符号属性能够很好地匹配?这可以用来描述实体及其关系，例如“罗马” − “意大利” + “法国” = “巴黎”。
- *因果关系*（causality）和*概率图模型*（probabilistic graphical models）问题：我们能否描述观察到的许多数据的根本原因？例如，如果我们有关于房价、污染、犯罪、地理位置、教育和工资的人口统计数据，我们能否简单地根据经验数据发现它们之间的关系？
- *生成对抗性网络*（generative adversarial networks）：为我们提供**一种合成数据的方法**，甚至像图像和音频这样复杂的非结构化数据。潜在的统计机制是检查真实和虚假数据是否相同的测试，它是无监督学习的另一个重要而令人兴奋的领域。

### 1.3.3. 与环境互动

 到目前为止，不管是监督学习还是无监督学习，我们都会预先获取大量数据，然后启动模型，不再与环境交互。 这里所有学习都是在算法与环境断开后进行的，被称为***离线学习***（offline learning）

### 1.3.4. 强化学习

如果你对使用机器学习开发与环境交互并采取行动感兴趣，那么你最终可能会专注于*强化学习*（reinforcement learning）。 这可能包括应用到机器人、对话系统，甚至开发视频游戏的人工智能（AI）**AlphaGo 程序在棋盘游戏围棋中击败了世界冠军，是两个突出强化学习的例子。**

在强化学习问题中，agent在一系列的时间步骤上与环境交互。 在每个特定时间点，agent从环境接收一些*观察*（observation），并且必须选择一个*动作*（action），然后通过某种机制（有时称为执行器）将其传输回环境，最后agent从环境中获得*奖励*（reward）。 此后新一轮循环开始，agent接收后续观察，并选择后续操作，依此类推。 强化学习的过程在 [图1.3.7](https://zh.d2l.ai/chapter_introduction/index.html#fig-rl-environment) 中进行了说明。 请注意，强化学习的目标是产生一个好的*策略*（policy）。 强化学习agent选择的“动作”受策略控制，即一个从环境观察映射到行动的功能。

![](C:\Users\86183\AppData\Roaming\Typora\typora-user-images\image-20221011182224082.png)

当环境可被完全观察到时，我们将强化学习问题称为*马尔可夫决策过程*（markov decision process）。 当状态不依赖于之前的操作时，我们称该问题为*上下文赌博机*（contextual bandit problem）。 当没有状态，只有一组最初未知回报的可用动作时，这个问题就是经典的*多臂赌博机*（multi-armed bandit problem）。

当环境可被完全观察到时，我们将强化学习问题称为*马尔可夫决策过程*（markov decision process）。 当状态不依赖于之前的操作时，我们称该问题为*上下文赌博机*（contextual bandit problem）。 当没有状态，只有一组最初未知回报的可用动作时，这个问题就是经典的*多臂赌博机*（multi-armed bandit problem）

## 1.4. 起源

## 1.5. 深度学习之路

 另外，廉价又高质量的传感器、廉价的数据存储（克莱德定律）以及廉价计算（摩尔定律）的普及，特别是GPU的普及，使大规模算力唾手可得。

## 1.6. 成功案例



## 1.7. 特点

到目前为止，我们已经广泛地讨论了机器学习，它既是人工智能的一个分支，也是人工智能的一种方法。 虽然深度学习是机器学习的一个子集，但令人眼花缭乱的算法和应用程序集让人很难评估深度学习的具体成分是什么。

如前所述，机器学习可以使用数据来学习输入和输出之间的转换，

深度学习是“深度”的，模型学习了许多“层”的转换，每一层提供一个层次的表示。 例如，靠近输入的层可以表示数据的低级细节，而接近分类输出的层可以表示用于区分的更抽象的概念。

深度学习的一个关键优势是它不仅取代了传统学习管道末端的浅层模型而且还取代了劳动密集型的特征工程过程。

# 二  预备知识

要学习深度学习，首先需要先掌握一些基本技能。 所有机器学习方法都涉及从数据中提取信息。 因此，我们先学习一些关于数据的实用技能，包括存储、操作和预处理数据。

## 2.1. 数据操作

1）获取数据；（2）将数据读入计算机后对其进行处理。

首先，我们介绍n维数组，也称为*张量*（tensor）。 使用过Python中NumPy计算包的读者会对本部分很熟悉。 无论使用哪个深度学习框架，它的*张量类*（在MXNet中为`ndarray`， 在PyTorch和TensorFlow中为`Tensor`）都与Numpy的`ndarray`类似。 但深度学习框架又比Numpy的`ndarray`多一些重要功能： 首先，GPU很好地支持加速计算，而**NumPy仅支持CPU**计算； 其次，张量类支持自动微分。 这些功能使得张量类更适合深度学习。 如果没有特殊说明，本书中所说的张量均指的是张量类的实例。

### 2.1.1. 入门

首先，我们导入`torch`。请注意，虽然它被称为PyTorch，但是代码中使用`torch`而不是`pytorch`。

```python
import torch
```

```python
import torch  #导入ptorch包
x=torch.arange(12)  #arange创建一个行向量，t）。例如，张量 x 中有 12 个元素。除非额外指定，新的张量将存储在内存中，并采用基于CPU的计算。x.shapr
x.shape  #查看张量形状
x.numel()  #查看张量中元素总数
X=x.reshape(2,6) #改变张量形状我们不需要通过手动指定每个维度来改变形状。 也就是说，如果我们的目标形状是（高度,宽度）， 即我们可以用x.reshape(-1,4)或x.reshape(3,-1)来取代x.reshape(3,4)。
X.shape #shape不带括号
torch.zeros((3,2,1))  #三个两行一列的张量，我们希望使用全0、全1、其他常量
torch.ones((2, 3, 4))#两个三行四列的张量，
```

