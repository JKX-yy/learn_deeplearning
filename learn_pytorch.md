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



pycharm配置环境以及一些常用包：

![image-20221018180818350](C:\Users\86183\AppData\Roaming\Typora\typora-user-images\image-20221018180818350.png)

python3(ipykernel)  python3的内核环境

python[conda evn:root]conda环境  root（默认存在）

python[conda evn:dl]另外配置的环境

python[conda evn:dp]另外配置的环境

深度学习的python一般选择3.7

终端换环境  conda  activate dp (在对应环境下安装包)

![image-20221018181418592](C:\Users\86183\AppData\Roaming\Typora\typora-user-images\image-20221018181418592.png)

### 常用库和包名

常用库：

![image-20221018190002117](C:\Users\86183\AppData\Roaming\Typora\typora-user-images\image-20221018190002117.png)

```py
import torch
import numpy as np  
##可以用来存储和处理大型矩阵，支持大量的维度数组与矩阵运算，此外也针对数组运算提供大量的数学函数库。
import os   #文件处理
#使用%matplotlib命令可以将matplotlib的图表直接嵌入到Notebook之中，或者使用指定的界面库显示图表，它有一个参数指定matplotlib图表的显示方式。inline表示将图表嵌入到Notebook中。
%matplotlib inline
d2l包
#d2l 包 是李沐老师对与《动手学习深度学习》 中提供代码的使用其他框架时候的公共库。
#包含 3 大类的可使用库：
#mxnet
#pytorch
#TensorFlow
#本文，主要针对这个 d2l 库进行一些基本的解析和学习，以便我们自己在进行深度学习代码编写的时候
#有所参考。包的总体结果如下：
from matplotlib_inline import backend_inline

```



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

# 二  预备知识()

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
#以下代码创建一个形状为（3,4）的张量。 其中的每个元素都从均值为0、标准差为1的标准高斯分布（正态分布）中随机采样。
torch.randn(3,4)
#我们还可以通过提供包含数值的Python列表（或嵌套列表），来为所需张量中的每个元素赋予确定值。 在这里，最外层的列表对应于轴0，内层的列表对应于轴1。
torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])

```

### 2.1.2. 运算符

```py
#对于任意具有相同形状的张量， 常见的标准算术运算符（+、-、*、/和**）都可以被升级为按元素运算。 我们可以在同一形状的任意两个张量上调用按元素操作。其中每个元素都是按元素操作的结果。
#2.1.2. 运算符
x=torch.tensor([1.0,2,4,8])#torch.tensor用于定义给固定值的张量  只要有一个为浮点数就都为浮点列表
y=torch.tensor([2,2,2,2])
x+y,x-y,x*y,x/y,x**y  #x**y =x的y次方
torch.exp(x)       #指数运算
#x为0-11的浮点型，3行4列
X=torch.arange(12,dtype=torch.float32).reshape((3,4))
Y=torch.tensor([[2.0,1,4,3],[1,2,3,4],[4,3,2,1]])
torch.cat((X,Y),dim=0),torch.cat((X,Y),dim=1) #拼接两个张量 torch.cat  dim=0纵轴  dim=1横轴
#有时，我们想通过逻辑运算符构建二元张量。 以X == Y为例： 对于每个位置，如果X和Y在该位置相等，则新张量中相应项的值为1。 这意味着逻辑语句X == Y在该位置处为真，否则该位置为0。
X==Y
#对张量中的所有元素进行求和，会产生一个单元素张量。
X.sum()

```

### 2.1.3. 广播机制

两个形状不同的张量仍然可以相加。

```py
#2.1.3
a=torch.arange(3).reshape((3,1))
b=torch.arange(2).reshape((1,2))
a,b,
#由于a和b分别是和矩阵，如果让它们相加，它们的形状不匹配。 我们将两个矩阵广播为一个更大的矩阵，如下所示：矩阵a将复制列， 矩阵b将复制行，然后再按元素相加。
a+b

(tensor([[0],
         [1],
         [2]]),
 tensor([[0, 1]]),
 tensor([[0, 1],
         [1, 2],
         [2, 3]]))

```



### 2.1.4. 索引和切片

张量中的元素可以通过索引访问。 与任何Python数组一样：第一个元素的索引是0，最后一个元素索引是-1； 可以指定范围以包含第一个元素和最后一个之前的元素。

如下所示，我们可以用`[-1]`选择最后一个元素，可以用`[1:3]`选择第二个和第三个元素：

```python
#2.1.4
X,Y
X[-2],X[1][-1],X[:2]  #倒数第二行，第二行最后一个元素，0，1行
#除读取外，我们还可以通过指定索引来将元素写入矩阵。
X[1][-1]=7.0 #X[1,-1]
X[1]=12  #一行多个元素付相同值


```

### 2.1.5. 节省内存

```py
#2.1.5
#运行Y = Y + X后，我们会发现id(Y)指向另一个位置。 这是因为Python首先计算Y + X，为结果分配新的内存，然后使Y指向内存中的这个新位置。
d1=id(Y)
Y=Y+X
d2=id(Y)
d1,d2
(2371743325536, 2371740398272)
#这可能是不可取的，原因有两个：首先，我们不想总是不必要地分配内存。 在机器学习中，我们可能有数百兆的参数，并且在一秒内多次更新所有参数。 通常情况下，我们希望原地执行这些更新。 其次，如果我们不原地更新，其他引用仍然会指向旧的内存位置， 这样我们的某些代码可能会无意中引用旧的参数。
#幸运的是，执行原地操作非常简单。 我们可以使用切片表示法将操作的结果分配给先前分配的数组，例如Y[:] = <expression>。 为了说明这一点，我们首先创建一个新的矩阵Z，其形状与另一个Y相同， 使用zeros_like来分配一个全的块。
Z=torch.zeros_like(Y)
print(f"idZ={id(Z)}")
Z[:]=X+Y
idx=id(X)
X[:]=X+Y   #切片可以减少内存开销，使新结果=原地址
idx2=id(X)
idx==idx2
X+=Z    #方法2
idx3=id(X)
idx==idx3
```

### 2.1.6. 转换为其他Python对象

将深度学习框架定义的张量转换为NumPy张量（`ndarray`）很容易，反之也同样容易。 torch张量和numpy数组将共享它们的底层内存，就地操作更改一个张量也会同时更改另一个张量。

```py
#2.1.6
A=X.numpy()
B=torch.tensor(A)
type(A),type(B)
#要将大小为1的张量转换为Python标量，我们可以调用item函数或Python的内置函数。
a=torch.tensor([3.5])
a,a.item,float(a),int(a)

(tensor([3.5000]), <function Tensor.item>, 3.5, 3)
```

## 2.2. 数据预处pandas

为了能用深度学习来解决现实世界的问题，我们经常从预处理原始数据开始， 而不是从那些准备好的张量格式数据开始。 **在Python中常用的数据分析工具中，我们通常使用`pandas`软件包。** 像庞大的Python生态系统中的许多其他扩展包一样，`pandas`可以与张量兼容。 本节我们将简要介绍使用`pandas`预处理原始数据，并将原始数据转换为张量格式的步骤。 我们将在后面的章节中介绍更多的数据预处理技术。

### 2.2.1. 读取数据集

创建一个人工数据集存储CSV文件

```py
import os
import pandas as pd
os.makedirs(os.path.join('test'),exist_ok=True)  #os.path.join()用于拼接路径，可以传入多个参数。返回一个路径>>> print(os.path.join('path','abc','yyy'))  path\abc\yyy
file=os.path.join('test','data.csv')
with open (file,'w') as f:
    f.write('NumRooms,Alley,Price\n')
    f.write('NA,Pave,127500\n')  # 每行表示一个数据样本
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')
data_=pd.read_csv(file)
print(data_)
 with  open(file,'r') as f:
     data=pd.read_csv(f)
     print(data)



#2.2.1
import os  #创建文件包含的库
import pandas as pd  #pandas用于数据处理
os.makedirs(os.path.join('data'),exist_ok=True)  #创建文佳佳
data_file=os.path.join('data','house_tiny.csv')#在data_file=/data/house_tiny.csv
with open (data_file,'w') as f:   #写模式    with open(r'filename.txt')as f:
    f.write('NumRooms,Alley,Price\n')
    f.write('NA,Pave,127500\n')  # 每行表示一个数据样本
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')
data=pd.read_csv(data_file)
print(data)


```

### 2.2.2. 处理缺失值

注意，“NaN”项代表缺失值。 为了处理缺失的数据，典型的方法包括*插值法*和*删除法*， 其中插值法用一个替代值弥补缺失值，而删除法则直接忽略缺失值。 在这里，我们将考虑插值法。

通过位置索引`iloc`，我们将`data`分成`inputs`和`outputs`， 其中前者为`data`的前两列，而后者为`data`的最后一列。 对于`inputs`中缺少的数值，我们用同一列的均值替换“NaN”项。

```py
#2.2.2. 处理缺失值
#print(data)
inputs,outputs=data.iloc[:,0:2],data.iloc[:,2]
inputs=inputs.fillna(inputs.mean())  #.fillna 填充NaN值。inputs.mean()计算平均值
print(inputs)
#对于inputs中的类别值或离散值，我们将“NaN”视为一个类别。 由于“巷子类型”（“Alley”）列只接受两种类型的类别值“Pave”和“NaN”， pandas可以自动将此列转换为两列“Alley_Pave”和“Alley_nan”。 巷子类型为“Pave”的行会将“Alley_Pave”的值设置为1，“Alley_nan”的值设置为0。 缺少巷子类型的行会将“Alley_Pave”和“Alley_nan”分别设置为0和1。
inputs=pd.get_dummies(inputs,dummy_na=True)
print(inputs)

   NumRooms Alley
0       3.0  Pave
1       2.0   NaN
2       4.0   NaN
3       3.0   NaN
   NumRooms  Alley_Pave  Alley_nan
0       3.0           1          0
1       2.0           0          1
2       4.0           0          1
3       3.0           0          1
```

### 2.2.3. 转换为张量格式

现在`inputs`和`outputs`中的所有条目都是数值类型，它们可以转换为张量格式。 当数据采用张量格式后，可以通过在 [2.1节](https://zh.d2l.ai/chapter_preliminaries/ndarray.html#sec-ndarray)中引入的那些张量函数来进一步操作。

```py
#2.2.3
import torch
X,y=torch.tensor(inputs.values),torch.tensor(outputs.values)
X,y

(tensor([[3., 1., 0.],
         [2., 0., 1.],
         [4., 0., 1.],
         [3., 0., 1.]], dtype=torch.float64),
 tensor([127500, 106000, 178100, 140000]))
```

`pandas`软件包是Python中常用的数据分析工具中，`pandas`可以与张量兼容。

用`pandas`处理缺失的数据时，我们可根据情况选择用插值法和删除法。

## 2.3. 线性代数

在你已经可以存储和操作数据后，让我们简要地回顾一下部分基本线性代数内容。 这些内容能够帮助你了解和实现本书中介绍的大多数模型。 本节我们将介绍线性代数中的基本数学对象、算术和运算，并用数学符号和相应的代码实现来表示它们。

### 2.3.1. 标量

标量由只有一个元素的张量表示。 在下面的代码中，我们实例化两个标量，并执行一些熟悉的算术运算，即加法、乘法、除法和指数。

```py

#2.3.1
import torch
x=torch.tensor(3.0)
y=torch.tensor(2.0)
x+y,x*y,x/y,x**y

(tensor(5.), tensor(6.), tensor(1.5000), tensor(9.))
```

### 2.3.2. 向量

你可以将向量视为标量值组成的列表。 我们将这些标量值称为向量的*元素*（element）或*分量*（component）

```py
x = torch.arange(4)
x
#我们可以使用下标来引用向量的任一元素。
x[3]

```

#### 2.3.2.1. 长度、维度和形状

向量只是一个数字数组，就像每个数组都有一个长度一样，每个向量也是如此。 在数学表示法中，如果我们想说一个向量x由n个实值标量组成， 我们可以将其表示为x∈Rn。 向量的长度通常称为向量的*维度*（dimension）。

与普通的Python数组一样，我们可以通过调用Python的内置`len()`函数来访问张量的长度。

```py
len(x)
#当用张量表示一个向量（只有一个轴）时，我们也可以通过.shape属性访问向量的长度。 形状（shape）是一个元素组，列出了张量沿每个轴的长度（维数）。 对于只有一个轴的张量，形状只有一个元素。
torch.Size([4])
```

### 2.3.3. 矩阵

正如向量将标量从零阶推广到一阶，矩阵将向量从一阶推广到二阶。 矩阵，我们通常用粗体、大写字母来表示 （例如，X、Y和Z）， 在代码中表示为具有两个轴的张量。

```py
A = torch.arange(20).reshape(5, 4)
A.T
B = torch.tensor([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
B == B.T
```

### 2.3.4. 张量

 张量（本小节中的“张量”指代数对象）为我们提供了描述具有任意数量轴的n维数组的通用方法。

 例如，向量是一阶张量，矩阵是二阶张量。 张量用特殊字体的大写字母表示（例如，X、Y和Z）， 它们的索引机制（例如xijk和[X]1,2i−1,3）与矩阵类似。

当我们开始处理图像时，张量将变得更加重要，图像以n维数组形式出现， 其中3个轴对应于**高度、宽度，**以及一个***通道*（channel）轴**， 用于表示颜色通道（红色、绿色和蓝色）。 现在，我们先将高阶张量暂放一边，而是专注学习其基础知识。

```py
X = torch.arange(24).reshape(2, 3, 4)
```

### 2.3.5. 张量算法的基本性质

标量、向量、矩阵和任意数量轴的张量（本小节中的“张量”指代数对象）有一些实用的属性。 例如，你可能已经从按元素操作的定义中注意到，任何按元素的一元运算都不会改变其操作数的形状。 同样，给定具有相同形状的任意两个张量，任何按元素二元运算的结果都将是相同形状的张量。 例如，将两个相同形状的矩阵相加，会在这两个矩阵上执行元素加法。

```py
A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
B = A.clone()  # 通过分配新内存，将A的一个副本分配给B
A, A + B
#具体而言，两个矩阵的按元素乘法称为Hadamard积（Hadamard product）（数学符号）。
A * B  #按元素乘积
#tensor([[  0.,   1.,   4.,   9.],
      #  [ 16.,  25.,  36.,  49.],
       # [ 64.,  81., 100., 121.],
       # [144., 169., 196., 225.],
       # [256., 289., 324., 361.]])
#将张量乘以或加上一个标量不会改变张量的形状，其中张量的每个元素都将与标量相加或相乘。
a = 2
X = torch.arange(24).reshape(2, 3, 4)
a + X, (a * X).shape

```

### 2.3.6. 降维

我们可以对任意张量进行的一个有用的操作是计算其元素的和。 在数学表示法中，我们使用∑符号表示求和。 为了表示长度为d的向量中元素的总和，可以记为∑i=1dxi。 在代码中，我们可以调用计算求和的函数：

```py
x = torch.arange(4, dtype=torch.float32)
x, x.sum()
#我们可以表示任意形状张量的元素和。 例如，矩阵中元素的和可以记为
A.shape, A.sum()
#(torch.Size([5, 4]), tensor(190.))


#2.3.6
import torch
x = torch.arange(4, dtype=torch.float32)
x, x.sum()
A,A.shape, A.sum()
A_sum_axis0 = A.sum(axis=0)  #以0轴求和
A_sum_axis0, A_sum_axis0.shape
A.sum(axis=[0,1])
#计算平均值 
# 一个与求和相关的量是平均值（mean或average）。 我们通过将总和除以元素总数来计算平均值。 在代码中，我们可以调用函数来计算任意形状张量的平均值。
A,A.numel(),A.sum(),A.mean()
```

```py
#2.3.6.1非降维求和
#例如，由于sum_A在对每行进行求和后仍保持两个轴，我们可以通过广播将A除以sum_A。
sum_A=A.sum(axis=0,keepdims=True)  #横着写纵轴计算
A,sum_A,A/sum_A,A.cumsum(axis=0)
#如果我们想沿某个轴计算A元素的累积总和， 比如axis=0（按行计算），我们可以调用cumsum函数。 此函数不会沿任何轴降低输入张量的维度。

```

```py
#2.3.7点积  dot 按元素乘积之和
y=torch.ones(4,dtype=torch.float32)
x,y,torch.dot(x,y)
#2.3.8矩阵-向量积   torch.mv用于实现两矩阵相乘 行-列。
A.shape,x.shape,torch.mv(A,x)
```

```py
#2.3.9. 矩阵-矩阵乘法
#2.3.9  矩阵-矩阵乘积  
B=torch.ones(4,3)
torch.mm(A,B)
#2.3.10范数  假设维向量中的元素是，其范数是向量元素平方和的平方根：
u=torch.tensor([3.0,-4.0])
# 线性代数中最有用的一些运算符是范数（norm）。
 #假设维向量中的元素是，其范数是向量元素平方和的平方根：
torch.norm(u)
#在深度学习中，我们更经常地使用范数的平方。 你还会经常遇到范数，它表示为向量元素的绝对值之和：
torch.abs(u).sum()

```

```py
#类似于向量的范数，矩阵的Frobenius范数（Frobenius norm）是矩阵元素平方和的平方根：
torch.norm(torch.ones((4, 9)))

```

## 2.4微积分

### 2.4.1. 导数和微分

为了更好地解释导数，让我们做一个实验。 定义u=f(x)=3x2−4x如下：

```py
#2.4微积分
#变得更好意味着小化一个损失函数
#2.4.1导数和微分  为了更好地解释导数，让我们做一个实验。 定义如下：
%matplotlib inline
#使用%matplotlib命令可以将matplotlib的图表直接嵌入到Notebook之中，或者使用指定的界面库显示图表，它有一个参数指定matplotlib图表的显示方式。inline表示将图表嵌入到Notebook中。

import numpy as np  #可以用来存储和处理大型矩阵，支持大量的维度数组与矩阵运算，此外也针对数组运算提供大量的数学函数库。
from matplotlib_inline import backend_inline
from d2l import torch as d2l
#为了更好地解释导数，让我们做一个实验。 定义如下：
def f(x):
    return 3*x**3-4*x
def numerical_lim(f,x,h):
    return (f(x+h)-f(x))/h
h=0.1
for i in range(5):
    print(f'h={h:.5f},numerical_lim={numerical_lim(f,1,h):.5f}')
    h*=0.1
    
```



```py
#为了对导数的这种解释进行可视化，我们将使用matplotlib， 这是一个Python中流行的绘图库。要配置matplotlib生成图形的属性，我们需要定义几个函数。 在下面，use_svg_display函数指定matplotlib软件包输出svg图表以获得更清晰的图像。注意，注释#@save是一个特殊的标记，会将对应的函数、类或语句保存在d2l包中。 因此，以后无须重新定义就可以直接调用它们（例如，d2l.use_svg_display()）。
def use_svg_display():#@save
    '''使用svg格式在jupyter中绘图显示'''
    backend_inline.set_matplotlib_formats('svg')
def set_figsize(figsize=(3.5,2.5)):#@save
    '''设置matplotlib的图标大小'''
    use_svg_display()
    d2l.plt.rcParams['figure.figsize']=figsize
#@save
def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """设置matplotlib的轴"""
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()

#@save
def plot(X, Y=None, xlabel=None, ylabel=None, legend=None, xlim=None,
         ylim=None, xscale='linear', yscale='linear',
         fmts=('-', 'm--', 'g-.', 'r:'), figsize=(3.5, 2.5), axes=None):
    """绘制数据点"""
    if legend is None:
        legend = []

    set_figsize(figsize)
    axes = axes if axes else d2l.plt.gca()

    # 如果X有一个轴，输出True
    def has_one_axis(X):
        return (hasattr(X, "ndim") and X.ndim == 1 or isinstance(X, list)
                and not hasattr(X[0], "__len__"))

    if has_one_axis(X):
        X = [X]
    if Y is None:
        X, Y = [[]] * len(X), X
    elif has_one_axis(Y):
        Y = [Y]
    if len(X) != len(Y):
        X = X * len(Y)
    axes.cla()
    for x, y, fmt in zip(X, Y, fmts):
        if len(x):
            axes.plot(x, y, fmt)
        else:
            axes.plot(y, fmt)
    set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
#现在我们可以绘制函数及其在处的切线， 其中系数是切线的斜率。
x=np.arange(0,3,0.1)
plot(x,[f(x),2*x-3],'x','f(x)',legend=['f(x)','Tangent line (x=1)'])


```

### 2.4.2. 偏导数

### 2.4.3. 梯度

我们可以连结一个多元函数对其所有变量的偏导数，以得到该函数的*梯度*（gradient）向量。 

设函数f:Rn→R的输入是 一个n维向量x=[x1,x2,…,xn]⊤，并且输出是一个标量。 函数f(x)相对于x的梯度是一个包含n个偏导数的向量:

![image-20221018191832323](C:\Users\86183\AppData\Roaming\Typora\typora-user-images\image-20221018191832323.png)

### 2.4.4. 链式法则

然而，上面方法可能很难找到梯度。 这是因为在深度学习中，多元函数通常是*复合*（composite）的， 所以我们可能没法应用上述任何规则来微分这些函数。 幸运的是，链式法则使我们能够微分复合函数。

![image-20221018191941562](C:\Users\86183\AppData\Roaming\Typora\typora-user-images\image-20221018191941562.png)

## 2.5自动微分

深度学习框架通过自动计算导数，即*自动微分*（automatic differentiation）来加快求导。 实际中，根据我们设计的模型，系统会构建一个*计算图*（computational graph）， 来跟踪计算是哪些数据通过哪些操作组合起来产生输出。 自动微分使系统能够随后反向传播梯度。 这里，*反向传播*（backpropagate）意味着跟踪整个计算图，填充关于每个参数的偏导数。

### 2.5.1例子

```py
#2.5.1
import torch

x = torch.arange(4.0)
x.requires_grad_(True)  # 等价于x=torch.arange(4.0,requires_grad=True)
x.grad  # 默认值是None
y = 2 * torch.dot(x, x)
y.backward()
y,x.grad
x.grad==4*x
#默认情况下pytorch会累计梯度，需要清除之前的值
x.grad.zero_()
y=x.sum()
y.backward()
x.grad


```

### 2.5.2 非标量变量的反向传播

当`y`不是标量时，向量`y`关于向量`x`的导数的最自然解释是一个矩阵。 对于高阶和高维的`y`和`x`，求导的结果可以是一个高阶张量。

```
# 对非标量调用backward需要传入一个gradient参数，该参数指定微分函数关于self的梯度。
# 在我们的例子中，我们只想求偏导数的和，所以传递一个1的梯度是合适的
```



```py
#2.5.2
x.grad.zero_()
y=x*x
# 等价于y.backward(torch.ones(len(x)))
y.sum().backward()  #标量反向传播不需要.sum()变成标量   非标量反向传播需要.sum（）变成标量
x.grad

```

2.5.3. 分离计算

]有时，我们希望将某些计算移动到记录的计算图之外。 例如，假设`y`是作为`x`的函数计算的，而`z`则是作为`y`和`x`的函数计算的。 想象一下，我们想计算`z`关于`x`的梯度，但由于某种原因，我们希望将`y`视为一个常数， 并且只考虑到`x`在`y`被计算后发挥的作用。

```py
#2.5.3
x.grad.zero_()
y=x*x
u=y.detach()  #detach  拆卸
z=u*x
z.sum().backward()
x.grad
#由于记录了y的计算结果，我们可以随后在y上调用反向传播， 得到y=x*x关于的x的导数，即2*x。
x.grad.zero_()
y.sum().backward()
x.grad
```

### 2.5.4. Python控制流的梯度计算

使用自动微分的一个好处是： 即使构建函数的计算图需要通过Python控制流（例如，条件、循环或任意函数调用），我们仍然可以计算得到的变量的梯度。 在下面的代码中，`while`循环的迭代次数和`if`语句的结果都取决于输入`a`的值。

```py
def f(a):
    b = a * 2
    while b.norm() < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c
a = torch.randn(size=(), requires_grad=True)
d = f(a)
d.backward()
a.grad == d / a

```

## 2.6概率



```py
counts=multinomial.Multinomial(10,fair_probs).sample((500,))  #进行500次实验每组抽取十个样本。
cum_counts=counts.cumsum(dim=0)  #cumsum返回累计和沿固定轴   这个函数的功能是返回给定axis上的累计和    第一行不动，累加到第二行，第二行累加到第三行。。。。

#cum_counts,cum_counts.sum(dim=1,keepdims=True)
estimates=cum_counts/cum_counts.sum(dim=1,keepdims=True)
# estimates,counts.shape
d2l.set_figsize((6,4.5))
for i in range(6):
  d2l.plt.plot(estimates[:, i].numpy(),
                 label=("P(die=" + str(i + 1) + ")"))
d2l.plt.axhline(y=0.167, color='black', linestyle='dashed')
d2l.plt.gca().set_xlabel('Groups of experiments')
d2l.plt.gca().set_ylabel('Estimated probability')
d2l.plt.legend();
```

### 2.6.1.1. 概率论公理

### 2.6.1.2. 随机变量

### 2.6.2. 处理多个随机变量

### 2.6.2.1. 联合概率

### 2.6.2.2. 条件概率

### 2.6.2.3. 贝叶斯定理

![image-20221019152710149](C:\Users\86183\AppData\Roaming\Typora\typora-user-images\image-20221019152710149.png)

### 2.6.2.4. 边际化

![image-20221019152807712](C:\Users\86183\AppData\Roaming\Typora\typora-user-images\image-20221019152807712.png)

### 2.6.2.5. 独立性

### 2.6.2.6. 应用

### 2.6.3. 期望和方差

# 3 线性神经网络

## 3.1. 线性回归

### 3.1.1. 线性回归的基本元素

我们希望根据房屋的面积（平方英尺）和房龄（年）来估算房屋价格（美元）。

 预测所依据的自变量（面积和房龄）称为*特征*（feature）或*协变量*（covariate）。

训练集   样本（每一行） 标签（预测的目标）

#### 3.1.1.1. 线性模型、

线性假设是指目标（房屋价格）可以表示为特征（面积和房龄）的加权和，如下面的式子：

(3.1.1)price=warea⋅area+wage⋅age+b.

[(3.1.1)](https://zh.d2l.ai/chapter_linear-networks/linear-regression.html#equation-eq-price-area)中的warea和wage 称为*权重*（weight），权重决定了每个特征对我们预测值的影响。 b称为*偏置*（bias）、*偏移量*（offset）或*截距*（intercept）。 偏置是指当所有特征都取值为0时，预测值应该为多少。 即使现实中不会有任何房子的面积是0或房龄正好是0年，我们仍然需要偏置项。 如果没有偏置项，我们模型的表达能力将受到限制。 严格来说， [(3.1.1)](https://zh.d2l.ai/chapter_linear-networks/linear-regression.html#equation-eq-price-area)是输入特征的一个 *仿射变换*（affine transformation）。 仿射变换的特点是通过加权和对特征进行*线性变换*（linear transformation）， 并通过偏置项来进行*平移*（translation）。

![image-20221019164907450](C:\Users\86183\AppData\Roaming\Typora\typora-user-images\image-20221019164907450.png)

这个过程中的求和将使用广播机制 （广播机制在 [2.1.3节](https://zh.d2l.ai/chapter_preliminaries/ndarray.html#subsec-broadcasting)中有详细介绍）。 给定训练数据特征X和对应的已知标签y， 线性回归的目标是找到一组权重向量w和偏置b： 当给定从X的同分布中取样的新样本特征时， 这组权重向量和偏置能够使得新样本预测标签的误差尽可能小。

虽然我们相信给定x预测y的最佳模型会是线性的， 但我们很难找到一个有n个样本的真实数据集，其中对于所有的1≤i≤n，y(i)完全等于w⊤x(i)+b。 无论我们使用什么手段来观察特征X和标签y， 都可能会出现少量的观测误差。 因此，即使确信特征与标签的潜在关系是线性的， 我们也会加入一个噪声项来考虑观测误差带来的影响。

在开始寻找最好的*模型参数*（model parameters）w和b之前， 我们还需要两个东西： （1）一种模型质量的度量方式； （2）一种能够更新模型以提高模型预测质量的方法。

#### 3.1.1.2. 损失函数

模型质量越好损失函数越小

损失函数：真实值与预测值之间的差值，差值越小损失越小，完美预测损失为零。回归问题中最常见的孙损失函数时：平方误差函数
$$
l^i(w,b)=1/2(y^(i)-y^(i))^2
$$
1/2求导后系数为一。

![image-20221019180334651](C:\Users\86183\AppData\Roaming\Typora\typora-user-images\image-20221019180334651.png)

#### 3.1.1.3. 解析解

线性回归十本书中最简单的模型，与其他模型不同，线性模型的姐可以用公式表述出来，叫做解析解。

![image-20221019180944902](C:\Users\86183\AppData\Roaming\Typora\typora-user-images\image-20221019180944902.png)

#### 3.1.1.4. 随机梯度下降

对于不能求出解析解的模型。

随机梯度下降法：几乎对于所有模型的优化都可以用随机梯度下降法。随即提速下降实在损失函数递减的方向上不断更新参数。

minibatch stochastic gradient descent(小批量随机梯度下降)：

梯度下降计算损失函数关于模型参数的导数。避免每次更新时都遍历数据集（费时），由此提出小批量随机梯度下降。每次更新参数时只取小批量样本。

<u>**参数更新过程：取小批量B  η表示学习率  （批量大小和学习率一般是手动预先指定但是可以调整  不是在跟新的参数称为超参数）**</u>

![image-20221019183133313](C:\Users\86183\AppData\Roaming\Typora\typora-user-images\image-20221019183133313.png)

算法步骤：

（1）随机初始化参数模型值。

（2）从数据集上随机抽取小批量样本并且在负梯度的方向上更新参数，不断迭代

![image-20221019183511208](C:\Users\86183\AppData\Roaming\Typora\typora-user-images\image-20221019183511208.png)

小结：

*调参*（hyperparameter tuning）是选择超参数的过程。 超参数通常是我们根据训练迭代结果来调整的， 而训练迭代结果是在独立的*验证数据集*（validation dataset）上评估得到的。

最终参数值只会使损失函数降低在最小值附近（收敛）并不会真正达到最小值。

线性回归是在整个领域中只有一个最小值学习的问题，但是对于深度神经网络这样复杂的模型来说有多个最小值（寻找这样一组最小值很难，而且需要泛化的过程，使这组参数在未知数据集上也能实现较低的损失）

#### 3.1.1.5. 用模型进行预测



### 3.1.2. 矢量化加速

希望同时处理整个小批量样本  需要对计算进行矢量化（利用线性代数库进行矢量化，节省Python开销）



```py
#3.1.2
%matplotlib inline
import math
import time
import numpy as np
import torch
from d2l import torch as d2l
n=10000
a=torch.ones(n)
b=torch.ones(n)
#由于在本书中我们将频繁地进行运行时间的基准测试，所以我们定义一个计时器：
class Timer:  #@save
    """记录多次运行时间"""
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        """启动计时器"""
        self.tik = time.time()

    def stop(self):
        """停止计时器并将时间记录在列表中"""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """返回平均时间"""
        return sum(self.times) / len(self.times)

    def sum(self):
        """返回时间总和"""
        return sum(self.times)

    def cumsum(self):
        """返回累计时间"""
        return np.array(self.times).cumsum().tolist()
#for循环实行每一位的加法
c = np.zeros(n)
timer = Timer()
timer.start()  #开始计时
for i in range(n):
    c[i] = a[i] + b[i]
f'{timer.stop():.5f} sec'  #停止计时

timer.start()  #开始计时
d=a+b
print(f'{timer.stop():.8f}')#停止计时

```



### 3.1.3. 正态分布与平方损失

![image-20221019200221198](C:\Users\86183\AppData\Roaming\Typora\typora-user-images\image-20221019200221198.png)

![image-20221019200249729](C:\Users\86183\AppData\Roaming\Typora\typora-user-images\image-20221019200249729.png)

在高斯噪声的假设下，最小化均方误差等于对线性模型的极大似然估计。

### 3.1.4. 从线性回归到深度网络

### 3.1.4.1. 神经网络图

![image-20221019200645793](C:\Users\86183\AppData\Roaming\Typora\typora-user-images\image-20221019200645793.png)

线性回归是只有一个神经元的神经网络、

#### 3.1.4.2. 生物学

### 3.1.5. 小结



### 3.1.6. 练习

1. 假设我们有一些数据x1,…,xn∈R。我们的目标是找到一个常数b，使得最小化∑i(xi−b)2。
   1. 找到最优值b的解析解。
   2. 这个问题及其解与正态分布有什么关系?
   3. 

```py
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
class TestDataSet(Dataset):
    def __init__(self):
        super(TestDataSet, self).__init__()
        self.X = torch.normal(5, 1, (n,))

    def __getitem__(self, item):
        return self.X[item]

    def __len__(self):
        return self.X.shape[0]

bias = 0
lr = 0.01
batch_size = 50
epochs = 3
X = TestDataSet()
dataLoader = DataLoader(X, batch_size, shuffle=True)
for epoch in range(epochs):
    for (i, x) in enumerate(dataLoader):
        bias = bias + lr * 2 * (x.sum() - batch_size * bias) / batch_size
print(bias)
```



## 3.2. 线性回归的从零开始实现











### 3.2.1. 生成数据集

### 3.2.2. 读取数据集

### 3.2.3. 初始化模型参数

### 3.2.4. 定义模型

### 3.2.5. 定义损失函数

### 3.2.6. 定义优化算法

### 3.2.7. 训练

### 3.2.8. 小结

### 3.2.9. 练习、

## 3.3. 线性回归的简洁实现

### 3.3.1. 生成数据集

### 3.3.2. 读取数据集

### 3.3.3. 定义模型

### 3.3.4. 初始化模型参数

### 3.3.5. 定义损失函数

### 3.3.6. 定义优化算法

### 3.3.7. 训练

### 3.3.8. 小结

### 3.3.9. 练习、、

## 3.4. softmax回归

### 3.4.1. 分类问题

### 3.4.2. 网络架构

### 3.4.3. 全连接层的参数开销[3.4.4. softmax运算

### 3.4.5. 小批量样本的矢量化

### 3.4.6. 损失函数

### 3.4.7. 信息论基础

### 3.4.8. 模型预测和评估

### 3.4.9. 小结

### 3.4.10. 练习

## 3.5. 图像分类数据集

### 3.5.1. 读取数据集

### 3.5.2. 读取小批量

### 3.5.3. 整合所有组件

### 3.5.4. 小结

### 3.5.5. 练习

## 3.6. softmax回归的从零开始实现

### 3.6.1. 初始化模型参数

### 3.6.2. 定义softmax操作

### 3.6.3. 定义模型

### 3.6.4. 定义损失函数

### 3.6.5. 分类精度

### 3.6.6. 训练

### 3.6.7. 预测[3.6.8. 小结

### 3.6.9. 练习

## 3.7. softmax回归的简洁实现

### 3.7.1. 初始化模型参数

### 3.7.2. 重新审视Softmax的实现

### 3.7.3. 优化算法

### 3.7.4. 训练

### 3.7.5. 小结

### 3.7.6. 练习
