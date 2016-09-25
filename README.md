#　使用神经网络识别手写数字

本项目我们将实现一个可以识别手写数字的神经网络。这个程序仅仅100多行，不使用特别的神经网络库。然而，这个短小的网络不需要人类帮助便可以超过 96% 的准确率识别数字。

# 数据集

本项目使用著名的[MNIST](http://yann.lecun.com/exdb/mnist/)数据集，进行训练、验证和测试。该数据集包含 60,000 个训练样本，和 10,000 个测试样本。每个样本都是 28*28 像素的灰度值图像。关于该数据集更多信息前到[这里](http://yann.lecun.com/exdb/mnist/)来查看。

# 知识点

本项目运用了以下知识点：

- S型神经元
- 前馈神经网络
- 梯度下降算法
- 二次代价函数、交叉熵代价函数
- 反向传播算法
- 过拟合、规范化

本项目使用 `jupyter notebook (ipython notebook)` 进行展示。

`Github` 加载 `.ipynb` 的速度较慢，建议在 [network](http://nbviewer.jupyter.org/github/longcd/recognize-handwritten-digits/blob/master/mnist_network.ipynb) 

[network2](http://nbviewer.jupyter.org/github/longcd/recognize-handwritten-digits/blob/master/mnist_network2.ipynb)中查看该项目。

----

## 1.一个简单的分类手写数字的网络

我们将使用一个三层神经网络来识别单个数字：

1. 网络的输入层包含给输入像素的值进行编码的神经元。我们给网络的训练数据会有很多扫描得到的 28×28 的手写数字的图像组成,所有输入层包含有 784=28×28 个神经元。
2. 网络的第二层是一个隐藏层。我们用 n 来表示神经元的数量,我们将给 n 实验不同的数值。
3. 网络的输出层包含有 10 个神经元。我们把输出神经元的输出赋予编号 0 到 9,并计算出那个神经元有最高的激活值。比如,如果编
号为 6 的神经元激活,那么我们的网络会猜到输入的数字是 6。其它神经元相同。

## 2.使用梯度下降算法进行学习

MNIST 数据分为两个部分。第一部分包含 60,000 幅用于训练数据的图像。这些图像是 28×28 大小的灰度图像。第二部分是 10,000 幅用于测试数据的图像,同样是 28 × 28 的灰度图像。我们将用这些测试数据来评估我们的神经网络学会识别数字有多好。

- 为了方便,把每个训练输入 x 看作一个 28×28=784 维的向量。
- 我们用 y = y(x) 表示对应的期望输出,这里 y 是一个 10 维的向量。

我们希望有一个算法,能让我们找到权重和偏置,以至于网络的输出 y(x) 能够拟合所有的训练输入x。为了量化我们如何实现这个目标,我们定义一个代价函数:
![image](https://github.com/longcd/recognize-handwritten-digits/raw/master/QuadraticCost.png)  

使用梯度下降算法解决代价函数的最小化问题：

![image](https://github.com/longcd/recognize-handwritten-digits/raw/master/SGD.png)  

## 3.实现我们的网络来分类数字

首先，我们将 60,000 个图像的 MNIST 训练集分成两个部分:一部分 50,000 个图像,我们将用来训练我们的神经网络,和一个单独的 10,000 个图像的验证集。

具体实现在 [network](http://nbviewer.jupyter.org/github/longcd/recognize-handwritten-digits/blob/master/mnist_network.ipynb)  中查看。

## 4.改进神经网络的学习方法

引入交叉熵代价函数:

![image](https://github.com/longcd/recognize-handwritten-digits/raw/master/CrossEntropyCost.png) 

规范化：

L2规范化：![image](https://github.com/longcd/recognize-handwritten-digits/raw/master/L2.png) 

L1规范化：![image](https://github.com/longcd/recognize-handwritten-digits/raw/master/L1 .png) 

---

[Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/index.html)