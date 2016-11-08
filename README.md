# 项目介绍

## 1. 内容简介

本项目最终将基于Python实现的前馈神经网络实现一个手写数字识别系统，系统会在服务器启动时自动读入训练好的神经网络文件，用户可以通过在html页面上手写数字发送给服务器来得到识别结果。

## 2. 知识点

本项目完成过程中，我们将学习：

1. 什么是神经网络
2. 在浏览器完成手写数据的输入与请求的发送
3. 在服务器端根据请求调用神经网络模块并给出响应
4. 实现BP神经网络
5. 多分类Logistic regression
6. sklearn库中SVM的使用

## 3. 系统构成

我们的手写数字识别系统分为5部分，分别写在4个文件中：

- 客户端（`digit_recognizer.js`）
- 服务器（`server.py`）
- 用户接口（`index.html`）
- 神经网络(`network2.py`)

客户端(`index.html`)是一个html页面，用户在canvans上写数字，之后点击选择预测。客户端(`digit_recognizer.js`)将收集到的手写数字组合成一个数组发送给服务器端(`server.py`)处理，服务器加载训练完成后的模型(`network2.json`)并进行预测，然后将结果返回给客户端。

## 4. 数据集来源

> 官网：http://yann.lecun.com/exdb/mnist/

MNIST数据集是一个手写数字的数据集。训练集包含了60,000个样本，测试集包含了10,000个样本。它的每个样本都被规范处理为一张28px*28px的灰度图。

该数据集包含4个文件：

- train-images-idx3-ubyte.gz:  training set images (9912422 bytes)
- train-labels-idx1-ubyte.gz:  training set labels (28881 bytes)
- t10k-images-idx3-ubyte.gz:   test set images (1648877 bytes)
- t10k-labels-idx1-ubyte.gz:   test set labels (4542 bytes) 

使用 `./data/code/run.sh` 下载上述4个文件并解压。

使用 `./data/code/transform.c` 将数据集转换为 csv 格式，并将训练集拆分成：50,000 个样本作为训练集和 10,000 个样本作为验证集。

# 5. 模型实现

## 5.1 k最近邻算法(kNN)

## 5.2 Logistic regression

## 5.3 SVM

## 5.4 前馈神经网络

# 6. 结果演示

输入`python server.py`打开服务器。

在页面上写一个数字预测看看：

![image](https://github.com/longcd/recognize-handwritten-digits/raw/master/2.png)

![image](https://github.com/longcd/recognize-handwritten-digits/raw/master/pred2.png)

