先加载 MNIST 数据
```
>>> import mnist_loader
>>> training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
```

在加载完MNIST数据之后，我们将设置一个有30个隐藏层神经元的Network。
```
>>> import network
>>> net = network.Network([784, 30, 10])
```

最后，将使用随机梯度下降来从MNIST training_data学习超过30次迭代期，小批量数据大小为10，学习速率η=3.0，
```
>>> net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
```