import numpy as np
import random

class Network(object):
    def __init__(self, sizes):
        """
            sizes: list类型, 代表各层包含神经元的个数
            
            偏差bias，权重weight使用高斯分布初始化
        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y,1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x,y in zip(sizes[:-1], sizes[1:])]
        
    def feedforward(self, a):
        """对于给定的输入a，返回神经网络的输出
        
            a: 输入向量
        """
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a
    
    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        """使用小批量随机梯度下降算法训练神经网络
        
            输入：
                training_data: a list of tuples``(x,y) 
                                    representing the training inputs and the desired outputs.
                                    
                epochs: 迭代次数
                
                mini_batch_size: 小批量的大小
                
                eta: 学习速率
                
                test_data: 测试数据集，形式跟training_data一样"""
        if test_data: n_test = len(test_data)
        n_train = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n_train, mini_batch_size)]
            
            for mini_batch in mini_batches:
                self.updata_mini_batch(mini_batch, eta)
                
            if test_data:
                print("Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete.".format(j))
                
    def updata_mini_batch(self, mini_batch, eta):
        """随机梯度下降，使用BP算法更新神经网络的权重和偏置
            通过计算当前 mini_batch 中的训练样本对 Network 的权重和偏置进行了更新
        
            输入: 
                mini_batch: a list of tuples ``(x,y)``
                
                eta: 学习速率learning rate"""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        
        for x,y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x,y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            
        self.weights = [w - (eta/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)]
        self.baises = [b - (eta/len(mini_batch))*nb for b, nb in zip(self.biases, nabla_b)]
        
    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the gradient for the cost function C_x."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
            
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)
    
    def evaluate(self, test_data):
        """返回test_data中，正确预测的数量"""
        test_results = [(np.argmax(self.feedforward(x)), y) for (x,y) in test_data]
        return sum(int(x == y) for (x,y) in test_results)
    
    def cost_derivative(self, output_activations, y):
        """返回损失函数的偏导数结果"""
        return (output_activations - y)
    
        
def sigmoid(z):
    """The sigmoid function."""
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))