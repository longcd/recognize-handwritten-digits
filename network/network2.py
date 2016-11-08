# library
import json
import random
import sys
import numpy as np

# 定义交叉熵代价函数
class CrossEntropyCost(object):
    """用来表示交叉熵代价的类"""
    
    @staticmethod
    def fn(a, y):
        """交叉熵代价函数"""
        # np.nan_to_num 调用确保了 Numpy 正确处理接近 0 的对数值
        return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))
        
    @staticmethod
    def delta(z, a, y):
        """网络输出误差"""
        return (a-y)
    
# 定义二次代价函数
class QuadraticCost(object):
    """二次代价函数"""
    
    @staticmethod
    def fn(a, y):
        """二次代价函数"""
        return 0.5 * np.linalg.norm(a-y)**2
    
    @staticmethod
    def delta(z, a, y):
        """网络输出误差"""
        return (a-y) * sigmoid_prime(z)

# 前馈神经网络
class Network(object):
    def __init__(self, sizes, cost=CrossEntropyCost):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.default_weight_initializer()
        self.cost = cost
        
    def default_weight_initializer(self):
        """
            新式改进后的初始化权重方法
            权重初始化，使用了均值为 0 而标准差为 1/sqrt(n),n 为对应的输入连接个数;
            偏置与之前一样，使用均值为 0 标准差为 1 的高斯分布.
        """
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x)/np.sqrt(x) for x,y in zip(self.sizes[:-1], self.sizes[1:])]
        
    def large_weight_initializer(self):
        """
            最开始的初始化权重方法
            权重和偏置，使用了均值为0，而标准差为1的高斯分布
        """
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(self.sizes[:-1], self.sizes[1:])]
        
    def feedforward(self, a):
        """对与给定的输入``a``， 返回网络的输出"""
        for b,w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a
    
    def SGD(self, training_data, epochs, mini_batch_size, eta, lmbda=0.0, evaluation_data=None, 
            monitor_evaluation_cost=False, monitor_evaluation_accuracy=False, 
            monitor_training_cost=False, monitor_training_accuracy=False):
        """使用 小批量随机梯度下降算法 训练神经网络
        
            返回包含4个list的tuple，分别是每次迭代的验证集的代价，验证集的精确率，训练集的代价，训练集的精确率"""
        if evaluation_data: n_eval = len(evaluation_data)
        n_train = len(training_data)
        evaluation_cost, evaluation_accuracy = [], []
        training_cost, training_accuracy = [], []
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n_train, mini_batch_size)]
            
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta, lmbda, len(training_data))
            
            print("Epoch %s training complete" % j)
            if monitor_training_cost:
                cost = self.total_cost(training_data, lmbda)
                training_cost.append(cost)
                print("Cost on training data: {}".format(cost))
            if monitor_training_accuracy:
                accuracy = self.accuracy(training_data, convert=True)
                training_accuracy.append(accuracy)
                print("Accuracy on training: {} / {}".format(accuracy, n_train))
            if monitor_evaluation_cost:
                cost = self.total_cost(evaluation_data, lmbda, convert=True)
                evaluation_cost.append(cost)
                print("Cost on evaluation data: {}".format(cost))
            if monitor_evaluation_accuracy:
                accuracy = self.accuracy(evaluation_data)
                evaluation_accuracy.append(accuracy)
                print("Accuracy on evaluation data: {} / {}".format(accuracy, n_eval))
                
        return evaluation_cost, evaluation_accuracy, training_cost, training_accuracy
    
    def update_mini_batch(self, mini_batch, eta, lmbda, n):
        """更新网络的权重"""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x,y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x,y)
            nabla_b = [nb+dnb for nb,dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw,dnw in zip(nabla_w, delta_nabla_w)]
            
        self.weights = [(1-eta*(lmbda/n))*w -(eta/len(mini_batch))*nw for w,nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb for b,nb in zip(self.biases, nabla_b)]
        
    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the gradient for the cost function C_x."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x]
        zs = []
        for b,w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = (self.cost).delta(zs[-1], activations[-1], y)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)
    
    def accuracy(self, data, convert=False):
        if convert:
            results = [(np.argmax(self.feedforward(x)), np.argmax(y)) for (x,y) in data]
        else:
            results = [(np.argmax(self.feedforward(x)), y) for (x,y) in data]
        return sum(int(x==y) for (x,y) in results)
    
    def total_cost(self, data, lmbda, convert=False):
        cost = 0.0
        for x, y in data:
            a = self.feedforward(x)
            if convert: y = vectorized_result(y)
            cost += self.cost.fn(a,y)/len(data)
        cost += 0.5*(lmbda/len(data))*sum(np.linalg.norm(w)**2 for w in self.weights)
        return cost
    
    def save(self, filename):
        data = {"sizes": self.sizes,
               "weights": [w.tolist() for w in self.weights],
               "biases": [b.tolist() for b in self.biases],
               "cost": str(self.cost.__name__)}
        f = open(filename, "w")
        json.dump(data, f)
        f.close()
        
def load(filename):
    f = open(filename, "r")
    data = json.load(f)
    f.close()
    cost = getattr(sys.modules[__name__], data["cost"])
    net = Network(data["sizes"], cost=cost)
    net.weights = [np.array(w) for w in data["weights"]]
    net.biases = [np.array(b) for b in data["biases"]]
    return net

def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z) * (1-sigmoid(z))