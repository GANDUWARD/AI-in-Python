import numpy as np
import math as math
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split


class LoR:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate  # 一般取0.0001
        self.classes = []
        self.class_map = {}
        self.images, self.targets = fetch_openml("mnist_784", return_X_y=True, as_frame=False)
        self.images = self.images.reshape(-1, 28, 28)
        self.W = np.random.rand(28 * 28, 3)  # 为W初始化

    def basic_struction_of_net(self):
        self.input_channel = 450 * 28 * 28 * 1  # N*D
        self.classify_channel = 28 * 28 * 3  # D*K
        self.output_channel = 3  # 1*K

    def logistic(self):
        lg = 1 / (np.exp((-1) * self.p) + 1)  # f(x):logistics函数
        return lg

    def cross_entropy(self, Y, P):  # 损失函数L(x)
        cey = (-1) * Y * math.log(P)
        return cey  # 最后返回平均值

    def sigmoid(self, u):  # 激活函数用sigmoid
        sig = 1.0000000000 / (1.0 + np.exp(u * (-1)))
        return sig

    def sigmoid_backward(self, dW, Z):  # sigmoid 反向传播
        sig = self.sigmoid(Z)
        return dW * sig * (1 - sig)

    def F(self):
        f = np.dot(self.X_train, self.W)
        return f

    def softmax(self, P):  # softmax函数激活
        sof = 0
        for i in P:
            for j in i:
                sof = sof + np.exp(j)
        f = np.exp(P)
        f = f / sof
        return f

    def softmax_backward(self):
        A = self.softmax(self.F())  # ai
        for i, k in zip(A, self.y_reckon):
            i[k] = i[k] + 1
        return np.dot(self.X_train.T,A)

    def grad_down(self):
        self.W = self.W - self.learning_rate * self.softmax_backward()  # 更新权重


    def data_processor(self):
        # 取3个类各200张图完成即可
        # 示例取类2,3,4
        classes = ['2', '3', '4']
        data = []
        label = [[i] * 200 for i in range(len(classes))]
        for l in classes:
            data.append(self.images[self.targets == l][: 200])
        data = np.concatenate(data, axis=0)
        label = np.concatenate(label)
        data = data.reshape(600, -1) / 255  # 压缩值0-1之间方便计算距离
        # 划分数据集为训练集和测试集
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(data, label)
        self.y_reckon = len(self.y_test) * [0]
        # 建立映射方便表示。由于我们只取3类,映射为0,1,2就行
        self.class_map = {'2': 0, '3': 1, '4': 2}

    def average(self, k):
        dudu = 0
        for i in k:  # 归一化处理
            for j in i:
                dudu = dudu + j
        k = k / dudu
        return k

    def work(self):
        o = self.F()  # 一次映射
        sof = self.softmax(o)  # 用激活函数
        cey = 0
        for i, j in zip(sof, self.y_reckon):  # 定位索引，找最大值来估计
            self.y_reckon[j] = np.argmax(i)  # 估计值
        for i, j in zip(self.y_reckon, self.y_train):  # 计算损失函数
            if j == i:
                cey = cey + self.cross_entropy(1, sof[i][self.y_reckon[i]])
        L = cey / 450
        index = 0
        for i, j in zip(self.y_reckon, self.y_train):
            if i != j:
                index = index + 1
        print(f"错误率是{index / 450}")
        print(f"损失函数为{L}")
        self.grad_down()
