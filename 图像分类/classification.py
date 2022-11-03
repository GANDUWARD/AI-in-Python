import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split


class plot:

    def __init__(self):
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_reckon = None
        self.distances_train = None
        self.distances_test = None
        self.classes = []
        self.class_map = {}
        self.images, self.targets = fetch_openml("mnist_784", return_X_y=True, as_frame=False)
        self.images = self.images.reshape(-1, 28, 28)
        self.k = 3
        self.L = 0

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
        self.distances_test = len(self.y_test) * [0]
        self.distances_train = len(self.y_train) * [0]

        # 建立映射方便表示。由于我们只取3类,映射为0,1,2就行
        self.class_map = {'2': 0, '3': 1, '4': 2}

    def show(self, count):
        plt.imshow(self.images[count], cmap='gray')
        plt.show()

    def get_test(self):
        print(self.X_test)
        print(self.y_test)
        print(self.X_train)
        print(self.y_train)

    def KNN(self):
        count = 0
        for i in self.X_test:
            current_distances = []
            for j in self.X_train:
                current_distances.append(np.sum((i ** 2 - j ** 2)) ** 0.5)
            distances = current_distances
            index_k = []
            for u in range(self.k):
                index_i = distances.index(min(distances))  # 得到列表的最小值，并得到该最小值的索引
                index_k.append(index_i)  # 记录最小值索引
                distances[index_i] = float('inf')
            fake_dis_in = []  # 根据索引寻找y值
            for h in index_k:
                fake_dis_in.append(self.y_train[h])
            self.y_reckon[count] = max(set(fake_dis_in), key=fake_dis_in.count)  # 为y的预测值赋值为索引元素中出现最多的标签
            count = count + 1

    def get_L(self):
        for i in range(len(self.y_test)):
            if self.y_test[i] != self.y_reckon[i]:
                self.L = self.L + 1
        print(f"损失函数为{self.L}")
