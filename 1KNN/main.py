import math
import copy
import numpy as np
import random


class KNN:

    def __init__(self):
        self.fp = open('C:\\Users\\pc\\Desktop\\py\\3线性回归\\test.txt')
        self.origin_data = np.ndarray(4, 270)
        self.train_data = np.ndarray(4, 270 * 0.7)
        self.test_data = np.ndarray(4, 270 * 0.3)
        self.K = 3
        self.X_train = []
        self.Y_train = []
        self.Z_train = []

    def write(self):  # 写入函数
        self.origin_data = self.fp.readlines()
        random.shuffle(self.fp.readlines())
        u = 0
        self.train_data = self.origin_data[4:int(270 * 0.7)]
        self.test_data = self.train_data[4:int(270 * 0.3)]
        map(float, self.train_data)
        map(float, self.test_data)
        return self.train_data, self.test_data

    def K_N_N(self, use_data):
        distance = []
        X = []
        Y = []
        Z = []
        index = []
        first_lines = []
        i = 1
        for line in use_data:
            first_line = line.split('\n')
            for a in first_line:
                if a == '':
                    first_line.remove('')
            first_lines.append(first_line)
            for b in first_lines:
                all_points = [u.split('\t') for u in b]
            for d in all_points:
                for e in d:
                    if i % 4 == 1:
                        X.append(e)
                        i += 1
                        continue
                    if i % 4 == 2:
                        Y.append(e)
                        i += 1
                        continue
                    if i % 4 == 3:
                        Z.append(e)
                        i += 1
                        continue
                    else:
                        index.append(e)
                        i += 1
                        continue
        X = list(map(float, X))
        Y = list(map(float, Y))
        Z = list(map(float, Z))
        index = list(map(float, index))
        return X, Y, Z, index

    def judge(self, x_test, y_test, z_test):
        distance = []
        for x, y, z in zip(self.X_train, self.Y_train, self.Z_train):
            ou = x ** 2 + y ** 2 + z ** 2
            ou = math.sqrt(ou)
            distance.append(ou)
        t = copy.deepcopy(distance)
        min_distance = []
        min_index = []
        for _ in range(self.K):
            number = min(t)
            index = t.index(number)
            t[index] = 0
            min_distance.append(number)
            min_index.append(index)
        t = []
        return min_distance, min_index


with open('C:\\Users\\pc\\Desktop\\py\\3线性回归\\train.txt') as fp:
    k = KNN
    k.fp = fp
    k.K = 3
    train_data, test_data = k.write(self=k)
    X_train, Y_train, Z_train, index_train = k.K_N_N(self=k, use_data=train_data)
    k.X_train = X_train
    k.Y_train = Y_train
    k.Z_train = Z_train
    X_test, Y_test, Z_test, index_test = k.K_N_N(self=k, use_data=test_data)
    index_test = [0.0]
    count = random.randint(0, 270 * 0.3)
    min_distance, min_index = k.judge(self=k, x_test=X_test[count], y_test=Y_test[count], z_test=Z_test[count])
    min_label = min(min_index, key=min_index.count)
    print(index_train[min_label])
