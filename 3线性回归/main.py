import random
import numpy as np
import matplotlib.pyplot as plt


def write(file_place):             #写入函数
    i = 1
    X = []
    Y = []
    first_lines = []
    for line in file_place:
        first_line = line.split('\n')
        for a in first_line:
            if a == '':
                first_line.remove('')
        first_lines.append(first_line)
        random.shuffle(first_lines)
        for b in first_lines:
            all_points = [u.split('\t') for u in b]
        for d in all_points:
            for e in d:
                if i % 2 != 0:
                    X.append(e)
                    i += 1
                else:
                    Y.append(e)
                    i += 1
    x_t = list(map(float, X))
    y_t = list(map(float, Y))
    return x_t, y_t


def paint(x, y):
    print(x)
    print(y)
    plt.scatter(x, y)
    plt.show()


def calculate(x_in, y_in):
    size = len(x_in)
    sum_xy = 0
    sum_x_square = 0
    sum_x = 0
    sum_y = 0
    for i, j in zip(x_in, y_in):
        sum_xy += (i * j)
        sum_x_square += i ** 2
        sum_x += i
        sum_y += j
    x_avg = sum_x / size
    y_avg = sum_y / size
    a = (sum_xy - size * x_avg * y_avg) / (sum_x_square - size * (x_avg ** 2))
    b = y_avg - (a * x_avg)
    return a, b, y_avg


def check(a, b, avg):
    with open('C:\\Users\\pc\\Desktop\\py\\3线性回归\\test.txt') as train_place:
        x_test, y_test = write(train_place)
        y_evaluate = []
        Rss = 0
        Tss = 0
        for v in x_test:
            y_evaluate.append((a * v) + b)
        for p, q in zip(y_evaluate, y_test):
            Rss += (p - q) ** 2
            Tss += (q - avg) ** 2
        R_square = 1 - (Rss / Tss)
        print(f"cost:{R_square}")
        return x_test, y_test, y_evaluate


def final_show(x_1, y_1, x_2, y_2, y_0, a, b):
    X_sum = []
    Y_sum = []
    Y = []
    for i, j in zip(x_1, y_1):
        X_sum.append(i)
        Y_sum.append(j)
    for i, j in zip(x_2, y_2):
        X_sum.append(i)
        Y_sum.append(j)
    for i in X_sum:
        Y.append(a * i + b)
    plt.scatter(X_sum, Y_sum)
    plt.plot(X_sum, Y)
    plt.show()


with open('C:\\Users\\pc\\Desktop\\py\\3线性回归\\train.txt') as fp:
    x_train, y_train = write(fp)
    k, l, y_ba = calculate(x_train, y_train)
    x_last, y_last, y_really = check(k, l, y_ba)
    final_show(x_train, y_train, x_last, y_last, y_really, k, l)