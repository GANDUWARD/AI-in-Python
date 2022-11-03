import GP
import numpy as np
import random


def get_data(fileplace):
    i = 1
    X = []
    Y = []
    first_lines = []
    for line in fileplace:
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


def data_come():
    fp = open('train.TXT')
    fp2 = open('test.TXT')
    x_train, y_train = get_data(fp)
    X_train = np.array(x_train).reshape(100, 1)
    Y_train = np.array(y_train).reshape(100, 1)
    x_test, y_test = get_data(fp2)
    X_test = np.array(x_test).reshape(100, 1)
    Y_test = np.array(y_test).reshape(100, 1)
    return X_train, Y_train, X_test, Y_test


def predict(X, X_train,Y_train, return_cov=True, alpha=0):
    assert return_cov
    K = kernel(X_train, X_train)
    L = np.linalg.cholesky(K + alpha * np.eye(X_train.shape[0], X_train.shape[0]))
    a = np.linalg.solve(L, Y_train)
    a = np.linalg.solve(L.T, a)
    f_mean = kernel(X_train, X).T @ a
    v = np.linalg.solve(L, kernel(X_train, X))
    y_cov = kernel(X, X) - v.T @ v + alpha * np.eye(X.shape[0])
    return f_mean, y_cov


def kernel(x1: np.ndarray, x2: np.ndarray, sigma=1, scale=1):
    l1, l2 = x1.shape[0], x2.shape[0]
    Gaussian_Kernel = np.zeros((l1, l2), dtype=float)
    for i in range(l1):
        for j in range(l2):
            Gaussian_Kernel[i, j] = sigma * np.exp(-0.5 * np.sum((x1[i] - x2[j]) ** 2)) / scale
    return Gaussian_Kernel


X_train, Y_train, X_test, Y_test = data_come()
#y_mean, y_cov = predict(X=X_test,X_train=X_train,Y_train=Y_train, return_cov=True)
Km =kernel(X_train,X_test)
print(Km)
