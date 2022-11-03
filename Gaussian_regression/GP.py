import numpy as np
import Kernel
import random


class Gaussian_Progress:
    def __init__(self, Kernel, sigma, scale, alpha=0.):
        self.kernel = Kernel.Kernel(sigma, scale)
        self.alpha = alpha
        X_train, Y_train, X_test, Y_test=self.data_come()
        self.X_train = X_train
        self.X_test = X_test
        self.Y_train = Y_train
        self.Y_train = Y_train

    def get_data(self,fileplace):
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

    def data_come(self):
        fp = open('train.TXT')
        fp2 = open('test.TXT')
        x_train, y_train = self.get_data(fp)
        X_train = np.array(x_train).reshape(100, 1)
        Y_train = np.array(y_train).reshape(100, 1)
        x_test, y_test = self.get_data(fp2)
        X_test = np.array(x_test).reshape(100, 1)
        Y_test = np.array(y_test).reshape(100, 1)
        return X_train,Y_train,X_test,Y_test

    def Predict(self, X, return_cov=True):
        assert return_cov
        K = self.kernel(self.X_train, self.X_train)
        L = np.linalg.cholesky(K + self.alpha * np.eye(self.X_train.shape[0], self.X_train.shape[0]))
        a = np.linalg.solve(L, self.Y_train)
        a = np.linalg.solve(L.T, a)
        f_mean= Kernel.Kernel(self.X_train, X).T @a
        v = np.linalg.solve(L, self.kernel(self.X_train, X))
        y_cov = self.kernel(X, X) - v.T @ v + self.alpha * np.eye(X.shape[0])
        return f_mean, y_cov

    def get_sample(self, X, n_samples=1):
        y_mean, y_cov = self.Predict(X, return_cov=True)
        sampled_y =np.random.multivariate_normal(y_mean, y_cov, size=n_samples)
        return sampled_y



