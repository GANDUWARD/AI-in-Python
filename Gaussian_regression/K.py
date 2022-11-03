import numpy as np
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process import GaussianProcessRegressor
import matplotlib.pyplot as plt

class Kernel:
    def calculate_kernel(self, sca_l, train_X, train_y, test_X):
        ker = RBF(length_scale=sca_l, length_scale_bounds='fixed')
        gpr = GaussianProcessRegressor(kernel=ker, n_restarts_optimizer=2, normalize_y=False)
        gpr.fit(train_X, train_y)
        mu, cov = gpr.predict(test_X, return_cov=True)
        test_y = mu.ravel()
        uncertainty = 1.96 * np.sqrt(np.diag(cov))
        plt.figure()
        plt.title("GPR")
        plt.fill_between(test_X.ravel(), test_y + uncertainty, test_y - uncertainty, alpha=0.1)
        plt.plot(test_X, test_y, label="predict")
        plt.scatter(train_X, train_y, label="train", c="red", marker="x")
        plt.legend()
        plt.show()


