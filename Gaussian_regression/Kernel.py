import numpy as np


class Kernel:
    def __init__(self, sigma, scale):
        self.sigma = sigma
        self.scale = scale

    def __call__(self, x1:np.ndarray, x2:np.ndarray):
        l1, l2 = x1.shape[0], x2.shape[0]
        Gaussian_Kernel = np.zeros((l1, l2), dtype=float)
        for i in range(l1):
            for j in range(l2):
                Gaussian_Kernel[i, j] = self.sigma* np.exp(-0.5*np.sum((x1[i] - x2[j]) ** 2 ))/ self.scale
        return Gaussian_Kernel
