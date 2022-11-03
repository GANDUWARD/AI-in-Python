from matplotlib import pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import numpy as np
import classification

if __name__ == '__main__':
    plt_class = classification.plot
    plt_class.__init__(plt_class)
    plt_class.data_processor(plt_class)
    plt_class.KNN(plt_class)
    plt_class.get_L(plt_class)
