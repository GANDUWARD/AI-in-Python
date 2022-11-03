from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

X, y, true_centers = make_blobs(n_samples=150, centers=3, n_features=2, random_state=0, return_centers=True)
plt.scatter(X[:, 0], X[:, 1])
plt.show()
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.scatter(true_centers[:, 0], true_centers[:, 1], marker='x', alpha=1, c='blue', label='true')
plt.legend()
plt.show()

