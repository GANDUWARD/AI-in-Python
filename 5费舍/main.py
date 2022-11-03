import numpy as np
import matplotlib.pyplot as plt


def create_data(size=50, add_outlier=False, add_class=False):
    assert size % 2 == 0
    x0 = np.random.normal(size=size).reshape(-1, 2) - 1
    x1 = np.random.normal(size=size).reshape(-1, 2) + 1
    if add_outlier:
        x = np.random.normal(size=10).reshape(-1, 2) + np.array([5, 10])
        return np.concatenate([x0, x1, x]), np.concatenate([np.zeros(size // 2), np.ones(size // 2 + 5)])
    if add_class:
        x = np.random.normal(size=size).reshape(-1, 2) + 3
        return np.concatenate([x0, x1, x]), np.concatenate(
            [np.zeros(size // 2), np.ones(size // 2), 2 * np.ones(size // 2)])
    return np.concatenate([x0, x1]), np.concatenate([np.zeros(size // 2), np.ones(size // 2)])


def data_transform(dots, labels):
    x1 = dots[labels == 0]
    x2 = dots[labels == 1]
    return x1, x2


def calculate_cov_and_avg(samples):
    u = np.mean(samples, axis=0)
    cov_m = np.zeros((samples.shape[1], samples.shape[1]))
    for s in samples:
        t = s - u
        cov_m += t * t.reshape(2, 1)
    return cov_m, u


def fisher(c_1, c_2):
    cov_1, u1 = calculate_cov_and_avg(c_1)
    cov_2, u2 = calculate_cov_and_avg(c_2)
    s_w = cov_1 + cov_2
    u, s, v = np.linalg.svd(s_w)
    s_w_inv = np.dot(np.dot(v.T, np.linalg.inv(np.diag(s))), u.T)
    return np.dot(s_w_inv, u1 - u2)


def judge(sample, w, c_1, c_2):
    u1 = np.mean(c_1, axis=0)
    u2 = np.mean(c_2, axis=0)
    center_1 = np.dot(w.T, u1)
    center_2 = np.dot(w.T, u2)
    pos = np.dot(w.T, sample)
    return abs(pos - center_1) < abs(pos - center_2)


def draw_picture(c_1, c_2, w):
    plt.clf()
    plt.scatter(c_1[:, 0], c_1[:, 1], c='#99CC99')
    plt.scatter(c_2[:, 0], c_2[:, 1], c='#FFCC00')
    line_x = np.arange(min(np.min(c_1[:, 0]), np.min(c_2[:, 0])),
                       max(np.max(c_1[:, 0]), np.max(c_2[:, 0])),
                       step=1)

    line_y = - (w[0] * line_x) / w[1]
    plt.plot(line_x, line_y)
    plt.show()


d1, d2 = create_data()
c_1, c_2 = data_transform(d1, d2)
w = fisher(c_1, c_2)
draw_picture(c_1, c_2, w)
x1_test, x2_test = np.meshgrid(np.linspace(-5, 5, 100), np.linspace(-5, 5, 100))
x_test = np.concatenate([x1_test, x2_test]).reshape(2, -1)
