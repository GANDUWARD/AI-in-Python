import matplotlib.pyplot as plt


def write(file_place):               #写入函数
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


def classify(x, y):
    y_new = []
    for i in y:
        if 1 <= i <= 7:
            y_new.append(4)
        if 8 <= i <= 14:
            y_new.append(11)
        if 15 <= i <= 21:
            y_new.append(18)
        if 22 <= i <= 28:
            y_new.append(25)
        if 29 <= i <= 35:
            y_new.append(32)
        if 36 <= i <= 42:
            y_new.append(39)
        if 43 <= i <= 47:
            y_new.append(46)
        return y_new


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
