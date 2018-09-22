# -*- coding:utf-8 -*-
# author: RyanXiang
# email: i@xiang578
# blog: www.xiang578.com
from numpy import *


def sigmoid(inx):
    return 1.0/(1.0+exp(-inx))


def stoc_grad_ascent1(data_matrix, class_labels, num_count=150):
    m, n = shape(data_matrix)
    weights = ones(n)
    for j in range(num_count):
        data_index = range(m)
        for i in range(m):
            alpha = 4/(1.0 + j + i) + 0.01
            # alpha = 0.001
            rand_index = int(random.uniform(0, len(data_index)))
            h = sigmoid(sum(data_matrix[rand_index]*weights))
            error = class_labels[rand_index] - h
            weights = weights + alpha * error * data_matrix[rand_index]
            print weights
            del(data_index[rand_index])
    return weights


if __name__ == '__main__':
    fp_data = open('ex4Data/ex4x.dat')
    fp_label = open('ex4Data/ex4y.dat')
    data_mat = []
    label_mat = []
    for line in fp_data:
        line = line.strip().split()
        data_mat.append([1.0, float(line[0]), float(line[1])])
    fp_data.close()
    for line in fp_label:
        line = line.strip().split()
        label_mat.append(float(line[0]))
    print label_mat
    weights = stoc_grad_ascent1(array(data_mat), label_mat, 50)
    error = 0
    # weights = grad_ascent(array(data_mat), array(label_mat))
    import matplotlib.pyplot as plt

    data_arr = array(data_mat)
    n = shape(data_arr)[0]
    x_cord_1 = []
    y_cord_1 = []
    x_cord_2 = []
    y_cord_2 = []
    for i in range(n):
        if int(label_mat[i]) == 1:
            x_cord_1.append(data_arr[i][1])
            y_cord_1.append(data_arr[i][2])
        else:
            x_cord_2.append(data_arr[i][1])
            y_cord_2.append(data_arr[i][2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x_cord_1, y_cord_1, s=30, c='red', marker='s')
    ax.scatter(x_cord_2, y_cord_2, s=30, c='green')
    x = arange(10, 70, 1)
    print weights
    # y = (-weights.getA()[0] - weights.getA()[1] * x) / weights.getA()[2]
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()
