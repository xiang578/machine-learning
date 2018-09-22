# -*- coding:utf-8 -*-
# author: RyanXiang
# email: i@xiang578
# blog: www.xiang578.com
from numpy import *


def load_data_set():
    data_mat = []
    label_mat = []
    fp = open('testSet.txt', 'r')
    for line in fp.readlines():
        x = line.strip().split()
        data_mat.append([1.0, float(x[0]), float(x[1])])
        label_mat.append(int(x[-1]))
    return data_mat, label_mat


def sigmoid(inx):
    return 1.0/(1+exp(-inx))


def classify_vector(inx, weights):
    prob = sigmoid(sum(inx * weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0


def grad_ascent(data_mat_in, class_labels):
    data_matrix = mat(data_mat_in)
    label_matrix = mat(class_labels).transpose()
    m, n = shape(data_matrix)
    alpha = 0.001
    max_cycles = 30000
    weights = ones((n, 1))
    for k in range(max_cycles):
        h = sigmoid(data_matrix * weights)
        error = (label_matrix - h)
        weights = weights + alpha * data_matrix.transpose() * error
    return weights


def stoc_grad_ascent0(data_matrix, class_labels):
    m, n = shape(data_matrix)
    alpha = 0.01
    weights = ones(n)
    for i in range(m):
        h = sigmoid(sum(data_matrix[i]*weights))
        error = class_labels[i] - h
        weights = weights + alpha * error * data_matrix[i]
    return weights


def stoc_grad_ascent1(data_matrix, class_labels, num_count=150):
    m, n = shape(data_matrix)
    weights = ones(n)
    for j in range(num_count):
        data_index = range(m)
        for i in range(m):
            alpha = 4/(1.0 + j + i) + 0.01
            rand_index = int(random.uniform(0, len(data_index)))
            h = sigmoid(sum(data_matrix[rand_index]*weights))
            error = class_labels[rand_index] - h
            weights = weights + alpha * error * data_matrix[rand_index]
            del(data_index[rand_index])
    return weights


def t_grad_ascent():
    data_mat, label_mat = load_data_set()
    print grad_ascent(data_mat, label_mat)


def t_stoc_grad_ascent0():
    data_mat, label_mat = load_data_set()
    weights = stoc_grad_ascent1(array(data_mat), label_mat)
    print weights
    plot_best_fit(weights)


def plot_best_fit(weights):
    import matplotlib.pyplot as plt
    data_mat, label_mat = load_data_set()
    # print data_mat
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
    x = arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1]*x)/weights[2]
    # print shape(x)
    # print shape(y)
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


def t_plot_best_fit():
    data_mat, label_mat = load_data_set()
    weights = grad_ascent(data_mat, label_mat)
    print weights
    plot_best_fit(weights.getA())


def colic_test():
    fp = open('horseColicTraining.txt', 'r')
    training_set = []
    training_label = []
    for line in fp.readlines():
        line = line.strip().split('\t')
        line_arr = []
        for i in line:
            line_arr.append(float(i))
        training_set.append(line_arr[:-1])
        training_label.append(line_arr[-1])
    # training_weights = grad_ascent(array(training_set), array(training_label))
    training_weights = stoc_grad_ascent1(array(training_set), training_label, 500)
    # plot_best_fit(training_weights)
    error_count = 0.
    num_test = 0
    fp_test = open('horseColicTest.txt')
    for line in fp_test.readlines():
        line = line.strip().split('\t')
        if len(line) != 22:
            continue
        num_test += 1
        line_arr = [float(i) for i in line]
        label = classify_vector(array(line_arr[:-1]), training_weights)
        if label != line_arr[-1]:
            error_count += 1
    fp_test.close()
    fp.close()
    print "the error rate is: %f" % (error_count/float(num_test))
    return error_count/float(num_test)


def multi_test():
    num_tests = 10
    error_sum = 0.0
    for k in range(num_tests):
        error_sum += colic_test()
    print "after %d iterations the averag error rate is: %f" % (num_tests, error_sum/float(num_tests))


def log_for_ex4():
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
    # weights = stoc_grad_ascent1(array(data_mat), label_mat, 500)
    weights = grad_ascent(array(data_mat), array(label_mat))
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
    y = (-weights.getA()[0] - weights.getA()[1] * x) / weights.getA()[2]
    # y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


if __name__ == '__main__':
    # t_grad_ascent()
    # t_plot_best_fit()
    # t_stoc_grad_ascent0()
    # multi_test()
    # colic_test()
    log_for_ex4()
