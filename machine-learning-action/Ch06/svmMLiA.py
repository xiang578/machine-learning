# -*- coding:utf-8 -*-
# author: Ryan Xiang
# email: i@xiang578
# blog: www.xiang578.com
import random
from numpy import *


def plot_best_fit(data_mat, label_mat, b, w, alphas):
    import matplotlib.pyplot as plt
    data_arr = array(data_mat)
    # print data_arr
    n = shape(data_arr)[0]
    x_cord_1 = []
    y_cord_1 = []
    x_cord_2 = []
    y_cord_2 = []
    for i in range(n):
        if int(label_mat[i]) == 1:
            x_cord_1.append(data_arr[i][0])
            y_cord_1.append(data_arr[i][1])
        else:
            x_cord_2.append(data_arr[i][0])
            y_cord_2.append(data_arr[i][1])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x_cord_1, y_cord_1, s=30, c='red', marker='s')
    ax.scatter(x_cord_2, y_cord_2, s=30, c='green')
    x = arange(-2.0, 10.0, 0.1)
    y = (-w[0, 0] * x - array(b)[0])/(w[1, 0])
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    for i in range(n):
        if alphas[i] > 0.0:
            ax.plot(data_arr[i][0], data_arr[i][1], 'bo')
    plt.show()


def load_data_set(file_name):
    fp = open(file_name)
    data_mat = []
    label_mat = []
    for line in fp.readlines():
        tmp = line.strip().split()
        data_mat.append([float(tmp[0]), float(tmp[1])])
        label_mat.append(float(tmp[-1]))
    return data_mat, label_mat


def select_j_rand(i, m):
    j = i
    while j == i:
        j = int(random.uniform(0, m))
    return j


def clip_alpha(aj, h, l):
    if aj > h:
        aj = h
    if l > aj:
        aj = l
    return aj


# 数据集，类别标签，常数c，容错率和最大循环次数
def smo_simple(data_mat_in, class_labels, c, toler, max_iter):
    data_matrix = mat(data_mat_in)
    label_mat = mat(class_labels).transpose()
    b = 0
    m, n = shape(data_matrix)
    alphas = mat(zeros((m, 1)))
    iter = 0
    while iter < max_iter:
        alpha_pairs_change = 0
        for i in range(m):
            fxi = float(multiply(alphas, label_mat).T * (data_matrix * data_matrix[i, :].T)) + b
            ei = fxi - float(label_mat[i])
            if ((label_mat[i] * ei < -toler) and (alphas[i] < c)) or ((label_mat[i] * ei > toler) and (alphas[i] > 0)):
                j = select_j_rand(i, m)
                fxj = float(multiply(alphas, label_mat).T * (data_matrix * data_matrix[j, :].T)) + b
                ej = fxj - float(label_mat[j])
                alpha_i_old = alphas[i].copy()
                alpha_j_old = alphas[j].copy()
                if label_mat[i] != label_mat[j]:
                    le = max(0, alphas[j] - alphas[i])
                    he = min(c, c + alphas[j] - alphas[i])
                else:
                    le = max(0, alphas[j] + alphas[i] - c)
                    he = min(c, alphas[j] + alphas[i])
                if le == he:
                    print "l=h"
                    continue
                eta = 2.0 * data_matrix[i, :] * data_matrix[j, :].T - data_matrix[i, :] * data_matrix[i, :].T \
                      - data_matrix[j, :] * data_matrix[j, :].T
                if eta >= 0:
                    print "eta>=0"
                    continue
                alphas[j] -= label_mat[j]*(ei - ej)/eta
                alphas[j] = clip_alpha(alphas[j], he, le)
                if abs(alphas[j] - alpha_j_old) < 0.00001:
                    print "j not moving enough"
                    continue
                alphas[i] += label_mat[j]*label_mat[i]*(alpha_j_old-alphas[j])
                b1 = b - ei - label_mat[i]*(alphas[i]-alpha_i_old)*data_matrix[i, :] * data_matrix[i, :].T - \
                    label_mat[j] * (alphas[j] - alpha_j_old) * data_matrix[i, :]*data_matrix[j, :].T
                b2 = b - ej - label_mat[i]*(alphas[i]-alpha_i_old)*data_matrix[i, :] * data_matrix[j, :].T - \
                    label_mat[j] * (alphas[j] - alpha_j_old) * data_matrix[j, :]*data_matrix[j, :].T
                if (0 < alphas[i]) and (c > alphas[i]):
                    b = b1
                elif (0 < alphas[j]) and (c > alphas[j]):
                    b = b2
                else:
                    b = (b1+b2)/2.0
                alpha_pairs_change += 1
                print "iter: %d i: %d, pairs changed %d" % \
                      (iter, i, alpha_pairs_change)
                # print alphas
                print b
        if alpha_pairs_change == 0:
            iter += 1
        else:
            iter = 0
        print "iteration number: %d" % iter
    return b, alphas


def calc_w(alphas, data_arr, label_arr):
    x = mat(data_arr)
    label_mat = mat(label_arr).transpose()
    m, n = shape(x)
    w = zeros((n, 1))
    for i in range(m):
        w += multiply(alphas[i] * label_arr[i], x[i, :].T)
    return w


def t_smo_simple(data_arr, label_arr):
    b, alphas = smo_simple(data_arr, label_arr, 0.6, 0.001, 40)
    w = calc_w(alphas, data_arr, label_arr)
    plot_best_fit(data_arr, label_arr, b, w, alphas)


if __name__ == "__main__":
    data_arr, label_arr = load_data_set('testSet.txt')
    t_smo_simple(data_arr, label_arr)
