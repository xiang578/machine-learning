# -*- coding:utf-8 -*-
# author: xiang578
# email: i@xiang578
# blog: www.xiang578.com

import random
from numpy import array, zeros, inner

DATE_FILE = 'hw1_15_train.dat'


def sign(num):
    if num <= 0:
        return -1
    return 1


def train(x, y):
    n = len(y)
    d = len(x[0])
    w = zeros(d)
    count = 0
    idx = range(n)
    idx = random.sample(idx, n)
    while True:
        ok = True
        for i in range(n):
            j = idx[i]
            if sign(inner(w, x[j])) != y[j]:
                w = w + y[j]*x[j]
                ok = False
                count = count + 1
        if ok:
            break
    return count


def load_date(filename):
    x = []
    y = []
    f = open(filename)
    for line in f:
        res = line.split()
        t = [1] + [float(v) for v in res[0:4]]
        x.append(tuple(t))
        y.append(int(res[-1]))
    return array(x), array(y)


def pla():
    x, y = load_date(DATE_FILE)
    count = 0
    for i in range(2000):
        count = count + train(x, y)
    print(count/2000)


if __name__ == '__main__':
    pla()  # 40

