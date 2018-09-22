# -*- coding:utf-8 -*-
# author: xiang578
# email: i@xiang578
# blog: www.xiang578.com

import random
from numpy import array, zeros, inner

DATE_FILE = 'hw1_18_train.dat'
DATE_TEST = 'hw1_18_test.dat'


def sign(num):
    if num <= 0:
        return -1
    return 1


def test(x, y, w):
    n = len(y)
    count = 0
    for i in range(n):
        if sign(inner(w, x[i])) != y[i]:
            count = count + 1
    return count/float(n)


def train(x, y, pocket=False):
    n = len(y)
    d = len(x[0])
    w = zeros(d)
    wg = w
    count = test(x, y, wg)
    k = 0
    while k < 50:
        idx = range(n)
        idx = random.sample(idx, n)
        for i in idx:
            if sign(inner(w, x[i])) != y[i]:
                w = w + y[i]*x[i]
                newcount = test(x, y, w)
                if newcount < count:
                    count = newcount
                    wg = w
                break
        k = k + 1
    if pocket:
        return wg
    else:
        return w


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
    xtest, ytest = load_date(DATE_TEST)
    avg = 0.0
    n = 200
    for i in range(n):
        w = train(x, y, pocket=True)
        avg = avg + test(xtest, ytest, w)
        if i % 100 == 0:
            print(str(i))
    print(avg/n)


if __name__ == '__main__':
    pla()  # 0.131609

