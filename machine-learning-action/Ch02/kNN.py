# -*- coding:utf-8 -*-
from numpy import *
import numpy as np
import operator
import matplotlib
import matplotlib.pyplot as plt
from os import listdir


def createdataset():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify0(inx, dataset, labels, k):
    datasetsize = dataset.shape[0]
    diffmat = tile(inx, (datasetsize, 1)) - dataset  # tile([1,2],(3,2))==>[[1, 2, 1, 2], [1, 2, 1, 2], [1, 2, 1, 2]]
    sqdiffmat = diffmat**2
    sqdistances = sqdiffmat.sum(axis=1)
    distances = sqdistances**0.5  # 计算和已知数据的距离

    sorteddistindicies = distances.argsort()  # 将元素从小到大排序，返回索引的序列
    classcount = {}  # 记录结果
    for i in range(k):
        voteilabel = labels[sorteddistindicies[i]]
        classcount[voteilabel] = classcount.get(voteilabel, 0) + 1  # get(key, default) 返回value，不存在时返回 default
    sortedclasscount = sorted(classcount.iteritems(), key=operator.itemgetter(1), reverse=True)  # 取 value 值排序
    return sortedclasscount[0][0]


def test1():
    group, labels = createdataset()
    print(group)
    print(labels)
    print(classify0([0.1, 0.1], group, labels, 3))


def file2matrix(filename):
    fr = open(filename)
    arrayoflines = fr.readlines()
    numberoflines = len(arrayoflines)
    returnmat = zeros((numberoflines, 3))
    classlabelvalue = []
    index = 0
    for line in arrayoflines:
        line = line.strip()
        listfromline = line.split('\t')
        returnmat[index, :] = listfromline[0:3]
        classlabelvalue.append(int(listfromline[-1]))  # 转化成为int
        index = index + 1
    return returnmat, classlabelvalue


def autonorm(dataset):
    minvals = dataset.min(0)
    maxvals = dataset.max(0)
    ranges = maxvals - minvals
    normdataset = zeros(shape(dataset))
    m = dataset.shape[0]
    normdataset = dataset - tile(minvals, (m, 1))
    normdataset = normdataset/tile(ranges, (m, 1))
    return normdataset, ranges, minvals


def datingclasstest1():
    datingdatamat, datinglabels = file2matrix('datingTestSet2.txt')
    fig = plt.figure()
    ax = fig.add_subplot(111)  # 画子图
    ax.scatter(datingdatamat[:, 0], datingdatamat[:, 1], 15.0*np.array(datinglabels), 15.0*np.array(datinglabels))
    plt.show()


def datingclasstest2():
    horatio = 0.10
    datingdatamat, datinglabels = file2matrix('datingTestSet2.txt')
    normmat, ranges, minvals = autonorm(datingdatamat)  # 归一化数值
    m = normmat.shape[0]
    numtestvecs = int(m*horatio)
    errorcount = 0.0
    for i in range(numtestvecs):
        result = classify0(normmat[i, :], normmat[numtestvecs:m, :], datinglabels[numtestvecs:m], 4)
        print("the classifier came back with: %d, the real answer is: %d" % (result, datinglabels[i]))
        if (result != datinglabels[i]):
            errorcount += 1.0
    print("the total error rate is : %f" % (errorcount/float(numtestvecs)))


def classfyperson():
    resultlist = ['not at all', 'is small doses', 'in large doses']
    percenttats = float(raw_input("percentage of time spent playing video games ?"))
    ffmiles = float(raw_input("frequent filer miles earned per year?"))
    icecream = float(raw_input("liters of ice cream consumed per year?"))
    datingdatamat, datinglabels = file2matrix('datingTestSet2.txt')
    normmat, ranges, minvals = autonorm(datingdatamat)  # 归一化数值
    inarr = array([ffmiles, percenttats, icecream])
    classfypersonresult = classify0((inarr-minvals)/ranges, normmat, datinglabels, 3)
    print "You will probably like this person: ", resultlist[classfypersonresult - 1]


def img2vector(filename):
    returnvec = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        linestr = fr.readline()
        for j in range(32):
            returnvec[0, 32*i+j] = int(linestr[j])
    return returnvec


def handwritingclasstest():
    hwlabels = []
    trainingfilelist = listdir('digits/trainingDigits')
    m = len(trainingfilelist)
    trainingmat = zeros((m, 1024))
    for i in range(m):
        filenamestr = trainingfilelist[i]
        filestr = filenamestr.split('.')[0]
        classnumstr = int(filestr.split('_')[0])
        hwlabels.append(classnumstr)
        trainingmat[i, :] = img2vector('digits/trainingDigits/%s' % filenamestr)
    testfilelist = listdir('digits/testDigits')
    mtest = len(testfilelist)
    errorcount = 0.0
    for i in range(mtest):
        testfilenamestr = testfilelist[i]
        testfilestr = testfilenamestr.split('.')[0]
        testclassnumstr = int(testfilestr.split('_')[0])
        testvect = img2vector('digits/testDigits/%s' % testfilenamestr)
        result = classify0(testvect, trainingmat, hwlabels, 3)
        if (testclassnumstr != result):
            errorcount += 1
        print 'the classifier came back with: %d, the real number is: %d.\n' % (result, testclassnumstr)
    print 'the total error is %d, the total error rate is: %f\n' % (errorcount, errorcount/float(mtest))


if __name__ == '__main__':
    # test1()
    # datingclasstest1()
    # datingclasstest2()  # the total error rate is : 0.040000
    # classfyperson()  # 10 10000 0.5 is small doses
    handwritingclasstest()
