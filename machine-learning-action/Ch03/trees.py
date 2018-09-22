# -*- coding:utf-8 -*-
# author: xiang578
# email: i@xiang578
# blog: www.xiang578.com

from math import log
import operator
import matplotlib.pyplot as plt


def calc_shannon_ent(dataset):
    numentries = len(dataset)
    labelcounts = {}
    for featvec in dataset:
        currentlabel = featvec[-1]
        if currentlabel not in labelcounts.keys():
            labelcounts[currentlabel] = 0
        labelcounts[currentlabel] += 1  # 统计标签的种类以及数量
    shannonent = 0.0
    for key in labelcounts:
        prob = float(labelcounts[key])/numentries
        shannonent -= prob * log(prob, 2)  # 计算熵
    return shannonent


def creat_dataset():
    dataset = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataset, labels


def test_shannon_ent():
    dataset, labels = creat_dataset()
    dataset[0][-1] = 'maybe'  # 0.97 and 1.37 混合的数据越多，信息量越大
    ans = calc_shannon_ent(dataset)
    print ans


def split_dataset(dataset, axis, value):
    retdataset = []  # 划分之后的子集
    for data in dataset:
        if data[axis] == value:
            reducefeatvec = data[:axis]  # 去除 axis 相关的特征
            reducefeatvec.extend(data[axis+1:])
            retdataset.append(reducefeatvec)
    return retdataset


def test_split_dataset():
    mydat, labels = creat_dataset()
    print mydat
    print split_dataset(mydat, 0, 1)
    print split_dataset(mydat, 0, 0)


def choose_best_feature_to_split(dataset):
    numfeatures = len(dataset[0]) - 1  # 每一条数据的特征数，最后一个为标签
    baseentropy = calc_shannon_ent(dataset)  # 计算初始数据中的熵
    bestinfogain = 0.0
    bestfeature = -1
    for idx in range(numfeatures):
        featlist = [data[idx] for data in dataset]  # 列表推导，计算出某个特征的全部值范围，并在下一步去重
        uniquevals = set(featlist)
        newentropy = 0.0
        for value in uniquevals:
            subdataset = split_dataset(dataset, idx, value)
            prob = len(subdataset)/float(len(dataset))
            newentropy += prob * calc_shannon_ent(subdataset)
        infogain = baseentropy - newentropy
        # print baseentropy, newentropy
        if (infogain > bestinfogain):
            bestinfogain = infogain
            bestfeature = idx
    return bestfeature


def teset_choose_best_feature_to_split():
    mydat, labels = creat_dataset()
    print choose_best_feature_to_split(mydat)


def majoritycnt(classlist):
    classcount = {}
    for vote in classlist:
        if vote not in classcount.keys():
            classcount[vote] = 0
        classcount[vote] += 1
    sortedclasscount = sorted(classcount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedclasscount[0][0]


def create_tree(dataset, labels):
    classlist = [example[-1] for example in dataset]  # 统计数据集中的标签信息
    if classlist.count(classlist[0]) == len(dataset):  # 结束递归条件：数据集中的标签全部相同，或者数据集中只剩下一个数据
        return classlist[0]
    if len(dataset[0]) == 1:
        return majoritycnt(classlist)
    bestfeature = choose_best_feature_to_split(dataset)  # 选择分类的标签
    bestfeaturelabel = labels[bestfeature]
    mytree = {bestfeaturelabel: {}}  # 利用mytree保存结果
    del(labels[bestfeature])
    featvalues = [example[bestfeature] for example in dataset]
    unqiuevals = set(featvalues)
    for value in unqiuevals:  # 利用特征的取值范围划分本层
        sublabels = labels[:]
        mytree[bestfeaturelabel][value] = create_tree(split_dataset(dataset, bestfeature, value), sublabels)
    return mytree


def test_create_tree():
    mydat, labels = creat_dataset()
    mytree = create_tree(mydat, labels)
    print mytree


def classify(inputtree, featlabels, testvec):
    firststr = inputtree.keys()[0]
    seconddict = inputtree[firststr]
    # print firststr, featlabels
    idx = featlabels.index(firststr)
    key = testvec[idx]
    if type(seconddict[key]).__name__ == 'dict':
        label = classify(seconddict[key], featlabels, testvec)
    else:
        label = seconddict[key]
    return label


def test_classify():
    mydat, labels = creat_dataset()
    vlabels = labels[:]  # 深拷贝
    mytree = create_tree(mydat, vlabels)
    # labels = ['no surfacing', 'flippers']
    print mytree
    print classify(mytree, labels, [1,0])
    print classify(mytree, labels, [1,1])


def store_tree(inputtree, filename):
    import pickle
    fw = open(filename, 'w')
    pickle.dump(inputtree,fw)
    fw.close()


def grab_tree(filename):
    import pickle
    fr = open(filename)
    return pickle.load(fr)


def t_pickle():
    mydat, labels = creat_dataset()
    mytree = create_tree(mydat, labels)
    store_tree(mytree, 'clasaifyierStorage.txt')
    print grab_tree('clasaifyierStorage.txt')


def t_Lenses():
    fr = open('lenses.txt')
    lenses = []
    for inst in fr.readlines():
        line = inst.strip()
        tmp = inst.strip().split('\t')
        lenses.append(tmp)
    # lenses = [inst.strip('\n').split('\t') for inst in fr.readline()]
    # print lenses
    lenseslabels = ['age', 'prescript', 'astigmatic', 'tearrate']
    lensestree = create_tree(lenses, lenseslabels)
    print lensestree


if __name__ == '__main__':
    #  test_shannon_ent()
    #  test_split_dataset()
    #  teset_choose_best_feature_to_split()
    #  test_create_tree()
    #  test_classify()
    #  t_pickle()
    t_Lenses()
