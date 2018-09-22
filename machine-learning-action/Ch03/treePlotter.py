# -*- coding:utf-8 -*-
# author: xiang578
# email: i@xiang578
# blog: www.xiang578.com


import matplotlib.pyplot as plt


# 定义画图相关的属性
decisonnode = dict(boxstyle="sawtooth", fc='0.8')
leafnode = dict(boxstyle='round4', fc='0.8')
arrow_args = dict(arrowstyle='<-')


def plot_node(nodetext, centerpt, parentpt, nodetype):
    create_plot.ax1.annotate(nodetext, xy=parentpt, xycoords='axes fraction', xytext=centerpt,
                             textcoords='axes fraction',
                             va="center", ha="center", bbox=nodetype, arrowprops=arrow_args)


def create_plot():
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    create_plot.ax1 = plt.subplot(111, frameon=False)  # 定义一个函数的属性，属于全局变量
    plot_node('decisonnode', (0.5, 0.1), (0.1, 0.5), decisonnode)
    plot_node('leafnode', (0.8, 0.1), (0.3, 0.8), leafnode)
    plt.show()


def test_create_plot():
    create_plot()


def retrieve_tree(i):
    istoftrees = [{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}},
                  {'no surfacing': {0: 'no', 1: {'flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}
                  ]
    return istoftrees[i]


def get_num_leafs(mytree):
    numleafs = 0
    firststr = mytree.keys()[0]
    seconddict = mytree[firststr]
    for key in seconddict.keys():
        if type(seconddict[key]).__name__ == 'dict':
            numleafs += get_num_leafs(seconddict[key])
        else:
            numleafs += 1
    return numleafs


def get_tree_depth(mytree):
    maxdepth = 0
    firststr = mytree.keys()[0]
    seconddict = mytree[firststr]
    for key in seconddict.keys():
        if type(seconddict[key]).__name__ == 'dict':
            thisdepth = 1 + get_tree_depth(seconddict[key])
        else:
            thisdepth = 1
        if thisdepth > maxdepth:
            maxdepth = thisdepth
    return maxdepth


def test_get():
    my_tree = retrieve_tree(0)
    print my_tree
    print get_tree_depth(my_tree)
    print get_num_leafs(my_tree)


if __name__ == '__main__':
    # test_create_plot()
    test_get()
