# 第 5 章：Logistics 回归

本章节用到了一些数学原理，但是书中直接一带而过。对像我这样的新手而言没有很大的挑战。

## 基础概念
 
 - 回归：假设有一些数据点，我们利用一条直线对这些点进行拟合，这个过程就是回归。
 - Logistics 回归：根据现有数据对分类边界线建立回归公式，以此进行分类。
 - Sigmoid 函数：${\sigma(z)=\frac{1}{1+e^{-z}}}$
    - Logistics 回归是对数据进行二值分类，Sigmoid 函数的输出值在 0~1 的范围之间，利用这个特性，把结果大于 0.5 的数据分入 1 类，小于 0.5 的归入 0 类。
    - 问题中，假设输入的数据形式为 ${Z=W^TX}$，其中 X 为输入，W 为回归系数

## 梯度上升法 

在上面的描述中，我们得到了回归系数这个概念。接下来，我们需要确定最佳的回归系数。可能这本书定位是实战，所以不太注重推导，详细的推到过程和原理可以参考下面的链接1。这里只给出相对于的 W 更新公式：
$${W:=W+	\alpha \nabla_wf(W)=W+\alpha X^T(Y-\frac{1}{1+e^{-W^TX}})}$$
其中$${\nabla_wf(W)}$$ 为梯度。

梯度下降相关算法
```python
def sigmoid(inx):
    return 1.0/(1+exp(-inx))


def grad_ascent(data_mat_in, class_labels):
    data_matrix = mat(data_mat_in)
    label_matrix = mat(class_labels).transpose()
    m, n = shape(data_matrix)
    alpha = 0.001
    max_cycles = 500
    weights = ones((n, 1))
    for k in range(max_cycles):
        h = sigmoid(data_matrix * weights)
        error = (label_matrix - h)
        weights = weights + alpha * data_matrix.transpose() * error
    return weights
```

由于梯度下降算法，每一次迭代时，需要计算全部的数值，所以复杂度比较高，书中又给出了一种在线算法-随机梯度下降法，主要思路是每次只使用一个测试数据更新 W 。

```python
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
```

## 预测病马的死亡率

讲这个例子的时候，作者还介绍了对一些缺省数据处理的技巧。比如，对于训练数据，可以进行一些适当的补全，但是对于测试数据权少标签时，建议直接放弃。
在联系这个例子时，我将训练得到的结果放到前一步完成的画图函数时，发现图像很不正常。仔细思考了一下才明白，之前是给二维数据分类训练，这一个例子的输入数据足足有 21 维。

```python
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
```
## 参考
1. [【机器学习笔记1】Logistic回归总结 - CSDN博客](https://blog.csdn.net/achuo/article/details/51160101)
2. [MachineLearning/5.Logistic回归.md at master · apachecn/MachineLearning](https://github.com/apachecn/MachineLearning/blob/master/docs/5.Logistic%E5%9B%9E%E5%BD%92.md)


