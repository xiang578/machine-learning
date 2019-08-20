""" Starter code for simple logistic regression model for MNIST
with tf.data module
MNIST dataset: yann.lecun.com/exdb/mnist/
Created by Chip Huyen (chiphuyen@cs.stanford.edu)
CS20: "TensorFlow for Deep Learning Research"
cs20.stanford.edu
Lecture 03
"""
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'

import numpy as np
import tensorflow as tf
import time
import utils
import math

# GPU
gpuConfig = tf.ConfigProto(allow_soft_placement=True)
gpuConfig.gpu_options.allow_growth = True
gpuConfig.log_device_placement = False
# Define paramaters for the model
learning_rate = 0.01
batch_size = 128
n_epochs = 1000
n_train = 60000
n_test = 10000

# Step 1: Read in data
mnist_folder = 'data/mnist'
utils.download_mnist(mnist_folder)
train, val, test = utils.read_mnist(mnist_folder, flatten=True)

# Step 2: Create datasets and iterator
# create training Dataset and batch it
train_data = tf.data.Dataset.from_tensor_slices(train)
train_data = train_data.shuffle(10000)  # if you want to shuffle your data
train_data = train_data.batch(batch_size)

# create testing Dataset and batch it
test_data = tf.data.Dataset.from_tensor_slices(test)
test_data = test_data.batch(batch_size)
#############################
########## TO DO ############
#############################


# create one iterator and initialize it with different datasets
iterator = tf.data.Iterator.from_structure(train_data.output_types,
                                           train_data.output_shapes)
img, label = iterator.get_next()
train_init = iterator.make_initializer(train_data)  # initializer for train_data
test_init = iterator.make_initializer(test_data)  # initializer for train_data

# Step 3: create weights and bias
# w is initialized to random variables with mean of 0, stddev of 0.01
# b is initialized to 0
# shape of w depends on the dimension of X and Y so that Y = tf.matmul(X, w)
# shape of b depends on Y
# n1 = 400
# n2 = 200
# n3 = 100
# n4 = 60
# n5 = 30
# n6 = 10
# pkeep = 0.8
#
#
# w1 = tf.get_variable("w1", shape=(784, n1), initializer=tf.random_normal_initializer(0, 0.01))
# w2 = tf.get_variable("w2", shape=(n1, n2), initializer=tf.random_normal_initializer(0, 0.01))
# w3 = tf.get_variable("w3", shape=(n2, n3), initializer=tf.random_normal_initializer(0, 0.01))
# w4 = tf.get_variable("w4", shape=(n3, n4), initializer=tf.random_normal_initializer(0, 0.01))
# w5 = tf.get_variable("w5", shape=(n4, n5), initializer=tf.random_normal_initializer(0, 0.01))
# w6 = tf.get_variable("w6", shape=(n5, n6), initializer=tf.random_normal_initializer(0, 0.01))
#
# b1 = tf.get_variable("b1", shape=(1, n1), initializer=tf.zeros_initializer)
# b2 = tf.get_variable("b2", shape=(1, n2), initializer=tf.zeros_initializer)
# b3 = tf.get_variable("b3", shape=(1, n3), initializer=tf.zeros_initializer)
# b4 = tf.get_variable("b4", shape=(1, n4), initializer=tf.zeros_initializer)
# b5 = tf.get_variable("b5", shape=(1, n5), initializer=tf.zeros_initializer)
# b6 = tf.get_variable("b6", shape=(1, n6), initializer=tf.zeros_initializer)


# Step 4: build model
# the model that returns the logits.
# this logits will be later passed through softmax layer
# nn1 = tf.nn.dropout(tf.nn.relu(tf.matmul(img, w1) + b1), pkeep)
# nn2 = tf.nn.dropout(tf.nn.relu(tf.matmul(nn1, w2) + b2), pkeep)
# nn3 = tf.nn.dropout(tf.nn.relu(tf.matmul(nn2, w3) + b3), pkeep)
# nn4 = tf.nn.dropout(tf.nn.relu(tf.matmul(nn3, w4) + b4), pkeep)
# nn5 = tf.nn.dropout(tf.nn.relu(tf.matmul(nn4, w5) + b5), pkeep)
# logits = tf.matmul(nn5, w6) + b6

p = 0.5

lr = tf.placeholder(tf.float32)

w1 = tf.get_variable("w1", shape=(3, 3, 1, 32), initializer=tf.random_normal_initializer(0, 0.1))
b1 = tf.get_variable("b1", shape=(1, 32), initializer=tf.zeros_initializer)

w2 = tf.get_variable("w2", shape=(14*14*32, 128), initializer=tf.random_normal_initializer(0, 0.1))
b2 = tf.get_variable("b2", shape=(1, 128), initializer=tf.zeros_initializer)

w5 = tf.get_variable("w5", shape=(128, 10), initializer=tf.random_normal_initializer(0, 0.1))
b5 = tf.get_variable("b5", shape=(1, 10), initializer=tf.zeros_initializer)

# new_img = tf.reshape(img, shape=(batch_size, 28, 28, 1))
# img (btach_size, 28, 28, 1)
conv1 = tf.nn.relu(tf.nn.conv2d(img, w1, strides=[1, 1, 1, 1], padding='SAME') + b1)
out2 = tf.nn.max_pool(conv1, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

yy = tf.reshape(out2, shape=[-1, 14*14*32])
y4 = tf.nn.relu(tf.matmul(yy, w2) + b2)
y4d = tf.nn.dropout(y4, p)
logits = tf.matmul(y4d, w5) + b5

# y5d = tf.nn.dropout(tf.nn.relu(tf.matmul(y4d, w5) + b5), p)
# y6d = tf.nn.dropout(tf.nn.relu(tf.matmul(y5d, w6) + b6), p)
# logits = tf.matmul(y6d, w7) + b7

# Step 5: define loss function
# use cross entropy of softmax of logits as the loss function
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=logits))


# Step 6: define optimizer
# using Adamn Optimizer with pre-defined learning rate to minimize loss
optimizer = tf.train.AdamOptimizer(lr).minimize(loss)


# Step 7: calculate accuracy with test set
preds = tf.nn.softmax(logits)
correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(label, 1))
accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))


def now_lr(step):
    max_lr, min_lr, decay_speed = 0.003, 0.0001, 2000.0
    return min_lr + (max_lr - min_lr) * math.exp(-step/decay_speed)


writer = tf.summary.FileWriter('./graphs/logreg', tf.get_default_graph())
with tf.Session(config=gpuConfig) as sess:
    start_time = time.time()
    sess.run(tf.global_variables_initializer())

    # train the model n_epochs times
    for i in range(n_epochs):
        sess.run(train_init)  # drawing samples from train_data
        total_loss = 0
        n_batches = 0
        try:
            while True:
                now = now_lr(i)
                _, l = sess.run([optimizer, loss], feed_dict={lr: now})
                total_loss += l
                n_batches += 1
        except tf.errors.OutOfRangeError:
            pass
        print('Average loss epoch {0}: {1}'.format(i, total_loss / n_batches))
        sess.run(test_init)  # drawing samples from test_data
        total_correct_preds = 0
        try:
            while True:
                accuracy_batch = sess.run(accuracy)
                total_correct_preds += accuracy_batch
        except tf.errors.OutOfRangeError:
            pass

        print('Accuracy {0}'.format(total_correct_preds / n_test))
    print('Total time: {0} seconds'.format(time.time() - start_time))

    # test the model
    sess.run(test_init)  # drawing samples from test_data
    total_correct_preds = 0
    try:
        while True:
            accuracy_batch = sess.run(accuracy)
            total_correct_preds += accuracy_batch
    except tf.errors.OutOfRangeError:
        pass

    print('Accuracy {0}'.format(total_correct_preds / n_test))
writer.close()
