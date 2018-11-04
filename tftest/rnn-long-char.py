import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
import pprint

pp = pprint.PrettyPrinter(indent=4)
sess = tf.InteractiveSession()

with tf.variable_scope('rnn_sample') as scope:
    sentence = ("if you want to build a ship, don't drum up people together to "
            "collect wood and don't assign them tasks and work, but rather "
            "teach them to long for the endless immensity of the sea.")
    char_set = list(set(sentence))
    char2idx = {c: i for i,c in enumerate(char_set)}

    dic_size = len(char2idx)
    rnn_hidden_size = len(char2idx)
    num_classes = len(char2idx)
    sequence_len = 10

    dataX = []
    dataY = []
    for i in range(0, len(sentence) - sequence_len):
        x_str = sentence[i:i+sequence_len]
        y_str = sentence[i + 1: i + sequence_len + 1]
        # print(i, x_str, '->', y_str)

        x = [char2idx[c] for c in x_str]
        y = [char2idx[c] for c in y_str]

        dataX.append(x)
        dataY.append(y)

    batch_size = len(dataX)



    X = tf.placeholder(tf.int32, [None, sequence_len])
    Y = tf.placeholder(tf.int32, [None, sequence_len])
#
    X_one_hot = tf.one_hot(X, num_classes)
    print(X_one_hot)

    cell = tf.contrib.rnn.BasicLSTMCell(num_units=rnn_hidden_size, state_is_tuple=True)
    cell = rnn.MultiRNNCell([cell] * 2, state_is_tuple=True)

#     initial_state = cell.zero_state(batch_size, tf.float32)
    outputs, _states = tf.nn.dynamic_rnn(cell, X_one_hot, dtype=tf.float32)
    X_for_softmax = tf.reshape(outputs, [-1, rnn_hidden_size])

    softmax_w = tf.get_variable("softmax_w", [rnn_hidden_size, num_classes])
    softmax_b = tf.get_variable("softmax_b", [num_classes])

    outputs = tf.matmul(X_for_softmax, softmax_w) + softmax_b

    outputs = tf.reshape(outputs, [batch_size, sequence_len, num_classes])

    weights = tf.ones([batch_size, sequence_len])
    sequence_loss = tf.contrib.seq2seq.sequence_loss(logits=outputs, targets=Y, weights=weights)
    loss = tf.reduce_mean(sequence_loss)
    # train = tf.train.GradientDescentOptimizer(learning_rate = 0.1).minimize(loss)
    train = tf.train.AdamOptimizer(learning_rate = 0.1).minimize(loss)
#
#     prediction = tf.argmax(outputs, axis=2)
#
#
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(500):
        _, l, results = sess.run([train, loss, outputs], feed_dict={X: dataX, Y: dataY})
        for j, result in enumerate(results):
            index = np.argmax(result, axis=1)
            print(i, j, ''.join([char_set[c] for c in index]), l)
        # result = sess.run(prediction, feed_dict={X: x_data})
#         result_str = [idx2char[c] for  c in np.squeeze(result)]
#         print(i, "loss: ", l, "Prediction: ", ''.join(result_str))

    results = sess.run(outputs, feed_dict={X: dataX})
    for j, result in enumerate(results):
        index = np.argmax(result, axis=1)
        if j is 0:
            print(''.join([char_set[t] for t in index]), end='')
        else:
            print(char_set[index[-1]], end='')

