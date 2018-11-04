import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
import pprint

pp = pprint.PrettyPrinter(indent=4)
sess = tf.InteractiveSession()

with tf.variable_scope('rnn_sample') as scope:
    sample = " if you want you"
    idx2char = list(set(sample))
    # print(idx2char)
    char2idx = {c: i for i,c in enumerate(idx2char)}

    sample_idx = [char2idx[c] for c in sample]

    dic_size = len(char2idx)
    rnn_hidden_size = len(char2idx)
    num_classes = len(char2idx)
    batch_size = 1
    sequence_len = len(sample) - 1

    x_data = [sample_idx[:-1]]
    y_data = [sample_idx[1:]]
    print(x_data, y_data)
    X = tf.placeholder(tf.int32, [None, sequence_len])
    Y = tf.placeholder(tf.int32, [None, sequence_len])

    X_one_hot = tf.one_hot(X, num_classes)
    cell = tf.contrib.rnn.BasicLSTMCell(num_units=rnn_hidden_size, state_is_tuple=True)
    initial_state = cell.zero_state(batch_size, tf.float32)
    outputs, _states = tf.nn.dynamic_rnn(cell, X_one_hot, initial_state=initial_state, dtype=tf.float32)

    weights = tf.ones([batch_size, sequence_len])
    sequence_loss = tf.contrib.seq2seq.sequence_loss(logits=outputs, targets=Y, weights=weights)
    loss = tf.reduce_mean(sequence_loss)
    # train = tf.train.GradientDescentOptimizer(learning_rate = 0.1).minimize(loss)
    train = tf.train.AdamOptimizer(learning_rate = 0.1).minimize(loss)

    prediction = tf.argmax(outputs, axis=2)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(100):
        l, _ = sess.run([loss, train], feed_dict={X:x_data, Y: y_data})
        result = sess.run(prediction, feed_dict={X: x_data})
        result_str = [idx2char[c] for  c in np.squeeze(result)]
        print(i, "loss: ", l, "Prediction: ", ''.join(result_str))



