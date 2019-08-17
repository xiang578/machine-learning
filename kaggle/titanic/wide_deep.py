import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
tf.set_random_seed(777)  # for reproducibility

# Normalize x data
def MinMaxScaler(data):
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    # noise term prevents the zero division
    return numerator / (denominator + 1e-7)

def load_file(is_test):
    if is_test:
        data_df = pd.read_csv("test.csv")
    else:
        data_df = pd.read_csv("train.csv")

    cols = ["Pclass", "Sex", "Age", "Fare",
            "Embarked_0", "Embarked_1", "Embarked_2"]

    data_df['Sex'] = data_df['Sex'].map({'female': 0, 'male': 1}).astype(int)

    # handle missing values of age
    data_df["Age"] = data_df["Age"].fillna(data_df["Age"].mean())
    data_df["Fare"] = data_df["Fare"].fillna(data_df["Fare"].mean())

    data_df['Embarked'] = data_df['Embarked'].fillna('S')
    data_df['Embarked'] = data_df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)
    data_df = pd.concat([data_df, pd.get_dummies(data_df['Embarked'], prefix='Embarked')], axis=1)

    # print(data_df.head())
    data = data_df[cols].values

    if is_test:
        sing_col = data_df["PassengerId"].values # Need it for submission
    else:
        sing_col = data_df["Survived"].values

    return sing_col, data

# Load data and min/max
# TODO: clean up this code

y_train, x_train = load_file(0)
y_train = np.expand_dims(y_train, 1)
train_len = len(x_train)
# Get train file
passId, x_test = load_file(1)

# print(x_train.shape, x_test.shape)

x_all = np.vstack((x_train, x_test))
# print(x_all.shape)

x_min_max_all = MinMaxScaler(x_all)
x_train = x_min_max_all[:train_len]
x_test = x_min_max_all[train_len:]

# print(x_train.shape, x_test.shape)

# Parameters
learning_rate = 0.1

# Network Parameters
n_input = 7  # x_train.shape[1]

n_hidden_1 = 32  # 1st layer number of features
n_hidden_2 = 64  # 2nd layer number of features

# placeholders for a tensor that will be always fed.
X = tf.placeholder(tf.float32, shape=[None, n_input])
Y = tf.placeholder(tf.float32, shape=[None, 1])

# Wide
W = tf.get_variable("wide_weight" , [n_input , 1] , initializer = tf.contrib.layers.xavier_initializer())
b = tf.get_variable("wide_bias", [1 , 1] , initializer = tf.zeros_initializer())

wide_pred = tf.add(tf.matmul(X, W), b)

# Deep
W1 = tf.get_variable("deep_weight_1" , [n_input , 3] , initializer = tf.contrib.layers.xavier_initializer())
b1 = tf.get_variable("deep_bias_1", [3] , initializer = tf.zeros_initializer())
L1 = tf.nn.relu(tf.matmul(X, W1) + b1)

W2 = tf.get_variable("deep_weight_2" , [3 , 1] , initializer = tf.contrib.layers.xavier_initializer())
b2 = tf.get_variable("deep_bias_2", [1] , initializer = tf.zeros_initializer())
deep_pred = tf.matmul(L1, W2) + b2

wide_weight = tf.Variable(tf.truncated_normal([1, 1]), name="wide_model_weight")
deep_weight = tf.Variable(tf.truncated_normal([1, 1]), name="deep_model_weight")
central_bias = tf.Variable([np.random.randn()], name="central_bias")

hypothesis = tf.sigmoid(tf.matmul(wide_pred, wide_weight) + tf.matmul(deep_pred, deep_weight) + central_bias)

# cost/loss function
cost = -tf.reduce_mean(Y*tf.log(hypothesis) + (1 - Y)*tf.log(1 - hypothesis))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)


# Accuracy computation
# True if hypothesis>0.5 else False
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

training_epochs = 20
batch_size = 32
display_step = 1
step_size = 1000

# Launch the graph
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        avg_accuracy = 0.
        # Loop over step_size
        for step in range(step_size):
            # Pick an offset within the training data, which has been randomized.
            # Note: we could use better randomization across epochs.
            offset = (step * batch_size) % (y_train.shape[0] - batch_size)
            # Generate a minibatch.
            batch_data = x_train[offset:(offset + batch_size), :]
            batch_labels = y_train[offset:(offset + batch_size), :]

            # Run optimization op (backprop) and cost op (to get loss value)
            _, c, a = sess.run([optimizer, cost, accuracy], feed_dict={X: batch_data,
                                                                       Y: batch_labels})
            avg_cost += c / step_size
            avg_accuracy += a / step_size

        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%02d' % (epoch + 1), "cost={:.4f}".format(avg_cost),
                  "train accuracy={:.4f}".format(avg_accuracy))
    print("Optimization Finished!")

    ## 4. Results (creating submission file)

    outputs = sess.run(predicted, feed_dict={X: x_test})
    submission = ['PassengerId,Survived']

    for id, prediction in zip(passId, outputs):
        submission.append('{0},{1}'.format(id, int(prediction)))

    submission = '\n'.join(submission)

    with open('submission.csv', 'w') as outfile:
        outfile.write(submission)