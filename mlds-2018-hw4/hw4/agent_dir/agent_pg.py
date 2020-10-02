from agent_dir.agent import Agent
import scipy
import numpy as np
import tensorflow as tf

def prepro(o,image_size=[80,80]):
    """
    Call this function to preprocess RGB image to grayscale image if necessary
    This preprocessing code is from
        https://github.com/hiwonjoon/tf-a3c-gpu/blob/master/async_agent.py
    
    Input: 
    RGB image: np.array
        RGB screen of game, shape: (210, 160, 3)
    Default return: np.array 
        Grayscale image, shape: (80, 80, 1)
    
    """
    y = o.astype(np.uint8)
    resized = scipy.misc.imresize(y, image_size)
    return np.expand_dims(resized.astype(np.float32),axis=2)


class Agent_PG(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """

        super(Agent_PG,self).__init__(env)

        if args.test_pg:
            #you can load your model here
            print('loading trained model')

        ##################
        # YOUR CODE HERE #
        ##################
        self.n_actions = env.action_space.n
        self.n_features = env.observation_space.shape[0]
        self.lr = 0.01
        self.gamma = 0.95
        self.ep_obs, self.ep_as, self.ep_rs = [], [], []
        self._build_net()
        self.env = env
        self.sess = tf.Session(tf.global_variables_initializer())

    def _build_net(self):
        with tf.name_scape('inputs'):
            self.tf_obs = tf.placeholder(tf.float32, [None, self.n_features], name="observation")
            self.tf_acts = tf.placeholder(tf.float32, [None, ], name="actions_num")
            self.tf_vt = tf.placeholder(tf.float32, [None, ], name="actions_value")
        layer = tf.layers.dense(inputs=self.n_features,
                                units=10,
                                activation=tf.nn.tanh,
                                kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
                                bias_initializer=tf.constant_initializer(0.1),
                                name='fc1')
        all_act = tf.layers.dense(inputs=layer,
                                units=self.n_actions,
                                activation=None,
                                kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
                                bias_initializer=tf.constant_initializer(0.1),
                                name='fc2')
        self.all_act_prob = tf.nn.softmax(all_act, name='act_prob')

        with tf.name_scope('loss'):
            neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=all_act, labels=self.tf_acts)
            loss = tf.reduce_mean(neg_log_prob * self.tf_vt)

        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)


    def init_game_setting(self):
        """

        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary

        """
        ##################
        # YOUR CODE HERE #
        ##################

        saver = tf.train.Saver(max_to_keep=0)
        saver.restore(self.sess, "pong_pg")


    def train(self):
        """
        Implement your training algorithm here
        """
        ##################
        # YOUR CODE HERE #
        ##################
        RENDER = False
        for i in range(1000):
            observation = self.env.reset()

            while True:
                if RENDER:
                    self.env.render()
                action = self.make_action(observation)
                observation_, reward, done, info = self.env.step(action)
                self.ep_obs.append(np.reshape(observation_, (210*160*3)))
                self.ep_as.append(action)
                self.ep_rs.append(reward)

                if done:
                    ep_rs_sum = sum(self.ep_rs)
                    if 'running_reward' not in globals():
                        running_reward = ep_rs_sum
                    else:
                        running_reward = running_reward * 0.99 + ep_rs_sum * 0.01
                    if running_reward > 100:
                        RENDER = True
                    print("episode: ", i, " R: ", int(running_reward))

                    discounted_ep_rs = np.zeros_like(self.ep_rs)
                    running_add = 0
                    for t in reversed(range(0, len(self.ep_rs))):
                        running_add = running_add * self.gamma + self.ep_rs[t]
                        discounted_ep_rs[t] = running_add
                    discounted_ep_rs -= np.mean(discounted_ep_rs)
                    discounted_ep_rs /= np.sum(discounted_ep_rs)

                    self.sess.run(self.train_op, feed_dict={
                        self.tf_obs: np.vstack(self.eb_obs),
                        self.tf_acts: np.array(self.ep_as),
                        self.tf_vt: discounted_ep_rs
                    })

                    break
                observation = observation_
        saver = tf.train.Saver(max_to_keep=0)
        saver.save(self.sess, "pong_pg")

    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent

        Input:
            observation: np.array
                current RGB screen of game, shape: (210, 160, 3)

        Return:
            action: int
                the predicted action from trained model
        """
        ##################
        # YOUR CODE HERE #
        ##################
        new_ob = np.reshape(observation, (1, 210*160*3))
        prob_weights = self.sess.run(self.all_act_prob, feed_dict={self.tf_obs: new_ob})
        action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())
        # return self.env.get_random_action()
        return action

