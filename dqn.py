# coding:utf-8

import tensorflow as tf

nb_frames = 4
grid_size = 80
nb_actions = 6


class CNN():
    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.01)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.01, shape=shape)
        return tf.Variable(initial)

    def conv2d(self, x, w, stride):
        return tf.nn.conv2d(x, w, strides=[1, stride, stride, 1],
                            padding="SAME")

    def max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1], padding="SAME")

    def __init__(self):
        self.x = tf.placeholder(tf.float32, [None, grid_size, grid_size, nb_frames]

        self.w1 = weight_variable([8, 8, 4, 32])
        self.b1 = bias_variable([32])
        self.h_conv1 = tf.nn.relu(conv2d(self.x, self.w1, 4) + self.b1)
        self.h_pool1 = max_pool_2x2(self.h_conv1)

        self.w2 = weight_variable([4, 4, 32, 64])
        self.b2 = bias_variable([64])
        self.h_conv2 = tf.nn.relu(conv2d(self.h_pool1, self.w2, 2) + self.b2)

        self.w3 = weight_variable([3, 3, 64, 64])
        self.b3 = bias_variable([64])
        self.h_conv3 = tf.nn.relu(conv2d(self.h_conv2, self.w3, 1) + self.b3)

        self.w_fc1 = weight_variable([1600, 512])
        self.b_fc1 = bias_variable([512])

        self.h_conv3_flat = tf.reshape(self.h_conv3, [-1, 1600])
        self.h_fc1 = tf.nn.relu(tf.matmul(self.h_conv3_flat, self.W_fc1) + self.b_fc1)

        self.W_fc2 = weight_variable([512, nb_actions])
        self.b_fc2 = bias_variable([nb_actions])

        self.output = tf.matmul(self.h_fc1, self.W_fc2) + self.b_fc2

        # loss function, optimizer
        self.a = tf.placeholder(tf.float32, [None, nb_actions])
        self.y = tf.placeholder(tf.float32, [None])
        self.apx_q = tf.reduce_sum(tf.mul(self.output, self.a), reduction_indices = 1)
        self.cost = tf.reduce_mean(tf.square(self.y - self.apx_q))
        self.train_step = tf.train.AdamOptimizer(1e-6).minimize(self.cost)

        self.sess = tf.Session()
        self.sess.run(tf.initialize_all_variables())

    def main():
        net = CNN()

if __name__ == "__main__":
    main()
