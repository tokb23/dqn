# coding:utf-8

import os
import cv2
import gym
import random
import numpy as np
import tensorflow as tf
from collections import deque


# Environment/Agent parameters
ENV_NAME = 'Breakout-v0'
FRAME_SIZE = 84
NUM_EPISODES = 10000  # number of episodes
STATE_LENGTH = 4  # number of most recent frames as input
GAMMA = 0.99  # discount factor
EXPLORATION_STEPS = 5000  # number of steps over which epsilon decays
REPLAY_START_SIZE = 1000  # number of steps before training starts
FINAL_EPSILON = 0.1  # final value of epsilon in epsilon-greedy
INITIAL_EPSILON = 1.0  # initial value of epsilon in epsilon-greedy
NUM_REPLAY_MEMORY = 10000  # replay memory size
BATCH_SIZE = 32  # mini batch size
UPDATE_FREQ = 1000  # update frequency for target network
ACTION_FREQ = 4  # action frequency
TRAIN_FREQ = 4  # training frequency
LEARNING_RATE = 0.00025  # learning rate
MOMENTUM = 0.95  # momentum for rmsprop
MIN_GRAD = 0.01  # small value for rmsprop
LOAD_NETWORK = False
SAVER_PATH = './saved_networks'
SUMMARY_PATH = './summary'

# Network parameters
CONV1_NUM_FILTERS = 32
CONV1_FILTER_SIZE = 8
CONV1_STRIDE = 4
CONV2_NUM_FILTERS = 64
CONV2_FILTER_SIZE = 4
CONV2_STRIDE = 2
CONV3_NUM_FILTERS = 64
CONV3_FILTER_SIZE = 3
CONV3_STRIDE = 1
FC_NUM_UNITS = 512


class Agent():
    def __init__(self, env):
        self.num_actions = env.action_space.n  # number of actions
        self.epsilon = INITIAL_EPSILON
        self.epsilon_decay = (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORATION_STEPS
        self.time_step = 0
        self.action = 0
        self.D = deque()  # replay memory
        self.total_reward = 0

        # q network
        self.s, self.w_conv1, self.b_conv1, self.w_conv2, self.b_conv2, \
            self.w_conv3, self.b_conv3, self.w_fc, self.b_fc, self.w_q, self.b_q, \
            self.q = self.build_network()

        # target q network
        self.st, self.w_conv1t, self.b_conv1t, self.w_conv2t, self.b_conv2t, \
            self.w_conv3t, self.b_conv3t, self.w_fct, self.b_fct, self.w_qt, self.b_qt, \
            self.qt = self.build_network()

        # update operation for target q network
        self.update_op = [self.w_conv1t.assign(self.w_conv1),
                        self.b_conv1t.assign(self.b_conv1),
                        self.w_conv2t.assign(self.w_conv2),
                        self.b_conv2t.assign(self.b_conv2),
                        self.w_conv3t.assign(self.w_conv3),
                        self.b_conv3t.assign(self.b_conv3),
                        self.w_fct.assign(self.w_fc),
                        self.b_fct.assign(self.b_fc),
                        self.w_qt.assign(self.w_q),
                        self.b_qt.assign(self.b_q)]

        # training operation
        self.build_training_op()

        self.saver = tf.train.Saver()
        self.sess = tf.InteractiveSession()
        self.summary_writer = tf.train.SummaryWriter(SUMMARY_PATH, self.sess.graph)
        self.summary_op = tf.merge_all_summaries()
        self.sess.run(tf.initialize_all_variables())

        if not os.path.exists(SAVER_PATH):
            os.mkdir(SAVER_PATH)

        # load network
        if LOAD_NETWORK:
            self.load_network()

    def build_network(self):
        s = tf.placeholder(tf.float32, [None, FRAME_SIZE, FRAME_SIZE, STATE_LENGTH])

        w_conv1 = self.weight_variable([CONV1_FILTER_SIZE, CONV1_FILTER_SIZE,
                                        STATE_LENGTH, CONV1_NUM_FILTERS])
        b_conv1 = self.bias_variable([CONV1_NUM_FILTERS])
        h_conv1 = tf.nn.relu(self.conv2d(s, w_conv1, CONV1_STRIDE) + b_conv1)

        w_conv2 = self.weight_variable([CONV2_FILTER_SIZE, CONV1_FILTER_SIZE,
                                        CONV1_NUM_FILTERS, CONV2_NUM_FILTERS])
        b_conv2 = self.bias_variable([CONV2_NUM_FILTERS])
        h_conv2 = tf.nn.relu(self.conv2d(h_conv1, w_conv2, CONV2_STRIDE) + b_conv2)

        w_conv3 = self.weight_variable([CONV3_FILTER_SIZE, CONV3_FILTER_SIZE,
                                        CONV2_NUM_FILTERS, CONV3_NUM_FILTERS])
        b_conv3 = self.bias_variable([CONV3_NUM_FILTERS])
        h_conv3 = tf.nn.relu(self.conv2d(h_conv2, w_conv3, CONV3_STRIDE) + b_conv3)

        h_conv3_shape = h_conv3.get_shape().as_list()
        h_conv3_flat_shape = h_conv3_shape[1] * h_conv3_shape[2] * h_conv3_shape[3]
        h_conv3_flat = tf.reshape(h_conv3, [-1, h_conv3_flat_shape])

        w_fc = self.weight_variable([h_conv3_flat_shape, FC_NUM_UNITS])
        b_fc = self.bias_variable([FC_NUM_UNITS])
        h_fc = tf.nn.relu(tf.matmul(h_conv3_flat, w_fc) + b_fc)

        w_q = self.weight_variable([FC_NUM_UNITS, self.num_actions])
        b_q = self.bias_variable([self.num_actions])
        q = tf.matmul(h_fc, w_q) + b_q

        return s, w_conv1, b_conv1, w_conv2, b_conv2, w_conv3, b_conv3, w_fc, b_fc, w_q, b_q, q

    def build_training_op(self):
        self.a = tf.placeholder(tf.int64, [None])
        self.target = tf.placeholder(tf.float32, [None])

        # convert to one hot vector
        a_one_hot = tf.one_hot(self.a, self.num_actions, 1.0, 0.0)
        q_a = tf.reduce_sum(tf.mul(self.q, a_one_hot), reduction_indices=1)

        # clip the error term from the update 'target - q'
        # to be between -1 and 1
        error = self.target - q_a
        clipped_error = tf.clip_by_value(error, -1, 1)
        self.loss = tf.reduce_mean(tf.square(clipped_error))

        tf.scalar_summary('loss', self.loss)

        global_step = tf.Variable(0, trainable=False)

        self.train_step = tf.train.RMSPropOptimizer(LEARNING_RATE,
            momentum=MOMENTUM, epsilon=MIN_GRAD).minimize(self.loss, global_step=global_step)

    def set_initial_input(self, frame):
        frame = cv2.cvtColor(cv2.resize(frame, (FRAME_SIZE, FRAME_SIZE)) / 255, cv2.COLOR_BGR2GRAY)
        self.state = np.stack((frame, frame, frame, frame), axis=2)

    def get_action(self):
        if self.time_step % ACTION_FREQ == 0:
            if random.random() < self.epsilon or self.time_step < REPLAY_START_SIZE:
                self.action = random.randrange(self.num_actions)
            else:
                self.action = np.argmax(self.q.eval(feed_dict={self.s: [self.state]})[0])

        # epsilon decay
        if self.epsilon > FINAL_EPSILON and self.time_step > REPLAY_START_SIZE:
            self.epsilon -= self.epsilon_decay

        return self.action

    def run(self, frame, action, reward, done):
        next_state = np.append(frame, self.state[:, :, 1:], axis=2)

        # store transition in replay memory
        self.D.append((self.state, action, reward, next_state, done))
        if len(self.D) > NUM_REPLAY_MEMORY:
            self.D.popleft()

        if self.time_step > REPLAY_START_SIZE:
            if self.time_step % TRAIN_FREQ == 0:
                self.train_network()

        if self.time_step % UPDATE_FREQ == 0:
            self.update_target_q_network()

        # *********************************
        # **************debug**************
        self.total_reward += np.sign(reward)
        if done:
            print 'TOTAL_REWARD: {0}'.format(self.total_reward)
            self.total_reward = 0

        if self.time_step % 100 == 0:
            if self.time_step <= REPLAY_START_SIZE:
                mode = 'observe'
            elif self.time_step > REPLAY_START_SIZE and \
            self.time_step <= REPLAY_START_SIZE + EXPLORATION_STEPS:
                mode = 'explore'
            else:
                mode = 'train'
            print 'TIMESTEP: {0} / STATE: {1} / EPSILON: {2}'.format(
                self.time_step, mode, self.epsilon)
        # *********************************
        # *********************************

        # save network
        if self.time_step % 10000 == 0:
            self.saver.save(self.sess, SAVER_PATH + '/network', global_step=self.time_step)

        self.state = next_state
        self.time_step += 1

    def train_network(self):
        # sample random minibatch of transition from replay memory
        minibatch = random.sample(self.D, BATCH_SIZE)
        state_batch = [data[0] for data in minibatch]
        action_batch = [data[1] for data in minibatch]
        reward_batch = [data[2] for data in minibatch]
        next_state_batch = [data[3] for data in minibatch]
        done_batch = [data[4] for data in minibatch]

        # clip all positive rewards at 1 and all negative rewards at -1,
        # leaving 0 rewards unchanged
        reward_batch = np.sign(reward_batch)

        target_batch = []

        # convert 'True' to 1, 'False' to 0
        done_batch = np.array(done_batch) + 0

        q_batch = self.qt.eval(feed_dict={self.st: next_state_batch})

        target_batch = reward_batch + (1 - done_batch) * GAMMA * np.max(q_batch)

        self.train_step.run(feed_dict={
            self.s: state_batch,
            self.a: action_batch,
            self.target: target_batch
        })

        if self.time_step % 100 == 0:
            summary_str, loss, q = self.sess.run([self.summary_op, self.loss, self.q], feed_dict={
                self.s: state_batch,
                self.a: action_batch,
                self.target: target_batch
            })
            print('LOSS: {0} / Q: {1}'.format(self.time_step, loss, q.mean()))
            self.summary_writer.add_summary(summary_str, self.time_step)
            self.summary_writer.flush()

    def update_target_q_network(self):
        self.sess.run(self.update_op)

    def load_network(self):
        checkpoint = tf.train.get_checkpoint_state(SAVER_PATH)
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            print 'Successfully loaded: {0}'.format(checkpoint.model_checkpoint_path)
        else:
            print 'Training new network'

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.01)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.01, shape=shape)
        return tf.Variable(initial)

    def conv2d(self, x, w, stride):
        return tf.nn.conv2d(x, w, strides=[1, stride, stride, 1], padding='VALID')


def preprocess(frame):
    frame = cv2.cvtColor(cv2.resize(frame, (FRAME_SIZE, FRAME_SIZE)) / 255, cv2.COLOR_BGR2GRAY)
    return np.reshape(frame, (FRAME_SIZE, FRAME_SIZE, 1))


def main():
    env = gym.make(ENV_NAME)
    agent = Agent(env)

    for eposode in range(NUM_EPISODES):
        done = False
        observation = env.reset()
        agent.set_initial_input(observation)
        while not done:
            env.render()
            action = agent.get_action()
            observation, reward, done, _ = env.step(action)
            observation = preprocess(observation)
            agent.run(observation, action, reward, done)


if __name__ == '__main__':
    main()
