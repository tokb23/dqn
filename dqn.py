# coding:utf-8

import tensorflow as tf
import numpy as np
import random
from collections import deque
import cv2
import tetris as game
import os


ACTIONS = 6
GAMMA = 0.99
OBSERVE = 500.
EXPLORE = 500.
FINAL_EPSILON = 0.05
INITIAL_EPSILON = 1.0
REPLAY_MEMORY = 100000
BATCH_SIZE = 32
FRAMES = 4
GRID_SIZE = 80
PATH = "./saved_networks"


class DQN():
    def __init__(self):
        self.replay_memory = deque()
        self.input, self.output, self.a, self.y, self.train_step = cnn()

        # save and load networks
        self.saver = tf.train.Saver()
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.initialize_all_variables())
        if not os.path.exists(PATH):
            os.mkdir(PATH)
        checkpoint = tf.train.get_checkpoint_state(PATH)
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            print "Successfully loaded:", checkpoint.model_checkpoint_path
        else:
            print "Could not find old network weights"

    def cnn(self):
        input = tf.placeholder(tf.float32, [None, GRID_SIZE, GRID_SIZE, FRAMES]

        w1 = weight_variable([8, 8, 4, 32])
        b1 = bias_variable([32])
        h_conv1 = tf.nn.relu(conv2d(x, w1, 4) + b1)
        h_pool1 = max_pool_2x2(h_conv1)

        w2 = weight_variable([4, 4, 32, 64])
        b2 = bias_variable([64])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, w2, 2) + b2)

        w3 = weight_variable([3, 3, 64, 64])
        b3 = bias_variable([64])
        h_conv3 = tf.nn.relu(conv2d(h_conv2, w3, 1) + b3)

        w_fc1 = weight_variable([1600, 512])
        b_fc1 = bias_variable([512])

        h_conv3_flat = tf.reshape(h_conv3, [-1, 1600])
        h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)

        W_fc2 = weight_variable([512, ACTIONS])
        b_fc2 = bias_variable([ACTIONS])

        output = tf.matmul(h_fc1, W_fc2) + b_fc2

        a = tf.placeholder(tf.float32, [None, ACTIONS])
        y = tf.placeholder(tf.float32, [None])
        apx_q = tf.reduce_sum(tf.mul(output, a), reduction_indices=1)
        cost = tf.reduce_mean(tf.square(y - apx_q))
        train_step = tf.train.AdamOptimizer(1e-6).minimize(cost)

        return input, output, a, y, train_step

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.01)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.01, shape=shape)
        return tf.Variable(initial)

    def conv2d(self, x, w, stride):
        return tf.nn.conv2d(x, w, strides=[1, stride, stride, 1], padding="SAME")

    def max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


class Agent(DQN):
    def __init__(self, env):
        self.state = set_init_state(env)
        self.epsilon = INITIAL_EPSILON
        self.time = 0

    def set_init_state(self, env):
        action0 = np.array([1,0,0,0,0,0]) # do nothing
        frame, _, _ = env.frame_step(action0)
        frame = cv2.cvtColor(cv2.resize(frame, (80, 80)), cv2.COLOR_BGR2GRAY)
        _, frame = cv2.threshold(frame, 1, 255, cv2.THRESH_BINARY)
        return np.stack((frame, frame, frame, frame), axis = 2)

    def get_action(self):
        qw = self.output.eval(feed_dict={self.input: [self.state]})[0]
        action = np.zeros([ACTIONS])
        action_index = 0
        if random.random() <= self.epsilon or self.time <= OBSERVE:
            action_index = random.randrange(ACTIONS)
            action[action_index] = 1
        else:
            action_index = np.argmax(qw)
            action[action_index] = 1

        if self.epsilon > FINAL_EPSILON and self.time > OBSERVE:
            self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        return action

    def learn(self, frame, action, reward, terminal):
        next_state = np.append(frame, state[:,:,1:], axis = 2)
        self.replay_memory.append((self.state, action, reward, next_state, terminal))
        if len(self.replay_memory) > REPLAY_MEMORY:
            self.replay_memory.popleft()
        if self.time > OBSERVE:
            train_dqn()

        mode = ""
        if self.time <= OBSERVE:
            mode = "observe"
        elif self.time > OBSERVE and self.time <= OBSERVE + EXPLORE:
            mode = "explore"
        else:
            mode = "train"
        print "TIMESTEP", self.time, "/ STATE", mode, "/ EPSILON", self.epsilon

        self.state = next_state
        self.time += 1

    def train_dqn(self):
        minibatch = random.sample(self.replay_memory, BATCH_SIZE)
        state_batch = [data[0] for data in minibatch]
        action_batch = [data[1] for data in minibatch]
        reward_batch = [data[2] for data in minibatch]
        next_state_batch = [data[3] for data in minibatch]

        y_batch = []
        qw_batch = self.output.eval(feed_dict={self.input: next_state_batch})
        for i in range(0, BATCH_SIZE):
            terminal = minibatch[i][4]
            if terminal:
                y_batch.append(reward_batch[i])
            else:
                y_batch.append(reward_batch[i] + GAMMA * np.max(qw_batch[i]))

        self.train_step.run(feed_dict={
            self.y: y_batch,
            self.a: action_batch,
            self.input: state_batch
            })

        if self.time % 1000 == 0:
            self.saver.save(self.sess, PATH + '/network' + '-dqn', global_step=self.time)


def preprocess(frame):
    frame = cv2.cvtColor(cv2.resize(frame, (80, 80)), cv2.COLOR_BGR2GRAY)
    _, frame = cv2.threshold(frame, 1, 255, cv2.THRESH_BINARY)
    return np.reshape(frame, (80, 80, 1))

def playgame():
    env = game.GameState()
    agent = Agent(env)

    while True:
        action = agent.get_action()
        frame, reward, terminal = env.frame_step(action)
        frame = preprocess(frame)
        agent.learn(frame, action, reward, terminal)

def main():
    playgame()


if __name__ == "__main__":
    main()
