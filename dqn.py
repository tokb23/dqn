# coding:utf-8

import os
import gym
import random
import numpy as np
import tensorflow as tf
from collections import deque
from skimage.transform import resize
from skimage.color import rgb2gray
from keras.layers import Convolution2D, Flatten, Dense, Input
from keras.models import Model

os.environ["KERAS_BACKEND"] = "tensorflow"

# Environment/Agent parameters
ENV_NAME = 'Breakout-v0'
FRAME_WIDTH = 84
FRAME_HEIGHT = 84
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
SAVE_PATH = './saved_networks'
SUMMARY_PATH = './summary'


class Agent():
    def __init__(self, num_actions):
        self.num_actions = num_actions
        self.epsilon = INITIAL_EPSILON
        self.epsilon_decay = (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORATION_STEPS
        self.time_step = 0
        self.repeat_action = 0

        # for summaries
        self.total_reward = 0
        self.total_max_q = 0
        self.episode_time = 0

        # Create replay memory
        self.D = deque()

        # Create q network
        self.s, q_network = self.build_network()
        network_params = q_network.trainable_weights
        self.q_values = q_network(self.s)

        # Create target network
        self.st, target_q_network = self.build_network()
        target_network_params = target_q_network.trainable_weights
        self.target_q_values = target_q_network(self.st)

        # Op for periodically updating target network
        self.update_target_network_params = [target_network_params[i].assign(network_params[i]) for i in range(len(target_network_params))]

        # Define loss and gradient update op
        self.a, self.y, self.grad_update = self.build_training_op(network_params)

        self.sess = tf.InteractiveSession()
        self.saver = tf.train.Saver()
        self.summary_placeholders, self.update_ops, self.summary_op = self.setup_summaries()
        self.summary_writer = tf.train.SummaryWriter(SUMMARY_PATH, self.sess.graph)

        if not os.path.exists(SAVE_PATH):
            os.mkdir(SAVE_PATH)

        # Load network
        if LOAD_NETWORK:
            self.load_network()
        else:
            self.sess.run(tf.initialize_all_variables())

        # Initialize target network
        self.sess.run(self.update_target_network_params)

    def setup_summaries(self):
        episode_total_reward = tf.Variable(0.)
        tf.scalar_summary("Episode/Total Reward", episode_total_reward)
        episode_avg_max_q = tf.Variable(0.)
        tf.scalar_summary("Episode/Average Max Q Value", episode_avg_max_q)
        summary_vars = [episode_total_reward, episode_avg_max_q]
        summary_placeholders = [tf.placeholder(tf.float32) for i in range(len(summary_vars))]
        update_ops = [summary_vars[i].assign(summary_placeholders[i]) for i in range(len(summary_vars))]
        summary_op = tf.merge_all_summaries()
        return summary_placeholders, update_ops, summary_op

    def build_network(self):
        s = tf.placeholder(tf.float32, [None, STATE_LENGTH, FRAME_WIDTH, FRAME_HEIGHT])
        inputs = Input(shape=(STATE_LENGTH, FRAME_WIDTH, FRAME_HEIGHT,))
        model = Convolution2D(nb_filter=32, nb_row=8, nb_col=8, subsample=(4, 4), activation='relu', border_mode='valid')(inputs)
        model = Convolution2D(nb_filter=64, nb_row=4, nb_col=4, subsample=(2, 2), activation='relu', border_mode='valid')(model)
        model = Convolution2D(nb_filter=64, nb_row=3, nb_col=3, subsample=(1, 1), activation='relu', border_mode='valid')(model)
        model = Flatten()(model)
        model = Dense(output_dim=512, activation='relu')(model)
        q_values = Dense(output_dim=self.num_actions, activation='linear')(model)
        m = Model(input=inputs, output=q_values)
        return s, m

    def build_training_op(self, network_params):
        a = tf.placeholder(tf.int64, [None])
        y = tf.placeholder(tf.float32, [None])

        # Convert to one hot vector
        a_one_hot = tf.one_hot(a, self.num_actions, 1.0, 0.0)
        action_q_values = tf.reduce_sum(tf.mul(self.q_values, a_one_hot), reduction_indices=1)

        # Clip the error term to be between -1 and 1
        error = y - action_q_values
        clipped_error = tf.clip_by_value(error, -1, 1)
        loss = tf.reduce_mean(tf.square(clipped_error))

        optimizer = tf.train.RMSPropOptimizer(LEARNING_RATE, momentum=MOMENTUM, epsilon=MIN_GRAD)
        grad_update = optimizer.minimize(loss, var_list=network_params)
        return a, y, grad_update

    def get_initial_state(self, frame):
        frame = resize(rgb2gray(frame), (FRAME_WIDTH, FRAME_HEIGHT))
        return np.stack((frame, frame, frame, frame), axis=0)

    def get_action(self, state):
        action = self.repeat_action

        if self.time_step % ACTION_FREQ == 0:
            if random.random() <= self.epsilon or self.time_step < REPLAY_START_SIZE:
                action = random.randrange(self.num_actions)
            else:
                action = np.argmax(self.q_values.eval(feed_dict={self.s: [state]}))
            self.repeat_action = action

        # Epsilon decay
        if self.epsilon > FINAL_EPSILON and self.time_step >= REPLAY_START_SIZE:
            self.epsilon -= self.epsilon_decay

        return action

    def run(self, state, action, reward, terminal, frame):
        next_state = np.append(frame, state[1:, :, :], axis=0)

        # Store transition in replay memory
        self.D.append((state, action, reward, next_state, terminal))
        if len(self.D) > NUM_REPLAY_MEMORY:
            self.D.popleft()

        # Train network
        if self.time_step >= REPLAY_START_SIZE:
            if self.time_step % TRAIN_FREQ == 0:
                self.train_network()

        # Update target network
        if self.time_step % UPDATE_FREQ == 0:
            self.sess.run(self.update_target_network_params)

        # Save network
        if self.time_step + 1 % 10000 == 0:
            self.saver.save(self.sess, SAVE_PATH + '/network', global_step=self.time_step)

        self.episode_time += 1
        self.total_reward += np.sign(reward)
        self.total_max_q += np.max(self.q_values.eval(feed_dict={self.s: [state]}))

        if terminal:
            stats = [self.total_reward, self.total_max_q / float(self.episode_time)]
            for i in range(len(stats)):
                self.sess.run(self.update_ops[i], feed_dict={self.summary_placeholders[i]: float(stats[i])})
            summary_str = self.sess.run(self.summary_op)
            self.summary_writer.add_summary(summary_str, self.time_step)

            # Debug
            if self.time_step < REPLAY_START_SIZE:
                mode = 'random'
            elif REPLAY_START_SIZE <= self.time_step < REPLAY_START_SIZE + EXPLORATION_STEPS:
                mode = 'explore'
            else:
                mode = 'exploit'
            print 'TIMESTEP: {0} / EPSILON: {1} / TOTAL_REWARD: {2} / AVG_MAX_Q: {3:.4f} / MODE: {4}'.format(self.time_step, self.epsilon, self.total_reward, self.total_max_q / float(self.episode_time), mode)
            self.total_reward = 0
            self.total_max_q = 0
            self.episode_time = 0

        self.time_step += 1

        return next_state

    def train_network(self):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        terminal_batch = []
        y_batch = []

        # Sample random minibatch of transition from replay memory
        minibatch = random.sample(self.D, BATCH_SIZE)
        for data in minibatch:
            state_batch.append(data[0])
            action_batch.append(data[1])
            reward_batch.append(data[2])
            next_state_batch.append(data[3])
            terminal_batch.append(data[4])

        """
        state_batch = [data[0] for data in minibatch]
        action_batch = [data[1] for data in minibatch]
        reward_batch = [data[2] for data in minibatch]
        next_state_batch = [data[3] for data in minibatch]
        terminal_batch = [data[4] for data in minibatch]
        """
        # Clip all positive rewards at 1 and all negative rewards at -1,
        # leaving 0 rewards unchanged
        reward_batch = np.sign(reward_batch)

        # Convert True to 1, False to 0
        terminal_batch = np.array(terminal_batch) + 0

        target_q_values_batch = self.target_q_values.eval(feed_dict={self.st: next_state_batch})

        y_batch = reward_batch + (1 - terminal_batch) * GAMMA * np.max(target_q_values_batch)

        self.sess.run(self.grad_update, feed_dict={
            self.s: state_batch,
            self.a: action_batch,
            self.y: y_batch
        })

        """
        if self.time_step % 100 == 0:
            summary_str, loss, q = self.sess.run([self.summary_op, self.loss, self.q], feed_dict={
                self.s: state_batch,
                self.a: action_batch,
                self.target: target_batch
            })
            print('LOSS: {0} / Q: {1}'.format(self.time_step, loss, q.mean()))
            self.summary_writer.add_summary(summary_str, self.time_step)
            self.summary_writer.flush()
        """

    def load_network(self):
        checkpoint = tf.train.get_checkpoint_state(SAVE_PATH)
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            print 'Successfully loaded: {0}'.format(checkpoint.model_checkpoint_path)
        else:
            print 'Training new network'


def preprocess(frame):
    frame = resize(rgb2gray(frame), (FRAME_WIDTH, FRAME_HEIGHT))
    return np.reshape(frame, (1, FRAME_WIDTH, FRAME_HEIGHT))


def main():
    env = gym.make(ENV_NAME)
    agent = Agent(num_actions=env.action_space.n)

    for eposode in range(NUM_EPISODES):
        terminal = False
        observation = env.reset()
        state = agent.get_initial_state(observation)
        while not terminal:
            env.render()
            action = agent.get_action(state)
            observation, reward, terminal, _ = env.step(action)
            observation = preprocess(observation)
            state = agent.run(state, action, reward, terminal, observation)


if __name__ == '__main__':
    main()
