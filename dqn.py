# coding:utf-8

import os
import gym
import random
import numpy as np
import tensorflow as tf
from collections import deque
from skimage.color import rgb2gray
from skimage.transform import resize
from keras.models import Model
from keras.layers import Convolution2D, Flatten, Dense, Input

os.environ['KERAS_BACKEND'] = 'tensorflow'

# Environment/Agent parameters
ENV_NAME = 'Breakout-v0'  # environment
FRAME_WIDTH = 84  # resized frame width
FRAME_HEIGHT = 84  # resized frame height
NUM_EPISODES = 10000  # number of episodes
STATE_LENGTH = 4  # number of most recent frames as input
GAMMA = 0.99  # discount factor
EXPLORATION_STEPS = 1000000  # number of steps over which epsilon decays
REPLAY_START_SIZE = 10000  # number of steps before training starts
FINAL_EPSILON = 0.1  # final value of epsilon in epsilon-greedy
INITIAL_EPSILON = 1.0  # initial value of epsilon in epsilon-greedy
NUM_REPLAY_MEMORY = 60000  # replay memory size
BATCH_SIZE = 32  # mini batch size
UPDATE_FREQ = 10000  # update frequency for target network
ACTION_FREQ = 4  # action frequency
TRAIN_FREQ = 4  # training frequency
LEARNING_RATE = 0.00025  # learning rate
MOMENTUM = 0.95  # momentum for rmsprop
MIN_GRAD = 0.01  # small value for rmsprop
LOAD_NETWORK = False
SAVE_FREQ = 1000000
SAVE_NETWORK_PATH = './saved_networks'
SAVE_SUMMARY_PATH = './summary'
TRAIN = True


class Agent():
    def __init__(self, num_actions):
        self.num_actions = num_actions
        self.epsilon = INITIAL_EPSILON
        self.epsilon_decay = (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORATION_STEPS
        self.time_step = 1
        self.repeat_action = 0

        # For summaries
        self.total_reward = 0
        self.total_max_q = 0
        self.episode_time = 1
        self.episode_step = 1
        self.total_loss = 0

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
        self.update_target_network_params = [target_network_params[i].assign(network_params[i])
            for i in xrange(len(target_network_params))]

        # Define loss and gradient update op
        self.a, self.y, self.loss, self.grad_update = self.build_training_op(network_params)

        self.sess = tf.InteractiveSession()
        self.saver = tf.train.Saver(network_params)
        self.summary_placeholders, self.update_ops, self.summary_op = self.setup_summaries()
        self.summary_writer = tf.train.SummaryWriter(SAVE_SUMMARY_PATH, self.sess.graph)

        if not os.path.exists(SAVE_NETWORK_PATH):
            os.mkdir(SAVE_NETWORK_PATH)

        self.sess.run(tf.initialize_all_variables())

        # Load network
        if LOAD_NETWORK:
            self.load_network()

        # Initialize target network
        self.sess.run(self.update_target_network_params)

    def build_network(self):
        s = tf.placeholder(tf.float32, [None, STATE_LENGTH, FRAME_WIDTH, FRAME_HEIGHT])
        inputs = Input(shape=(STATE_LENGTH, FRAME_WIDTH, FRAME_HEIGHT,))
        model = Convolution2D(nb_filter=32, nb_row=8, nb_col=8, subsample=(4, 4),
                            activation='relu', border_mode='valid')(inputs)
        model = Convolution2D(nb_filter=64, nb_row=4, nb_col=4, subsample=(2, 2),
                            activation='relu', border_mode='valid')(model)
        model = Convolution2D(nb_filter=64, nb_row=3, nb_col=3, subsample=(1, 1),
                            activation='relu', border_mode='valid')(model)
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
        return a, y, loss, grad_update

    def get_initial_state(self, observation):
        observation = resize(rgb2gray(observation), (FRAME_WIDTH, FRAME_HEIGHT)) * 255
        observation = np.uint8(observation)
        state = [observation for _ in xrange(STATE_LENGTH)]
        state = np.stack(state, axis=0)
        return state

    def get_action(self, state):
        action = self.repeat_action

        if self.time_step % ACTION_FREQ == 0:
            if random.random() <= self.epsilon or self.time_step <= REPLAY_START_SIZE:
                action = random.randrange(self.num_actions)
            else:
                action = np.argmax(self.q_values.eval(feed_dict={self.s: [state]}))
            self.repeat_action = action

        # Epsilon decay
        if self.epsilon > FINAL_EPSILON and self.time_step > REPLAY_START_SIZE:
            self.epsilon -= self.epsilon_decay

        return action

    def run(self, state, action, reward, terminal, observation):
        next_state = np.append(observation, state[1:, :, :], axis=0)

        reward = np.int8(np.sign(reward))

        # Store transition in replay memory
        self.D.append((state, action, reward, next_state, terminal))
        if len(self.D) > NUM_REPLAY_MEMORY:
            self.D.popleft()

        # Train network
        if self.time_step > REPLAY_START_SIZE:
            if self.time_step % TRAIN_FREQ == 0:
                self.train_network()

        # Update target network
        if self.time_step % UPDATE_FREQ == 0:
            self.sess.run(self.update_target_network_params)

        # Save network
        if self.time_step % SAVE_FREQ == 0:
            save_path = self.saver.save(self.sess, SAVE_NETWORK_PATH + '/network', global_step=self.time_step)
            print 'Successfully saved: ', save_path

        self.total_reward += np.sign(reward)
        self.total_max_q += np.max(self.q_values.eval(feed_dict={self.s: [state]}))

        if terminal:
            # Write summaries
            if self.time_step > REPLAY_START_SIZE + TRAIN_FREQ:
                stats = [self.total_reward,
                        self.total_max_q / float(self.episode_time),
                        self.episode_time,
                        self.total_loss / (float(self.episode_time) / float(TRAIN_FREQ))]
                for i in xrange(len(stats)):
                    self.sess.run(self.update_ops[i], feed_dict={
                        self.summary_placeholders[i]: float(stats[i])
                    })
                summary_str = self.sess.run(self.summary_op)
                self.summary_writer.add_summary(summary_str, self.episode_step)

            # Debug
            if self.time_step <= REPLAY_START_SIZE:
                mode = 'random'
            elif REPLAY_START_SIZE < self.time_step <= REPLAY_START_SIZE + EXPLORATION_STEPS:
                mode = 'explore'
            else:
                mode = 'exploit'
            print 'EPISODE: {0:6d} / TIMESTEP: {1:8d} / DURATION: {2:5d} / EPSILON: {3:.5f} / TOTAL_REWARD: {4:3.0f} / AVG_MAX_Q: {5:2.4f} / AVG_LOSS: {6:.5f} / MODE: {7}'.format(
                self.episode_step, self.time_step, self.episode_time, self.epsilon,
                self.total_reward, self.total_max_q / float(self.episode_time),
                self.total_loss / (float(self.episode_time) / float(TRAIN_FREQ)), mode)

            self.total_reward = 0
            self.total_max_q = 0
            self.total_loss = 0
            self.episode_time = 0
            self.episode_step += 1

        self.episode_time += 1
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

        # Clip all positive rewards at 1 and all negative rewards at -1,
        # leaving 0 rewards unchanged
        # reward_batch = np.sign(reward_batch)

        # Convert True to 1, False to 0
        terminal_batch = np.array(terminal_batch) + 0

        target_q_values_batch = self.target_q_values.eval(feed_dict={self.st: next_state_batch})

        y_batch = reward_batch + (1 - terminal_batch) * GAMMA * np.max(target_q_values_batch)

        loss, _ = self.sess.run([self.loss, self.grad_update], feed_dict={
            self.s: state_batch,
            self.a: action_batch,
            self.y: y_batch
        })

        self.total_loss += loss

    def setup_summaries(self):
        episode_total_reward = tf.Variable(0.)
        tf.scalar_summary('Episode/Total Reward', episode_total_reward)
        episode_avg_max_q = tf.Variable(0.)
        tf.scalar_summary('Episode/Average Max Q Value', episode_avg_max_q)
        episode_duration = tf.Variable(0.)
        tf.scalar_summary('Episode/Duration', episode_duration)
        episode_avg_loss = tf.Variable(0.)
        tf.scalar_summary('Episode/Average Loss', episode_avg_loss)
        summary_vars = [episode_total_reward, episode_avg_max_q,
                        episode_duration, episode_avg_loss]
        summary_placeholders = [tf.placeholder(tf.float32) for i in xrange(len(summary_vars))]
        update_ops = [summary_vars[i].assign(summary_placeholders[i])
            for i in xrange(len(summary_vars))]
        summary_op = tf.merge_all_summaries()
        return summary_placeholders, update_ops, summary_op

    def load_network(self):
        checkpoint = tf.train.get_checkpoint_state(SAVE_NETWORK_PATH)
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            print 'Successfully loaded: ', checkpoint.model_checkpoint_path
        else:
            print 'Training new network...'

    def get_action_play(self, state):
        if random.random() <= 0.05:
            action = random.randrange(self.num_actions)
        else:
            action = np.argmax(self.q_values.eval(feed_dict={self.s: [state]}))

        return action


def preprocess(observation):
    observation = resize(rgb2gray(observation), (FRAME_WIDTH, FRAME_HEIGHT)) * 255
    observation = np.uint8(observation)
    return np.reshape(observation, (1, FRAME_WIDTH, FRAME_HEIGHT))


def main():
    env = gym.make(ENV_NAME)
    agent = Agent(num_actions=env.action_space.n)

    if TRAIN:
        for eposode in xrange(NUM_EPISODES):
            terminal = False
            observation = env.reset()
            state = agent.get_initial_state(observation)
            while not terminal:
                # env.render()
                action = agent.get_action(state)
                observation, reward, terminal, _ = env.step(action)
                observation = preprocess(observation)
                state = agent.run(state, action, reward, terminal, observation)
    else:
        for episode in xrange(10):
            terminal = False
            observation = env.reset()
            state = agent.get_initial_state(observation)
            while not terminal:
                env.render()
                action = agent.get_action_play(state)
                observation, reward, terminal, _ = env.step(action)
                observation = preprocess(observation)
                state = np.append(observation, state[1:, :, :], axis=0)


if __name__ == '__main__':
    main()
