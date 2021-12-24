import argparse
import datetime
import os

import gym
import numpy as np
import collections
import tensorflow.compat.v1 as tf
from buffer import ReplayBuffer
from functools import partial

tf.disable_v2_behavior()

env = gym.make('CartPole-v1')

np.random.seed(1)


class PolicyNetwork:
    def __init__(self, state_size, action_size, learning_rate, name='policy_network'):
        # Estimates the policy's action distribution for a given input state
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate

        with tf.variable_scope(name):
            self.state = tf.placeholder(tf.float32, [None, self.state_size], name="state")
            self.action = tf.placeholder(tf.int32, [self.action_size], name="action")
            self.R_t = tf.placeholder(tf.float32, name="total_rewards")

            self.W1 = tf.get_variable("W1", [self.state_size, 12],
                                      initializer=tf.keras.initializers.glorot_normal(seed=0))
            self.b1 = tf.get_variable("b1", [12], initializer=tf.keras.initializers.glorot_normal(seed=0))
            self.W2 = tf.get_variable("W2", [12, self.action_size],
                                      initializer=tf.keras.initializers.glorot_normal(seed=0))
            self.b2 = tf.get_variable("b2", [self.action_size], initializer=tf.keras.initializers.glorot_normal(seed=0))

            self.Z1 = tf.add(tf.matmul(self.state, self.W1), self.b1)
            self.A1 = tf.nn.relu(self.Z1)
            self.output = tf.add(tf.matmul(self.A1, self.W2), self.b2)

            # Model Architecture: (I,H1,O) = (4,12,2),  n_params = 86

            # Softmax probability distribution over actions
            self.actions_distribution = tf.squeeze(tf.nn.softmax(self.output))
            # Loss with negative log probability
            self.neg_log_prob = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.output, labels=self.action)
            self.loss = tf.reduce_mean(self.neg_log_prob * self.R_t)

            self.summary = tf.summary.scalar('loss_p', self.loss)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)


class ValueNetwork:
    def __init__(self, state_size, gamma, learning_rate, name='value_network'):
        # Used for a baseline-function which is independent in the action taken. b(s_t)
        self.state_size = state_size
        self.learning_rate = learning_rate
        self.gamma = gamma

        with tf.variable_scope(name):
            self.state = tf.placeholder(tf.float32, [None, self.state_size], name="state")
            self.next_state = tf.placeholder(tf.float32, [None, self.state_size], name="next_state")
            self.reward = tf.placeholder(tf.float32, [None, 1], name="reward")
            self.done = tf.placeholder(tf.float32, [None, 1], name="done")

            self.W1 = tf.get_variable("W1", [self.state_size, 12],
                                      initializer=tf.keras.initializers.glorot_normal(seed=0))
            self.b1 = tf.get_variable("b1", [12], initializer=tf.keras.initializers.glorot_normal(seed=0))
            self.W2 = tf.get_variable("W2", [12, 1], initializer=tf.keras.initializers.glorot_normal(seed=0))
            self.b2 = tf.get_variable("b2", [1], initializer=tf.keras.initializers.glorot_normal(seed=0))

            self.Z1 = tf.add(tf.matmul(self.state, self.W1), self.b1)
            self.A1 = tf.nn.relu(self.Z1)
            self.output = tf.add(tf.matmul(self.A1, self.W2), self.b2)

            # Model Architecture: (I,H1,O) = (4,12,1),  n_params = 73

            self.W1_target = tf.get_variable("W1_target", [self.state_size, 12])
            self.b1_target = tf.get_variable("b1_target", [12])
            self.W2_target = tf.get_variable("W2_target", [12, 1])
            self.b2_target = tf.get_variable("b2_target", [1])

            self.Z1_target = tf.add(tf.matmul(self.next_state, self.W1_target), self.b1_target)
            self.A1_target = tf.nn.relu(self.Z1_target)
            self.output_target = tf.add(tf.matmul(self.A1_target, self.W2_target), self.b2_target)

            # Target-Model Architecture: (I,H1,O) = (4,12,1),  n_params = 73

            # V\hat (S_t) = R_t + \gamma * V^-\hat (S_{t+1})
            self.loss = tf.reduce_mean((self.output -
                                        (self.reward + self.gamma * self.output_target * (1 - self.done))) ** 2)
            # MSE loss
            self.summary = tf.summary.scalar('loss_v', self.loss)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

    def update_weights(self):
        # Target-Model <- Model
        self.W1_target.assign(self.W1)
        self.b1_target.assign(self.b1)
        self.W2_target.assign(self.W2)
        self.b2_target.assign(self.b2)


class CriticNetwork:
    def __init__(self, state_size, gamma, learning_rate, name='critic_network'):
        # Estimates the value-function of the actor's policy. V^(\pi_\theta). Also used as a baseline-function.
        self.state_size = state_size
        self.learning_rate = learning_rate
        self.gamma = gamma

        with tf.variable_scope(name):
            self.state = tf.placeholder(tf.float32, [None, self.state_size], name="state")
            self.Gt_1 = tf.placeholder(tf.float32, [None, 1], name="Gt_1")

            self.W1 = tf.get_variable("W1", [self.state_size, 12],
                                      initializer=tf.keras.initializers.glorot_normal(seed=0))
            self.b1 = tf.get_variable("b1", [12], initializer=tf.keras.initializers.glorot_normal(seed=0))
            self.W2 = tf.get_variable("W2", [12, 1], initializer=tf.keras.initializers.glorot_normal(seed=0))
            self.b2 = tf.get_variable("b2", [1], initializer=tf.keras.initializers.glorot_normal(seed=0))

            self.Z1 = tf.add(tf.matmul(self.state, self.W1), self.b1)
            self.A1 = tf.nn.relu(self.Z1)
            self.output = tf.add(tf.matmul(self.A1, self.W2), self.b2)

            # V\hat (S_t) = R_t + \gamma * V^-\hat (S_{t+1})
            self.loss = tf.reduce_mean((self.output - self.Gt_1) ** 2)
            self.summary = tf.summary.scalar('loss_c', self.loss)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)


# Define hyperparameters
state_size = 4
action_size = env.action_space.n

max_episodes = 5000
max_steps = 501
discount_factor = 0.99
learning_rate_p = 0.0004
learning_rate_v = 0.0012
learning_rate_c = 0.0012
learning_rate_q = learning_rate_v

render = False
debug = True
# Initialize the policy network
tf.reset_default_graph()
policy = PolicyNetwork(state_size, action_size, learning_rate_p)
value_net = ValueNetwork(state_size, discount_factor, learning_rate_v)
critic_net = CriticNetwork(state_size, discount_factor, learning_rate_c)

# for value network:
batch_size = 256
capacity = 100000
update_freq = 50
replay_buffer = ReplayBuffer(capacity)

# tensorboard variables
base_log_path = os.path.join(os.getcwd(), "logs/fit/tensorboard")
global_v_step = 0
global_p_step = 0


# Start training the agent with REINFORCE algorithm
def get_feed_dict(name, transition, total_discounted_return=0):
    feed_dicts = {
        'value': {value_net.state: transition.state, value_net.next_state: transition.next_state,
                  value_net.done: [transition.done], value_net.reward: [transition.reward]},
        'policy': {policy.state: transition.state, policy.R_t: total_discounted_return,
                   policy.action: transition.action}}
    return feed_dicts[name]


def reinforce(sess, episode_transitions, episode, writer=None, done=False):
    """
    :param sess: The tensorflow session that is running
    :param episode_transitions: list of all the transitions in the episode
    :param episode: number of the current episode
    :param writer: tensorboard writer object
    :param done: if the episode is terminated (used for generalization of the training step function)
    :return: None
    """
    global global_p_step
    if not done:  # this algorithm must finish the episode in order to apply training steps
        return
    for t, transition in enumerate(episode_transitions):
        total_discounted_return = sum(
            discount_factor ** i * t.reward for i, t in enumerate(episode_transitions[t:]))  # Gt
        feed_dict = get_feed_dict('policy', transition, total_discounted_return)
        summ, _, loss_p = sess.run([policy.summary, policy.optimizer, policy.loss], feed_dict)
        writer.add_summary(summ, global_step=global_p_step)
        global_p_step += 1


def reinforce_advantage(sess, episode_transitions, episode, writer=None, done=False):
    """
        :param sess: The tensorflow session that is running
        :param episode_transitions: list of all the transitions in the episode
        :param episode: number of the current episode
        :param writer: tensorboard writer object
        :param done: if the episode is terminated (used for generalization of the training step function)
        :return: None
    """
    global global_p_step
    global global_v_step
    if not done:  # this algorithm must finish the episode in order to apply training steps
        return
    global_v_step = global_p_step
    if len(replay_buffer) > 1e3:  # wait until replay buffer have sufficient experiences.
        for _ in episode_transitions:
            states, _, rewards, next_states, done = replay_buffer.sample(batch_size=batch_size)
            feed_dict_v = {value_net.state: states, value_net.reward: rewards,
                           value_net.next_state: next_states, value_net.done: done}
            summ, _, loss_v = sess.run([value_net.summary, value_net.optimizer, value_net.loss], feed_dict_v)
            writer.add_summary(summ, global_step=global_v_step)
            global_v_step += 1
        if (episode + 1) % update_freq == 0:
            value_net.update_weights()

    for t, transition in enumerate(episode_transitions):
        total_discounted_return = sum(
            discount_factor ** i * t.reward for i, t in enumerate(episode_transitions[t:]))  # Gt
        advantage = total_discounted_return - sess.run(value_net.output, {value_net.state: transition.state})
        feed_dict_p = {policy.state: transition.state, policy.R_t: advantage,
                       policy.action: transition.action}
        summ, _, loss_p = sess.run([policy.summary, policy.optimizer, policy.loss], feed_dict_p)
        writer.add_summary(summ, global_step=global_p_step)
        global_p_step += 1


def actor_critic(sess, episode_transitions, episode, writer=None, done=False, td_steps=1):
    """
        :param sess: The tensorflow session that is running
        :param episode_transitions: list of all the transitions in the episode
        :param episode: number of the current episode
        :param writer: tensorboard writer object
        :param done: if the episode is terminated (used for generalization of the training step function)
        :param td_steps: number of real experience steps used in the return estimation
        :return: None
    """
    global global_p_step
    global global_v_step

    def train_step(index, traj_len):
        """
        :param index: index of current transition within the trajectory
        :param traj_len: number of real experience steps used in the return estimation
        :return:
        """
        global global_p_step
        # state, action, reward are the current state, action, reward we look at
        # next_state, done are the end of trajectory next_state and done.
        state, action, reward, next_state, _done = states[index], actions[index], rewards[index], next_states[-1], \
                                                   dones[-1]

        ret_reward = (discount_factor ** np.arange(traj_len) * rewards[index:]).sum()
        # estimate of G_t
        estimated_discounted_return = ret_reward + discount_factor ** (traj_len) * \
                                      sess.run(critic_net.output, {critic_net.state: next_state}) * (1 - _done)

        # estimate of A_t
        advantage = estimated_discounted_return - sess.run(critic_net.output, {critic_net.state: state})
        summ, _, loss_c = sess.run(
            [critic_net.summary, critic_net.optimizer, critic_net.loss],
            feed_dict={critic_net.state: state,
                       critic_net.Gt_1: estimated_discounted_return})
        writer.add_summary(summ, global_step=global_p_step)
        feed_dict_p = {policy.state: state, policy.R_t: advantage,
                       policy.action: action}
        summ, _, loss_p = sess.run([policy.summary, policy.optimizer, policy.loss], feed_dict_p)
        writer.add_summary(summ, global_step=global_p_step)
        global_p_step += 1

    global_v_step = global_p_step
    if not done and len(episode_transitions) < td_steps:  # not enough steps to evaluate G_t by N-steps ahead
        return
    # unpack the last N-steps transitions into corresponding np.arrays
    states, actions, rewards, next_states, dones = tuple(map(np.asarray, zip(*episode_transitions[-td_steps:])))
    traj_len = len(episode_transitions[-td_steps:])
    if done:  # apply training steps for the remaining of the trajectory.
        for i in range(traj_len):
            train_step(i, traj_len - i)
    else:  # apply training step only for the current state we look at (at index 0).
        train_step(0, traj_len)


def train(td_steps=0, max_episodes=5000, exp_dir='def_dir', exp_name='No_Name'):
    """
    :param td_steps: determines the algorithm used.
                     if td_steps<0 -> reinforce,
                     if td_steps=0 -> advantage using replay-buffer and target-network
                     if td_steps=N>0 -> uses N-steps in the actor-critic method.
                     For advantage only, without replay-buffer and target-network, use td_steps>500
    :param max_episodes: max episodes to train
    :param exp_dir: directory for saving logs within the log-dir
    :param exp_name: name of the specific experiment
    :return:
    """
    if td_steps < 0:
        train_fun = reinforce
    elif td_steps == 0:
        train_fun = reinforce_advantage
    else:
        train_fun = partial(actor_critic, td_steps=td_steps)

    path = os.path.join(base_log_path, exp_dir)

    with tf.Session() as sess:
        writer_train = tf.summary.FileWriter(os.path.join(path,
                                                          exp_name + '_' + datetime.datetime.now().strftime(
                                                              "%d%m%Y-%H%M")), sess.graph)
        sess.run(tf.global_variables_initializer())
        solved = False
        Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])
        episode_rewards = np.zeros(max_episodes)
        average_rewards = 0.0
        # variables for tensorboard
        total_score = tf.placeholder(tf.int32, name="total_score")
        avg_score = tf.placeholder(tf.float32, name="avg_score")
        summary_scores = tf.summary.merge(
            [tf.summary.scalar('Score', total_score), tf.summary.scalar('Avg_Score', avg_score)])

        for episode in range(max_episodes):
            state = env.reset()
            state = state.reshape([1, state_size])
            episode_transitions = []
            score = 0
            for step in range(max_steps):
                actions_distribution = sess.run(policy.actions_distribution, {policy.state: state})
                action = np.random.choice(np.arange(len(actions_distribution)), p=actions_distribution)
                next_state, reward, done, _ = env.step(action)
                score += reward
                next_state = next_state.reshape([1, state_size])

                if render:
                    env.render()

                action_one_hot = np.zeros(action_size)
                action_one_hot[action] = 1
                transition = Transition(state=state, action=action_one_hot, reward=reward, next_state=next_state,
                                        done=done)
                if td_steps == 0:
                    replay_buffer.push(transition)
                episode_transitions.append(transition)
                episode_rewards[episode] += reward

                train_fun(sess, episode_transitions, episode, writer=writer_train, done=done)
                if done:
                    if episode > 98:
                        # Check if solved
                        average_rewards = np.mean(episode_rewards[(episode - 99):episode + 1])
                    print(
                        "Episode {} Reward: {} Average over 100 episodes: {}".format(episode, episode_rewards[episode],
                                                                                     round(average_rewards, 2)))
                    if average_rewards > 475:
                        print(' Solved at episode: ' + str(episode))
                        solved = True
                    break
                state = next_state

            summ = sess.run(summary_scores, feed_dict={total_score: score, avg_score: average_rewards})  #
            writer_train.add_summary(summ, global_step=episode)
            if solved:
                break


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default=None)
    parser.add_argument('--N', type=int, default=-1)
    parser.add_argument('--exp', type=str, default='No_Name')
    parser.add_argument('--n_episodes', type=int, default=5000)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    train(td_steps=args.N, max_episodes=args.n_episodes, exp_dir=args.exp, exp_name=args.exp)
