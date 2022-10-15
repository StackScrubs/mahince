from sklearn.preprocessing import KBinsDiscretizer
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import gym

env = gym.make("CartPole-v1")

OBSERVATION_SPACE_N = env.observation_space.shape[0]
ACTION_SPACE_N = env.action_space.n
DISCRETIZE_N = 16385
DISCRETIZED_INDEX_OFFSET = np.arange(0, OBSERVATION_SPACE_N, 1) * DISCRETIZE_N

n_bins = (DISCRETIZE_N, ) * OBSERVATION_SPACE_N


def discretize(observations):
    """Convert continous state intro a discrete state"""
    est = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')
    X = [
        env.observation_space.low,
        env.observation_space.high,
    ]

    est.fit(X)
    T = est.transform([observations])
    return tuple(map(int, T[0]))


def discretize_idx(observations):
    return discretize(observations) + DISCRETIZED_INDEX_OFFSET


Q_table = np.zeros((OBSERVATION_SPACE_N * DISCRETIZE_N, ACTION_SPACE_N))

ALPHA = 1  # [0,1]
GAMMA = 1  # [0,1]

EPISODES = 100_000
scores = np.zeros(EPISODES)
for e in tqdm(range(EPISODES)):
    # Discretize state into buckets
    obs = env.reset()[0]
    state = discretize_idx(obs)

    epsilon = max(.01, min(1., 1. - np.log10((e + 1) / 25)))
    terminated = False
    truncated = False
    reward = 0

    while not terminated or truncated:
        # policy action
        action = np.argmax(np.max(Q_table[state, :], axis=0))

        # insert random action
        # epsilon greedy strategy
        if np.random.random() < epsilon:
            action = env.action_space.sample()  # explore

        # increment enviroment
        obs, _, terminated, truncated, _ = env.step(action)
        new_state = discretize_idx(obs)

        # Update Q-Table
        Q_table[state, action] += epsilon*ALPHA*(
            reward +
            epsilon*GAMMA*np.max(Q_table[new_state, :]) -
            Q_table[state, action]
        )

        state = new_state
        reward += 1

    scores[e] = reward

print(scores)

plt.plot(scores,  c='blue', label='epochs')
plt.savefig("plot.png")
