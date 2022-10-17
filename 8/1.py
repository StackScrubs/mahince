from sklearn.preprocessing import KBinsDiscretizer
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import gym

env = gym.make("CartPole-v1")

OBSERVATION_SPACE_N = env.observation_space.shape[0]
ACTION_SPACE_N = env.action_space.n
DISCRETIZE_N = 64
MAX_REWARD = 500

n_bins = (DISCRETIZE_N, ) * OBSERVATION_SPACE_N


def discretize(observations):
    """Convert continous state intro a discrete state"""
    est = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')
    X = [
        env.observation_space.low,
        env.observation_space.high,
    ]

    est.fit(X)
    T = est.transform([observations])[0]

    state = 0
    mult = 1
    for i in T:
        state += int(i) * mult
        mult *= DISCRETIZE_N

    return state


Q_table = np.zeros((DISCRETIZE_N**OBSERVATION_SPACE_N, ACTION_SPACE_N))

ALPHA = 0.1  # [0,1]
GAMMA = 0.4  # [0,1]

EPISODES = 100_000
scores = np.zeros(EPISODES)
for e in tqdm(range(EPISODES)):
    # Discretize state into buckets
    obs = env.reset()[0]
    state = discretize(obs)

    # epsilon greedy strategy
    epsilon = max(.01, min(1., 1. - np.log10((e + 1) / 25)))
    terminated = False
    truncated = False
    reward = 0

    while not terminated and not truncated:
        # policy action
        action = np.argmax(Q_table[state])

        # insert random action
        if np.random.random() < epsilon:
            action = env.action_space.sample()  # explore

        # increment enviroment
        obs, _, terminated, truncated, _ = env.step(action)
        new_state = discretize(obs)

        # Update Q-Table
        Q_table[state, action] += ALPHA*(
            reward +
            GAMMA*np.max(Q_table[new_state, :]) -
            Q_table[state, action]
        )

        state = new_state
        reward += 1

    scores[e] = reward

print(scores)

plt.plot(scores / MAX_REWARD,  c='blue', label='epochs')
plt.savefig("plot.png")
