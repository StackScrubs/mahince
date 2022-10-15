from sklearn.preprocessing import KBinsDiscretizer
import numpy as np 
import time, math, random
from typing import Tuple
import matplotlib.pyplot as plt

# import gym 
import gym

env = gym.make("CartPole-v1", render_mode="human")

OBSERVATION_SPACE_N = env.observation_space.shape[0]
ACTION_SPACE_N = env.action_space.n
DISCRETIZE_N = 129

n_bins = ( DISCRETIZE_N, ) * OBSERVATION_SPACE_N
def discretizer( position , velocity , pole_angle, pole_angular_velocity ) -> Tuple[int,...]:
    """Convert continous state intro a discrete state"""
    est = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')
    X = [
        env.observation_space.low,
        env.observation_space.high,
    ]

    est.fit(X)
    T = est.transform([[position, velocity, pole_angle, pole_angular_velocity]])
    return tuple(map(int,T[0]))

# Q_table = np.zeros(n_bins + (env.action_space.n,))
Q_table = np.zeros((OBSERVATION_SPACE_N * DISCRETIZE_N, ACTION_SPACE_N))

ALPHA = 0.2     # <0,1>
GAMMA = 0.7    # <0,1>

EPISODES = 10
scores = np.zeros(EPISODES)
for e in range(EPISODES):
    print(e)
    # Discretize state into buckets
    obs = env.reset()[0]
    current_state = discretizer(*obs)
    
    terminated = False
    truncated = False
    reward = 0
    while not terminated or truncated:
          
        idx_add = np.arange(0, OBSERVATION_SPACE_N, 1) * DISCRETIZE_N
        current_state_idx = current_state + idx_add

        # policy action 
        action = np.argmax(np.max(Q_table[current_state_idx, :], axis=0))

        # insert random action
        # epsilon greedy strategy
        epsilon = max(.01, min(1., 1. - math.log10((e + 1) / 25)))
        if np.random.random() < epsilon: 
            action = env.action_space.sample() # explore 

        # increment enviroment
        obs, _, terminated, truncated, _ = env.step(action)

        new_state = discretizer(*obs)

        # Update Q-Table
        new_state_idx = new_state + idx_add
        Q_table[current_state_idx, action] = Q_table[current_state_idx, action] + ALPHA*(reward + GAMMA*np.max(Q_table[new_state_idx, :]) - Q_table[current_state_idx, action])

        current_state = new_state
        
        # Render the cartpole environment
        env.render()
        reward += 1
    scores[e] = reward

print(scores)

plt.plot(scores,  c='blue', label='epochs')
plt.show()
