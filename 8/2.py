from sklearn.preprocessing import KBinsDiscretizer
import numpy as np 
import time, math, random
from typing import Tuple

# import gym 
import gym

env = gym.make("CartPole-v1", render_mode="human")

OBSERVATION_SPACE_N = env.observation_space.shape[0]
ACTION_SPACE_N = env.action_space.n
DISCRETIZE_N = 3

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
Q_table = np.zeros(n_bins + (ACTION_SPACE_N,))
print(Q_table)

ALPHA = 0.9 # <0,1>
GAMMA = 0.89 # <0,1>

EPISODES = 10000 
for e in range(EPISODES):
    
    # Discretize state into buckets
    obs = env.reset()[0]
    current_state, done = discretizer(*obs), False
    
    while not done:

        # policy action 
        #np.argmax(Q_table, axis=1)
        #action = np.argmax(np.max(Q_table[current_state])) # exploit
        # action = np.argmax(np.max(Q_table, axis=0))
        #action = np.argmin(np.min(Q_table, axis=0))

        # Maximize what gives us the biggest reward
        max_obs = np.argmax(np.abs(obs / env.observation_space.high))
        action = np.argmax(Q_table[max_obs])
        print(action)
        
        # insert random action
        #epsilon greedy strategy
        exploration_rate = max(.01, min(1., 1. - math.log10((e + 1) / 25)))
        if np.random.random() < exploration_rate : 
            action = env.action_space.sample() # explore 

        # increment enviroment
        obs, reward, done, _, _ = env.step(action)
        new_state = discretizer(*obs)
        # Update Q-Table
        # learning_rate = max(0.01, min(1.0, 1.0 - math.log10((e + 1) / 25)))
        Q_table[current_state][action] = (1-ALPHA)*Q_table[current_state][action] + GAMMA*reward + 1 * np.max(Q_table[new_state])
        # SUS DETECTED
        #Q_table[max_obs][action] = (1-ALPHA)*Q_table[max_obs][action] + GAMMA*reward + 1 * np.max(Q_table[max_obs])
        #BELLMAN: Q_table[current_state][action] = Q_table[current_state][action] * ALPHA*(reward + GAMMA*ESTIMATE_OPTIMAL_FUTURE - Q_table[current_state][action])
        print(current_state)
        print(new_state)
        print(reward)
        print(current_state, action)
        #Q_table[current_state, action] = Q_table[current_state, action] * ALPHA*(reward + GAMMA*np.max(Q_table[new_state, :]) - Q_table[current_state, action])
        print(f"DA TAYBEL {Q_table[current_state, action]}")

        #        A0    A1
        #     |-----|-----|
        # S0  |  1  |  5  |
        # S1  |  2  |  6  |
        # S2  |  3  |  7  |
        # S3  |  4  |  8  |
        #     |-----|-----|

        print(Q_table)
        current_state = new_state
        
        # Render the cartpole environment
        env.render()
