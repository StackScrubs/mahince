from base64 import encode
from typing import Tuple
from sklearn.preproccessing import KBinsDiscretizer
import gym
import numpy as np

env = gym.make("CartPole-v1", render_mode="human")
prev_states, info = env.reset(seed=42)
STATE_SPACE_N = env.observation_space
ACTION_SPACE_N = env.action_space.n

q_table = np.zeros((STATE_SPACE_N, ACTION_SPACE_N))

ALPHA = 0.9 # <0,1>
GAMMA = 0.89 # <0,1>

lower_bounds = [ env.observation_space.low[2], -math.radians(50) ]
upper_bounds = [ env.observation_space.high[2], math.radians(50) ]

def cont_to_disc(pos, vel, ang, ang_vel) -> Tuple[int,...]:
    est = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')
    est.fit([lower_bounds, upper_bounds ])
    return tuple(map(int, est.transform([[pos, vel, ang, ang_vel]])[0]))


for _ in range(5): 
    action = env.action_space.sample()
    states, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated: 
        prev_states, info = env.reset() 
        
    for state_idx in range(STATE_SPACE_N): 
        prev = q_table[state_idx, action] 
        q_table[state_idx, action] = prev + ALPHA*(reward + GAMMA*np.max(q_table[state_idx]) - prev) 
        #q_table[:, action] = q_table[:, action] + ALPHA*(reward + GAMMA*np.argmax(np.max(q_table, axis=0)) - q_table[:, action]) 

# prev_states = statesenv.close()
# q_table[obs, action] = q_table[obs, action] + (
#     ALPHA * (reward + GAMMA * np.max(q_table[obs, :]) - q_table[obs, action])
# )
