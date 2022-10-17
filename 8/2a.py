from grid_world import GridWorldEnv, INTENT_SQUARE
import numpy as np

env = GridWorldEnv(grid_world_size=10, render_mode='human')

STATE_SPACE_N = env.state_space_n
ACTION_SPACE_N = env.action_space_n

Q_table = np.zeros((STATE_SPACE_N**2, ACTION_SPACE_N))
print(Q_table.shape)

ALPHA = 0.1  # range [0,1]
GAMMA = 0.4  # range [0,1]

def get_state_index(state):
    return state.pos.x * STATE_SPACE_N + state.pos.y

EPISODES = 100_000
for e in range(EPISODES):
    env.reset()
    
    state = get_state_index(env.state)
    
    # epsilon greedy strategy
    epsilon = max(.01, min(1., 1. - np.log10((e + 1) / 25)))
    terminated = False
    truncated = False
    reward = 0

    while not terminated:
        if np.random.random() < epsilon:
            action = env.sample_action_space()  # explore
        else:
            # policy action
            action = np.argmax(Q_table[state])

        new_state, new_reward, terminated = env.step(action, np.argmax(Q_table, axis=1))
        reward += new_reward
        new_state = get_state_index(new_state)

        # Update Q-Table
        Q_table[state, action] += ALPHA*(
            reward +
            GAMMA*np.max(Q_table[new_state, action]) -
            Q_table[state, action]
        )

        state = new_state
    # print(Q_table)

env._render()

env.close()
