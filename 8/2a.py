from grid_world import GridWorldEnv
# import numpy as np

env = GridWorldEnv(render_mode='human')

EPISODES = 100_000
for e in range(EPISODES):
    env.reset()
    terminated = False

    while not terminated:
        action = env.sample_action_space() # explore
        reward, terminated = env.step(action)


env._render()

env.close()