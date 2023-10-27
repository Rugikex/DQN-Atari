import os

import gymnasium as gym
import numpy as np

from dql import DeepQLearning
from stacked_frames import StackedFrames


game_name = 'Assault'
version = 4
difficulty = 0

env = gym.make(
    f"{game_name}-v{version}",
    mode=0,
    difficulty=difficulty,
    obs_type='rgb',
    frameskip=6,
    full_action_space=True,
    render_mode='human'
)

agent = DeepQLearning(env.action_space.n)

agent.load_weights(os.path.join(os.getcwd(), 'models', f'{game_name}_v{version}'))

stacked_frames = StackedFrames(4)

T = 1000

state, info = env.reset()
stacked_frames.reset(state)
total_reward = 0
lifes = info['lives']

for t in range(1, T + 1):
    q_values = agent(stacked_frames.get_frames())
    action = np.argmax(q_values)

    next_state, reward, done, _, info = env.step(action)
    stacked_frames.append(next_state)
    if info['lives'] != lifes:
        lifes = info['lives']
        reward = -1
    real_reward = np.sign(reward)

    total_reward += real_reward

    if done:
        break

    state = next_state


env.close()

print(f'Total reward: {total_reward} in {t} steps')
