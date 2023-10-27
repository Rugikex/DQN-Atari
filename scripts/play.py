import os
import sys

import gymnasium as gym
import numpy as np

sys.path.append(os.path.join(os.getcwd()))

from classes.dql import DeepQLearning
from classes.stacked_frames import StackedFrames


game_name = sys.argv[1]

env = gym.make(
    game_name,
    mode=sys.argv[2],
    difficulty=sys.argv[3],
    obs_type='rgb',
    frameskip=6,
    full_action_space=True,
    render_mode='human'
)


max_file = None

if sys.argv[4]:
    max_file = f'ep_{sys.argv[4]}'
    if not os.path.isfile(os.path.join('models', game_name, max_file)):
        raise Exception('No model found')
else:
    max_number = 0
    for filename in os.listdir(os.path.join('models', game_name)):
        if filename.startswith('ep_'):
            # Extract the number of episode from the filename
            number = int(filename.split('_')[1])

            # Check if this number is greater than the current max_number
            if number > max_number:
                max_number = number
                max_file = filename

if not max_file:
    raise Exception('No model found')


agent = DeepQLearning(env.action_space.n)
agent.load_weights(os.path.join('models', game_name, max_file))

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
