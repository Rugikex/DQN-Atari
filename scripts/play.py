import os
import sys

import gymnasium as gym
import numpy as np
from tensorflow.keras.models import load_model

sys.path.append(os.path.join(os.getcwd()))

from classes.dqn import DeepQNetwork
from classes.stacked_frames import StackedFrames
from global_functions import get_model_path


game_name = sys.argv[1]

env = gym.make(
    game_name,
    mode=int(sys.argv[2]),
    difficulty=int(sys.argv[3]),
    frameskip=1,
    repeat_action_probability=0.0,
    obs_type='rgb',
    full_action_space=False,
    render_mode='human'
)


model_path, _ = get_model_path(game_name, sys.argv[4])
agent = DeepQNetwork(env.action_space.n)
agent.build((84, 84, 4))
agent.load_weights(os.path.join('models', game_name, model_path))

stacked_frames = StackedFrames(4)

T = 1000

state, info = env.reset()
stacked_frames.reset(state)
total_reward = 0
lives = info['lives']
previous_state = state

print("=======")
print(f"Playing {game_name} with model {model_path}")
print("=======")

t = 0
skip_frames = 4

while True:
    q_values = agent(stacked_frames.get_frames())
    action = np.argmax(q_values)

    next_state = previous_state
    sum_reward = 0.0
    done: bool
    info: dict

    for skip in range(skip_frames):
        if skip <= skip_frames - 2:
            previous_state = next_state

        next_state, reward, done, _, info = env.step(action)
        if info['lives'] != lives:
            lives = info['lives']
            reward = -1.0

        sum_reward += np.sign(reward)

        if done:
            break

    stacked_frames.append(next_state, previous_state)
    real_reward = np.sign(sum_reward)

    total_reward += real_reward

    if done:
        break

    state = next_state
    t += 1


env.close()

print(f'Total reward: {total_reward} in {t} steps')
