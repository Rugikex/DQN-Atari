import os
import sys

import gymnasium as gym
import numpy as np
from tensorflow.keras.models import load_model

sys.path.append(os.path.join(os.getcwd()))

from classes.stacked_frames import StackedFrames
from global_functions import get_model_path


game_name = sys.argv[1]

env = gym.make(
    game_name,
    mode=int(sys.argv[2]),
    difficulty=int(sys.argv[3]),
    obs_type='rgb',
    frameskip=5,
    full_action_space=True,
    render_mode='human'
)


model_path, _ = get_model_path(game_name, sys.argv[4])
agent = load_model(os.path.join('models', game_name, model_path))

stacked_frames = StackedFrames(4)

T = 1000

state, info = env.reset()
stacked_frames.reset(state)
total_reward = 0
lives = info['lives']

print(f"Playing {game_name} with model {model_path}")

t = 0
while True:
    q_values = agent(stacked_frames.get_frames())
    action = np.argmax(q_values)

    next_state, reward, done, _, info = env.step(action)
    stacked_frames.append(next_state)
    if info['lives'] != lives:
        lives = info['lives']
        reward = -1
    real_reward = np.sign(reward)

    total_reward += real_reward

    if done:
        break

    state = next_state
    t += 1


env.close()

print(f'Total reward: {total_reward} in {t} steps')
