import os
import random
import sys

import gymnasium as gym
import numpy as np
import torch

sys.path.append(os.path.join(os.getcwd()))

from classes.dqn import DeepQNetwork
from classes.env_wrapper import AtariWrapper
from classes.stacked_frames import StackedFrames
from global_functions import get_model_path

import parameters


device = parameters.device
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

env = AtariWrapper(env, skip_frames=4, play=True)

model_path = get_model_path(game_name, sys.argv[4])
states_dict = torch.load(os.path.join('models', game_name, model_path))
# Load the model
agent = DeepQNetwork(env.action_space.n).to(device)
agent.load_state_dict(states_dict['state_dict'])
agent.eval()


stacked_frames = StackedFrames(4)

T = 1000

state, _ = env.reset()
stacked_frames.reset(state)
total_reward = 0

print("=======")
print(f"Playing {game_name} with model {model_path}")
print("=======")

t = 0
while True:
    # TODO: Decomment this to play randomly
    # if random.random() < 0.05:
    #     action = env.action_space.sample()
    # else:
    observation = torch.tensor(stacked_frames.get_frames(), dtype=torch.float32).unsqueeze(0).to(device)
    q_values = agent(observation)
    action = torch.argmax(q_values).item()
    print(q_values)

    state, reward, done, _, info = env.step(action)
    previous_state = info['previous_state']

    stacked_frames.append(state, previous_state)
    real_reward = np.sign(reward)

    total_reward += real_reward

    t += 1
    if done:
        break


env.close()

print(f'Total reward: {total_reward} in {t} steps')
