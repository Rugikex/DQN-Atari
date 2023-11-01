import os
import random
import sys

import gymnasium as gym
import numpy as np
import torch

sys.path.append(os.path.join(os.getcwd()))

from classes.dqn import DeepQNetwork
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

# Get fire action if exists to not get stuck in the game
check_fire_action = [action for action, meaning in enumerate(env.unwrapped.get_action_meanings()) if meaning == 'FIRE']
if len(check_fire_action) > 0:
    fire_action = check_fire_action[0]
else:
    fire_action = None


model_path, _ = get_model_path(game_name, sys.argv[4])
# Load the model
agent = DeepQNetwork(env.action_space.n).to(device)
agent.load_state_dict(torch.load(os.path.join('models', game_name, model_path)))
agent.eval()


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
has_fire_action = False

for _ in range(random.randint(1, 30)):
    action = env.action_space.sample()
    if fire_action is not None and action == fire_action:
        has_fire_action = True
    next_state, reward, done, _, info = env.step(action)
    if info['lives'] != lives:
        lives = info['lives']
        done = True
    stacked_frames.append(next_state, previous_state)
    previous_state = next_state
    if done:
        state, info = env.reset()
        stacked_frames.reset(state)
        lives = info['lives']
        previous_state = state


if fire_action is not None:
    action = fire_action


while True:
    # if random.random() < 0.05:
    #     action = env.action_space.sample()
    # else:
    observation = torch.tensor(stacked_frames.get_frames(), dtype=torch.float32).unsqueeze(0).to(device)
    q_values = agent(observation)
    action = torch.argmax(q_values).item()

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
