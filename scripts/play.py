import os
import random
import sys

import gymnasium as gym
import torch

sys.path.append(os.path.join(os.getcwd()))

from classes.dqn import DeepQNetwork
from classes.env_wrapper import AtariWrapper
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
    obs_type="rgb",
    full_action_space=False,
    render_mode="human",
)

env = AtariWrapper(env, skip_frames=4, play=True)

model_path = get_model_path(game_name, sys.argv[4])
states = torch.load(os.path.join("models", game_name, model_path))
# Load the model
agent = DeepQNetwork(env.action_space.n).to(device)
agent.load_state_dict(states["state_dict"])
agent.eval()

episodes = states["episodes"]
steps = states["steps"]
hours = states["hours"]

T = 1000

state, _ = env.reset()
total_reward = 0

print("=======")
print(f"Playing {game_name} with model {model_path}, trained for {hours} hours")
print("=======")

t = 0
while True:
    if random.random() < 0.05:
        action = env.action_space.sample()
    else:
        observation = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = agent(observation)
        action = torch.argmax(q_values).item()

    state, reward, done, _, info = env.step(action)

    total_reward += reward

    t += 1
    if done:
        break


env.close()

print(f"Total reward: {total_reward} in {t} steps")
