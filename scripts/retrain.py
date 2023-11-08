import os
import re
import sys

import gymnasium as gym
import torch
import torch.optim as optim

sys.path.append(os.path.join(os.getcwd()))

from classes.dqn import DeepQNetwork
from global_functions import get_model_path, train_model
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
    render_mode="rgb_array",
)

model_path = get_model_path(game_name, sys.argv[5])
states = torch.load(os.path.join("models", game_name, model_path))
n_actions = env.action_space.n

agent = DeepQNetwork(n_actions).to(device)
agent.load_state_dict(states["state_dict"])

target_agent = DeepQNetwork(env.action_space.n).to(device)
target_agent.load_state_dict(states["target_state_dict"])

optimizer = optim.Adam(
    agent.parameters(),
    lr=0.000_25,
)
optimizer.load_state_dict(states["optimizer"])

hours_to_train = int(sys.argv[4])

episodes = states["episodes"]
steps = states["steps"]
hours = states["hours"]

parts_model_name = sys.argv[5].split("_")
model_name = "_".join(parts_model_name[:-1])

print("=======")
print(
    f"Retraining on {game_name} with model {model_path} (episodes: {episodes}, steps: {steps}, hours: {hours})"
)
print("=======")

episodes_done, steps_done, hours_done = train_model(
    agent,
    target_agent,
    env,
    optimizer,
    game_name,
    model_name,
    hours_to_train,
    episodes,
    steps,
    hours,
)

# Save model
torch.save(
    {
        "state_dict": agent.state_dict(),
        "target_state_dict": target_agent.state_dict(),
        "optimizer": optimizer.state_dict(),
        "episodes": episodes_done,
        "steps": steps_done,
        "hours": hours_done,
    },
    os.path.join("models", game_name, f"{model_name}_last.pt"),
)
