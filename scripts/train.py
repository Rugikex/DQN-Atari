from collections import deque
import os
import sys

import gymnasium as gym
import torch
import torch.optim as optim

sys.path.append(os.path.join(os.getcwd()))

from classes.dqn import DeepQNetwork
from global_functions import train_model
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

n_actions = env.action_space.n

agent = DeepQNetwork(n_actions).to(device)
target_agent = DeepQNetwork(n_actions).to(device)
target_agent.load_state_dict(agent.state_dict())
target_agent.eval()

replay_memory = deque(maxlen=parameters.replay_memory_maxlen)

# optimizer = optim.RMSprop(
#     agent.parameters(),
#     lr=0.000_25,
#     momentum=0.95,
#     eps=0.01
# )
optimizer = optim.Adam(
    agent.parameters(),
    lr=0.000_25,
)

print("=======")
print(f"Training on {game_name}")
print("=======")

episode_done = train_model(
    agent,
    target_agent,
    env,
    replay_memory,
    optimizer,
)


# Save model
if not os.path.exists(os.path.join("models", game_name)):
    os.makedirs(os.path.join("models", game_name))

model_name: str
if sys.argv[5]:
    model_name = sys.argv[5]
else:
    counter = 0
    for file in os.listdir(os.path.join("models", game_name)):
        if file.startswith("model_"):
            counter += 1
    model_name = f"model_{counter}"

state = {
    "state_dict": agent.state_dict(),
    "target_state_dict": target_agent.state_dict(),
    "optimizer": optimizer.state_dict(),
    "episode": episode_done,
}
torch.save(state, os.path.join("models", game_name, f"{model_name}.pt"))
