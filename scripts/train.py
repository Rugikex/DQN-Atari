from collections import deque
import os
import pickle
import sys

import gymnasium as gym
import torch
import torch.optim as optim

sys.path.append(os.path.join(os.getcwd()))

from classes.dqn import DeepQNetwork
from classes.policy import EpsilonGreedyPolicy
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
    obs_type='rgb',
    full_action_space=False,
    render_mode='rgb_array'
)

n_actions = env.action_space.n

agent = DeepQNetwork(n_actions).to(device)
target_agent = DeepQNetwork(n_actions).to(device)
target_agent.load_state_dict(agent.state_dict())

replay_memory = deque(maxlen=parameters.N)
M = parameters.M * int(sys.argv[4])
C = 1

minibatch_size = 32
gamma = 0.99
epsilon = EpsilonGreedyPolicy(1.0)
optimizer = optim.RMSprop(
    agent.parameters(),
    lr=0.000_25,
    momentum=0.95,
)
    
print("=======")
print(f'Training on {game_name} for episode {M}')
print("=======")

train_model(
    agent,
    target_agent,
    env,
    replay_memory,
    epsilon,
    optimizer,
    M,
    C
)

from classes.stacked_frames import StackedFrames

state, _ = env.reset()
stacked_frames = StackedFrames(4)
stacked_frames.reset(state)
q_values = agent(torch.tensor(stacked_frames.get_frames(), dtype=torch.float32).unsqueeze(0).to(device))
print(q_values.detach().cpu().numpy())


# Save trained model and replay memory
if not os.path.exists(os.path.join('models', game_name)):
    os.makedirs(os.path.join('models', game_name))

# Save model TODO
torch.save(agent.state_dict(), os.path.join('models', game_name, f'episode_{M}.h5'))

# Recheck this
# with open(os.path.join('models', game_name, f'replay_memory_{M}.pkl'), 'wb') as file:
#     pickle.dump(replay_memory, file)
