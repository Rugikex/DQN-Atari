from collections import deque
import os
import pickle
import re
import sys

import gymnasium as gym
import tensorflow as tf
from tensorflow.keras.models import load_model

sys.path.append(os.path.join(os.getcwd()))

from classes.dqn import DeepQNetwork
from classes.policy import EpsilonGreedyPolicy
from global_functions import get_model_path, train_model
import parameters


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


model_path, replay_memory_path = get_model_path(game_name, sys.argv[5])

agent: DeepQNetwork = load_model(os.path.join('models', game_name, model_path))
target_agent: DeepQNetwork = load_model(os.path.join('models', game_name, model_path))

replay_memory: deque
with open(os.path.join('models', game_name, replay_memory_path), 'rb') as f:
    replay_memory = pickle.load(f)

M = parameters.M * int(sys.argv[4])
epoque_already_played = int(re.match(r"replay_memory_(\d+)\.pkl", replay_memory_path).group(1))

# Restoring C and epsilon to the value it had when the model was saved
C = (1 + epoque_already_played * parameters.T) % parameters.C_max
epsilon = EpsilonGreedyPolicy(1.0, steps=epoque_already_played * parameters.T)

minibatch_size = 32
gamma = 0.99
optimizer = tf.keras.optimizers.RMSprop(
    learning_rate=0.0025,
    momentum=0.95,
)

print("=======")
print(f'Retraining on {game_name} for episode {M} with {epoque_already_played} already played')
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

# Save learned model
if not os.path.exists(os.path.join('models', game_name)):
    os.makedirs(os.path.join('models', game_name))

agent.save(os.path.join('models', game_name, f'episode_{M + epoque_already_played}.keras'))
with open(os.path.join('models', game_name, f'replay_memory_{M + epoque_already_played}.pkl'), 'wb') as file:
    pickle.dump(replay_memory, file)
