from collections import deque
import os
import pickle
import sys

import gymnasium as gym
import tensorflow as tf

sys.path.append(os.path.join(os.getcwd()))

from classes.dqn import DeepQNetwork
from classes.policy import EpsilonGreedyPolicy
from global_functions import train_model
import parameters


game_name = sys.argv[1]

env = gym.make(
    game_name,
    mode=int(sys.argv[2]),
    difficulty=int(sys.argv[3]),
    obs_type='rgb',
    full_action_space=True,
    render_mode='rgb_array'
)

n_actions = env.action_space.n

agent = DeepQNetwork(n_actions)
target_agent = DeepQNetwork(n_actions)
target_agent.set_weights(agent.get_weights())

replay_memory = deque(maxlen=parameters.N)
M = parameters.M * int(sys.argv[4])
C = 1

minibatch_size = 32
gamma = 0.99
epsilon = EpsilonGreedyPolicy(1.0)
optimizer = tf.keras.optimizers.experimental.RMSprop(
    learning_rate=0.0025,
    momentum=0.95,
)

print("=======")
print(f'Training on {game_name} for episode {M}')

trained_agent = train_model(
    agent,
    target_agent,
    env,
    replay_memory,
    epsilon,
    optimizer,
    M,
    C
)

# Save trained model and replay memory
if not os.path.exists(os.path.join('models', game_name)):
    os.makedirs(os.path.join('models', game_name))

# agent.save_weights(os.path.join('models', game_name, f'episode_{M}'))
trained_agent.save(os.path.join('models', game_name, f'episode_{M}.keras'))
with open(os.path.join('models', game_name, f'replay_memory_{M}.pkl'), 'wb') as file:
    pickle.dump(replay_memory, file)
