from collections import deque
import os
import pickle
import random
import sys

import gymnasium as gym
import numpy as np
import tensorflow as tf
from tqdm import tqdm

sys.path.append(os.path.join(os.getcwd()))

from classes.dql import DeepQLearning
from classes.stacked_frames import StackedFrames
from classes.policy import EpsilonGreedyPolicy
import parameters


game_name = sys.argv[1]

env = gym.make(
    game_name,
    mode=int(sys.argv[2]),
    difficulty=int(sys.argv[3]),
    obs_type='rgb',
    frameskip=5,
    full_action_space=True,
    render_mode='rgb_array'
)

n_actions = env.action_space.n

agent = DeepQLearning(n_actions)
target_agent = DeepQLearning(n_actions)
target_agent.set_weights(agent.get_weights())

stacked_frames = StackedFrames(4)

replay_memory = deque(maxlen=parameters.N)
M = parameters.M
T = parameters.T
C = 1
C_max = parameters.C_max

minibatch_size = 32
gamma = 0.99
epsilon = EpsilonGreedyPolicy(1.0)
optimizer = tf.keras.optimizers.experimental.RMSprop(
    learning_rate=0.0025,
    momentum=0.95,
)

max_reward = -np.inf


for episode in tqdm(range(1, M + 1), desc='Episodes'):
    state, info = env.reset()
    stacked_frames.reset(state)
    memory_state = stacked_frames.get_frames()
    lifes = info['lives']

    total_reward: np.float64 = 0.0

    step_bar = tqdm(range(1, T + 1), desc=f'Total reward: {total_reward} - Steps', leave=False)
    for t in step_bar:
        # Choose an action using epsilon-greedy policy
        action: np.int64
        if random.uniform(0, 1) < epsilon.get_epsilon():
            # Choose a random action
            action = env.action_space.sample()
        else:
            # Choose the action with the highest Q-value
            q_values = agent(stacked_frames.get_frames())
            action = np.argmax(q_values)

        next_state, reward, done, _, info = env.step(action)
        stacked_frames.append(next_state)
        if info['lives'] != lifes:
            lifes = info['lives']
            reward = -1
        real_reward = np.sign(reward)

        total_reward += real_reward

        # Store the transition in the replay memory
        replay_memory.append((memory_state, action, real_reward, stacked_frames.get_frames(), done))
        memory_state = stacked_frames.get_frames()

        if len(replay_memory) < minibatch_size:
            continue

        # Sample a minibatch from the replay memory
        minibatch = random.sample(replay_memory, minibatch_size)

        # Perform Q-network update
        for state_batch, action_batch, reward_batch, next_state_batch, done_batch in minibatch:
            with tf.GradientTape() as tape:
                target = reward_batch + gamma * tf.reduce_max(target_agent(next_state_batch), axis=0) * (1 - done_batch)
                q_values = agent(state_batch)
                # TODO: L1 then L2 loss (prof's tip)
                loss = tf.reduce_mean(tf.square(target - q_values[action_batch]))
            gradients = tape.gradient(loss, agent.trainable_variables)
            optimizer.apply_gradients(zip(gradients, agent.trainable_variables))

        # Update the target network every C steps
        if C == C_max:
            target_agent.set_weights(agent.get_weights())
            C = 0

        if done:
            state, info = env.reset()
            stacked_frames.reset(state)
            memory_state = stacked_frames.get_frames()
            lifes = info['lives']
        else:
            state = next_state

        C = min(C_max, C + 1)

        step_bar.set_description(f'Total reward: {total_reward} - Steps')
        step_bar.update()

    step_bar.close()
    
    epsilon.update_epsilon()

    if total_reward > max_reward:
        max_reward = total_reward

env.close()

print(f'Max reward in single episode: {max_reward}')

# Save learned model and replay memory
if not os.path.exists(os.path.join('models', game_name)):
    os.makedirs(os.path.join('models', game_name))

agent.save_weights(os.path.join('models', game_name, f'episode_{M}'))
with open(os.path.join('models', game_name, f'replay_memory_{M}.pkl'), 'wb') as file:
    pickle.dump(replay_memory, file)
