from collections import deque
import os
import random

import gymnasium as gym
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from dql import DeepQLearning
from stacked_frames import StackedFrames
from policy import EpsilonGreedyPolicy


game_name = 'Assault'
version = 4
difficulty = 0

env = gym.make(
    f"{game_name}-v{version}",
    mode=0,
    difficulty=difficulty,
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

# TODO: Change hyperparameters
# replay_memory = deque(maxlen=3000)
# M = 50
# T = 250
# C = 1
# C_max = 300

replay_memory = deque(maxlen=200)
M = 20
T = 50
C = 1
C_max = 75

minibatch_size = 32
gamma = 0.99
epsilon = EpsilonGreedyPolicy(1.0)
optimizer = tf.keras.optimizers.experimental.RMSprop(
    learning_rate=0.00025,
    momentum=0.95,
)

for episode in tqdm(range(1, M + 1), desc='Episodes'):
    state, _ = env.reset()
    stacked_frames.reset(state)
    memory_state = stacked_frames.get_frames()

    total_reward = 0

    for t in tqdm(range(1, T + 1), desc='Steps', leave=False):
        # Choose an action using epsilon-greedy policy
        action: int
        if random.uniform(0, 1) < epsilon.get_epsilon():
            # Choose a random action
            action = env.action_space.sample()
        else:
            # Choose the action with the highest Q-value
            q_values = agent(stacked_frames.get_frames())
            action = np.argmax(q_values[0])

        next_state, reward, done, _, _ = env.step(action)
        stacked_frames.append(next_state)
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
            target = reward_batch
            if not done_batch:
                target = reward_batch + gamma * np.max(target_agent(next_state_batch))
                target = np.sign(target)
            
            # TODO: Recheck this
            with tf.GradientTape() as tape:
                q_values = agent(state_batch)
                # TODO: L1 then L2 loss (prof's tip)
                loss = tf.reduce_mean(tf.square(target - q_values[0][action_batch]))
            gradients = tape.gradient(loss, agent.trainable_variables)
            optimizer.apply_gradients(zip(gradients, agent.trainable_variables))

        # Update the target network every C steps
        if C == C_max:
            target_agent.set_weights(agent.get_weights())
            C = 0

        if done:
            state, _ = env.reset()
            stacked_frames.reset(state)
            memory_state = stacked_frames.get_frames()
        else:
            state = next_state

        C = min(C_max, C + 1)

    epsilon.update_epsilon()


agent.save_weights(os.path.join(os.getcwd(), 'models', f'{game_name}_v{version}'))
