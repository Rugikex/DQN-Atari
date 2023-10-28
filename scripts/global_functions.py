from collections import deque
import random
import re
import os
import sys

import gymnasium as gym
import numpy as np
import tensorflow as tf
from tqdm import tqdm

sys.path.append(os.path.join(os.getcwd()))

from classes.dqn import DeepQNetwork
from classes.policy import EpsilonGreedyPolicy
from classes.stacked_frames import StackedFrames
import parameters


gamma = 0.99
minibatch_size = 32

pattern = r"episode_(\d+).keras"

def get_model_path(game_name: str, episode: str) -> tuple[str, str]:
    model_path = None
    max_number: int
    try:
        episode = int(episode)
    except ValueError:
        episode = None

    if episode is not None:
        model_path = f'episode_{episode}.keras'
        if not os.path.exists(os.path.join('models', game_name, f'{model_path}')):
            raise Exception('No model found')
        max_number = episode
        
    else:
        max_number = 0
        for filename in os.listdir(os.path.join('models', game_name)):
            match = re.match(pattern, filename)
            if match:
                # Extract the number of episode from the filename
                number = int(match.group(1))

                # Check if this number is greater than the current max_number
                if number > max_number:
                    max_number = number
                    model_path = filename

    if not model_path:
        raise Exception('No model found')
    
    # Check if the replay memory associated with the model exists
    replay_memory_path = f'replay_memory_{max_number}.pkl'
    if not os.path.exists(os.path.join('models', game_name, replay_memory_path)):
        raise Exception('No replay memory found')

    return model_path, replay_memory_path

def train_model(
        agent: DeepQNetwork,
        target_agent: DeepQNetwork,
        env: gym.Env,
        replay_memory: deque,
        epsilon: EpsilonGreedyPolicy,
        optimizer: tf.keras.optimizers.Optimizer,
        M: int,
        C: int,
) -> DeepQNetwork:
    stacked_frames = StackedFrames(4)
    max_reward = -np.inf
    C_max = parameters.C_max
    T = parameters.T

    for _ in tqdm(range(1, M + 1), desc='Episodes'):
        state, info = env.reset()
        stacked_frames.reset(state)
        memory_state = stacked_frames.get_frames()
        lives = info['lives']

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
            if info['lives'] != lives:
                lives = info['lives']
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
                    target_clip = tf.clip_by_value(target, -1, 1)
                    q_values = agent(state_batch)
                    # TODO: L1 then L2 loss (prof's tip)
                    loss = tf.reduce_mean(tf.square(target_clip - q_values[action_batch]))
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
                lives = info['lives']
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

    return agent
