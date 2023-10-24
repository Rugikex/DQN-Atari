from collections import deque
import random

import gymnasium as gym
import numpy as np
import tensorflow as tf

from dql import DeepQLearning
from stacked_frames import StackedFrames


env = gym.make(
    'Asterix-v4',
    mode=0,
    difficulty=0,
    obs_type='rgb',
    frameskip=5,
    # repeat_action_probability=None,
    full_action_space=True,
    render_mode='rgb_array'
)

n_actions = env.action_space.n

agent = DeepQLearning(n_actions)
target_agent = DeepQLearning(n_actions)
target_agent.set_weights(agent.get_weights())

stacked_frames = StackedFrames(4)

# TODO: Change hyperparameters
replay_memory = deque(maxlen=300)
M = 10
T = 100
epsilon = 0.1
C = 1
C_max = 50

minibatch_size = 32
gamma = 0.99
optimizer = tf.optimizers.Adam(learning_rate=0.00025)


for episode in range(1, M + 1):
    print(f'Episode {episode}')
    state, _ = env.reset()
    stacked_frames.reset(state)
    memory_state = stacked_frames.get_frames()

    for t in range(1, T + 1):
        # TODO: Decay epsilon
        # Choose an action using epsilon-greedy policy
        action: int
        if random.uniform(0, 1) < epsilon:
            # Choose a random action
            action = env.action_space.sample()
        else:
            # Choose the action with the highest Q-value
            q_values = agent(stacked_frames.get_frames())
            action = np.argmax(q_values[0])

        next_state, reward, done, _, _ = env.step(action)
        stacked_frames.append(next_state)
        real_reward = np.sign(reward)

        if t % 10 == 0:
            print(f'Step {t}, Reward: {real_reward}')

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
                # TODO: Have to np.sign() the reward?
                target = reward_batch + gamma * np.max(target_agent(next_state_batch))
            
            # TODO: Recheck this
            with tf.GradientTape() as tape:
                q_values = agent(state_batch)
                # TODO: L1 then L2 loss (prof's tip)
                loss = tf.reduce_mean(tf.square(target - q_values[0][action_batch]))
            gradients = tape.gradient(loss, agent.trainable_variables)
            # TODO: Normalize gradients?
            optimizer.apply_gradients(zip(gradients, agent.trainable_variables))

        # Update the target network every C steps
        if C == C_max:
            target_agent.set_weights(agent.get_weights())
            C = 0

        # TODO: Maybe just reset the environment instead of breaking
        if done:
            break

        state = next_state
        C = min(C_max, C + 1)

# TODO: Save the model
