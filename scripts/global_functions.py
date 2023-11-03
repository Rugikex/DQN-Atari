from collections import deque
import os
import random
import sys
import time

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sys.path.append(os.path.join(os.getcwd()))

from classes.dqn import DeepQNetwork
from classes.env_wrapper import AtariWrapper
from classes.policy import EpsilonGreedyPolicy
from classes.stacked_frames import StackedFrames
import parameters

logs_path = os.path.join('logs')
if not os.path.exists(logs_path):
    os.makedirs(logs_path)

device = parameters.device
gamma = 0.99
minibatch_size = 32

extension = '.pt'


def get_model_path(game_name: str, model_name: str) -> str:
    model_path = f'{model_name}{extension}'
    if not os.path.exists(os.path.join('models', game_name, f'{model_path}')):
        raise Exception('No model found')

    return model_path


def update_q_network(agent: DeepQNetwork, target_agent: DeepQNetwork, minibatch: dict, optimizer: torch.optim, gamma: float) -> float:
    # Unpack the minibatch
    states, actions, rewards, next_states, dones = zip(*minibatch)

    # Convert data to PyTorch tensors
    states = torch.tensor(np.array(states), dtype=torch.float32).to(parameters.device)
    actions = torch.tensor(actions).to(parameters.device)
    rewards = torch.tensor(rewards).to(parameters.device)
    next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(parameters.device)
    not_dones = torch.logical_not(torch.tensor(dones).to(parameters.device))

    # Compute Q-values for the current state
    q_values = agent(states)
    q_values = q_values.gather(1, actions.view(-1, 1))

    print(q_values.shape)
    print(actions.shape)
    print(actions.view(-1, 1).shape)
    print(actions.unsqueeze(1).shape)

    # Compute the target Q-values using the target network
    next_q_values = target_agent(next_states)
    max_next_q_values, _ = next_q_values.max(1)
    target_q_values = rewards + not_dones * gamma * max_next_q_values

    # Compute loss
    loss = F.huber_loss(q_values, target_q_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def train_model(
        agent: DeepQNetwork,
        target_agent: DeepQNetwork,
        env: gym.Env,
        replay_memory: deque,
        optimizer: torch.optim,
        episodes_already_done: int=0,
        steps_already_done: int=0
) -> int:
    # TODO: better logging -> track with retrain script
    writter = SummaryWriter(log_dir=os.path.join('logs', time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())))
    env = AtariWrapper(env)

    epsilon = EpsilonGreedyPolicy(1.0, steps=steps_already_done)
    stacked_frames = StackedFrames(4)
    update_target_network = parameters.update_target_network

    state, _ = env.reset()
    stacked_frames.reset(state)
    memory_state = stacked_frames.get_frames()

    episode = episodes_already_done

    max_seconds = parameters.seconds_per_training
    progress_bar = tqdm(total=max_seconds, desc="Training", unit="s")

    step = 0
    start_time = time.time()
    time_spent = 0
    while time_spent < max_seconds:
        state, _ = env.reset()
        stacked_frames.reset(state)
        memory_state = stacked_frames.get_frames()

        episode_reward: np.float64 = 0.0
        episode_step = 0
        done = False
        epsilon_value: float
        reward_last_100_episodes = deque(maxlen=100)

        while episode_step < parameters.steps_per_episode:
            action: np.int64
            epsilon_value = epsilon.get_epsilon()
            if random.uniform(0, 1) < epsilon_value:
                # Choose a random action
                action = env.action_space.sample()
            else:
                # Choose the action with the highest Q-value
                observation = torch.tensor(stacked_frames.get_frames(), dtype=torch.float32).unsqueeze(0).to(device)
                with torch.no_grad():
                    q_values = agent(observation)
                action = torch.argmax(q_values).item()

            state, reward, done, _, info = env.step(action)
            previous_state = info['previous_state']

            stacked_frames.append(state, previous_state)
            real_reward = np.sign(reward).astype(np.int8)

            # Store the transition in the replay memory
            replay_memory.append((memory_state, action, real_reward, stacked_frames.get_frames(), done))
            memory_state = stacked_frames.get_frames()

            if step % 4 == 0 and len(replay_memory) == parameters.start_update:
                # Sample a minibatch from the replay memory
                minibatch = random.sample(replay_memory, minibatch_size)
                loss = update_q_network(agent, target_agent, minibatch, optimizer, gamma)
                writter.add_scalar('Loss', loss, step)

            # Update the target network every C steps
            if step % update_target_network == 0 and len(replay_memory) == parameters.start_update:
                target_agent.load_state_dict(agent.state_dict())

            episode_reward += real_reward
            episode_step += 1
            step += 1

            if done:
                break

        reward_last_100_episodes.append(episode_reward)
        writter.add_scalar('Reward last 100 episodes', np.mean(reward_last_100_episodes), episode)
        writter.add_scalar('Episode reward', episode_reward, episode)
        writter.add_scalar('Episode length', episode_step, episode)
        writter.add_scalar('Epsilon at the end of the episode', epsilon_value, episode)
        episode += 1
        time_spent = int(time.time() - start_time)
        progress_bar.update(time_spent - progress_bar.n)

    writter.flush()
    writter.close()

    env.close()

    return episode
