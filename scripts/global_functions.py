from collections import deque
import random
import re
import os
import sys

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

pattern = r"episode_(\d+)\.h5"


def get_model_path(game_name: str, episode: str) -> tuple[str, str]:
    model_path = None
    max_number: int
    try:
        episode = int(episode)
    except ValueError:
        episode = None

    if episode is not None:
        model_path = f'episode_{episode}.h5'
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
    # if not os.path.exists(os.path.join('models', game_name, replay_memory_path)):
    #     raise Exception('No replay memory found')

    return model_path, replay_memory_path


def update_q_network(agent: DeepQNetwork, target_agent: DeepQNetwork, minibatch: dict, optimizer: torch.optim, gamma: float) -> float:
    # Unpack the minibatch
    states, actions, rewards, next_states, dones = zip(*minibatch)

    # Convert data to PyTorch tensors
    states = torch.tensor(np.array(states, dtype=np.float32), dtype=torch.float32).to(parameters.device)
    next_states = torch.tensor(np.array(next_states, dtype=np.float32), dtype=torch.float32).to(parameters.device)
    actions = torch.tensor(actions).to(parameters.device)
    rewards = torch.tensor(rewards, dtype=torch.float32).to(parameters.device)
    dones = torch.tensor(dones, dtype=torch.float32).to(parameters.device)

    # Compute Q-values for the current state
    q_values = agent(states)
    q_values = q_values.gather(1, actions.view(-1, 1))

    # Compute the target Q-values using the target network
    with torch.no_grad():
        next_q_values = target_agent(next_states)
        max_next_q_values, _ = next_q_values.max(1)
        target_q_values = rewards + (1 - dones) * gamma * max_next_q_values

    # # Clip the error
    # error = q_values - target_q_values.view(-1, 1)
    # clipped_error = torch.clamp(error, -1, 1)

    # loss = torch.mean(torch.abs(clipped_error))

    # Compute the loss (e.g., mean squared error)
    loss = F.mse_loss(q_values, target_q_values.view(-1, 1))

    # Perform backpropagation and update the Q-network
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def train_model(
        agent: DeepQNetwork,
        target_agent: DeepQNetwork,
        env: gym.Env,
        replay_memory: deque,
        epsilon: EpsilonGreedyPolicy,
        optimizer: torch.optim,
        M: int,
        C: int,
) -> None:
    writter = SummaryWriter(log_dir=os.path.join('logs'))
    env = AtariWrapper(env)

    stacked_frames = StackedFrames(4)
    C_max = parameters.C_max

    state, _ = env.reset()
    stacked_frames.reset(state)
    memory_state = stacked_frames.get_frames()

    episode = 0
    for step in tqdm(range(1, parameters.frame_per_trainings + 1), desc='Steps'):
        state, _ = env.reset()
        stacked_frames.reset(state)
        memory_state = stacked_frames.get_frames()

        episode_reward: np.float64 = 0.0
        episode_step = 0
        done = False

        while not done:
            action: np.int64
            if random.uniform(0, 1) < epsilon.get_epsilon():
                # Choose a random action
                action = env.action_space.sample()
            else:
                # Choose the action with the highest Q-value
                observation = torch.tensor(stacked_frames.get_frames(), dtype=torch.float32).unsqueeze(0).to(parameters.device)
                q_values = agent(observation)
                action = torch.argmax(q_values).item()

            state, reward, done, _, info = env.step(action)
            previous_state = info['previous_state']

            stacked_frames.append(state, previous_state)
            real_reward = np.sign(reward)

            # Store the transition in the replay memory
            replay_memory.append((memory_state, action, real_reward, stacked_frames.get_frames(), done))
            memory_state = stacked_frames.get_frames()

            if step % 4 == 0 and len(replay_memory) >= parameters.N:
                # Sample a minibatch from the replay memory
                minibatch = random.sample(replay_memory, minibatch_size)
                update_q_network(agent, target_agent, minibatch, optimizer, gamma)

            # Update the target network every C steps
            if step % C_max == 0 and len(replay_memory) >= parameters.N:
                target_agent.load_state_dict(agent.state_dict())

            episode_reward += real_reward
            episode_step += 1
            step += 1

            if done:
                break

        writter.add_scalar('Episode reward', episode_reward, episode)
        writter.add_scalar('Episode length', episode_step, episode)
        episode += 1

    writter.flush()
    writter.close()

    env.close()
