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

    stacked_frames = StackedFrames(4)
    best_reward = -np.inf
    best_episode = 0
    C_max = parameters.C_max
    T = parameters.T
    skip_frames = 4

    state, info = env.reset()
    stacked_frames.reset(state)
    memory_state = stacked_frames.get_frames()
    lives = info['lives']
    previous_state = state

    # Start with 50_000 random moves
    for _ in tqdm(range(50_000), leave=False, desc='Random Moves to fill the replay memory'):
        action = env.action_space.sample()

        next_state = previous_state
        sum_reward = 0.0
        done: bool

        for skip in range(skip_frames):
            if skip <= skip_frames - 2:
                previous_state = next_state

            next_state, reward, done, _, info = env.step(action)
            if info['lives'] != lives:
                lives = info['lives']
                reward = -1.0
                done = True

            sum_reward += np.sign(reward)

            if done:
                break

        stacked_frames.append(next_state, previous_state)
        real_reward = np.sign(sum_reward)

        # Store the transition in the replay memory
        replay_memory.append((memory_state, action, real_reward, stacked_frames.get_frames(), done))
        memory_state = stacked_frames.get_frames()

        if done:
            next_state, info = env.reset()
            stacked_frames.reset(state)
            lives = info['lives']

        state = next_state

    for episode in tqdm(range(1, M + 1), desc='Episodes'):
        state, info = env.reset()
        stacked_frames.reset(state)
        memory_state = stacked_frames.get_frames()
        lives = info['lives']
        previous_state = state

        total_reward: np.float64 = 0.0
        
        step_bar = tqdm(range(1, T + 1), desc=f'Total reward: {total_reward} -- First Move  -- Steps', leave=False)
        for _ in step_bar:
            # Choose an action using epsilon-greedy policy
            action: np.int64
            move_type: str
            if random.uniform(0, 1) < epsilon.get_epsilon():
                # Choose a random action
                action = env.action_space.sample()
                move_type = 'Random Move'
            else:
                # Choose the action with the highest Q-value
                observation = torch.tensor(stacked_frames.get_frames(), dtype=torch.float32).unsqueeze(0).to(parameters.device)
                q_values = agent(observation)
                action = torch.argmax(q_values).item()
                move_type = 'Max Move   '


            next_state = previous_state
            sum_reward = 0.0
            done: bool

            for skip in range(skip_frames):
                if skip <= skip_frames - 2:
                    previous_state = next_state

                next_state, reward, done, _, info = env.step(action)
                if info['lives'] != lives:
                    lives = info['lives']
                    reward = -1.0
                    done = True

                sum_reward += np.sign(reward)

                if done:
                    break

            stacked_frames.append(next_state, previous_state)
            real_reward = np.sign(sum_reward)
            total_reward += real_reward

            # Store the transition in the replay memory
            replay_memory.append((memory_state, action, real_reward, stacked_frames.get_frames(), done))
            memory_state = stacked_frames.get_frames()

            if len(replay_memory) >= parameters.end_full_random:
                # Sample a minibatch from the replay memory
                minibatch = random.sample(replay_memory, minibatch_size)
                loss = update_q_network(agent, target_agent, minibatch, optimizer, gamma)
                writter.add_scalar('Loss', loss, episode)

            # Update the target network every C steps
            if C == C_max:
                target_agent.load_state_dict(agent.state_dict())
                C = 0

            if done:
                break

            state = next_state
            C = min(C_max, C + 1)

            step_bar.set_description(f'Total reward: {total_reward} -- {move_type} -- Steps')
            writter.add_scalar('Total reward', total_reward, episode)

        step_bar.close()

        if total_reward > best_reward:
            best_reward = total_reward
            best_episode = episode
        

    writter.flush()
    writter.close()

    # env.close()
    print(f'Best reward: {best_reward} in episode {best_episode}')
