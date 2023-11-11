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
from classes.replay_memory import ReplayMemory
from parameters import (
    DEVICE,
    START_UPDATE,
    REPLAY_MEMORY_MAXLEN,
    UPDATE_TARGET_NETWORK,
    STEPS_PER_EPISODE,
)

GAMMA = 0.99
MINIBATCH_SIZE = 32
SECOND_PER_HOUR = 3600


# Create the logs directory if it does not exist
logs_path = os.path.join("logs")
if not os.path.exists(logs_path):
    os.makedirs(logs_path)


def fill_replay_memory(
    agent: DeepQNetwork,
    env: gym.Env,
    replay_memory: ReplayMemory,
    epsilon: EpsilonGreedyPolicy,
) -> None:
    """
    Fill the replay memory with random actions
    Use for retraining

    Parameters
    ----------
    agent: DeepQNetwork
        Agent used to get the action
    env: gym.Env
        Environment to interact with the agent
    replay_memory: ReplayMemory
        Replay memory to fill
    epsilon: EpsilonGreedyPolicy
        Epsilon greedy policy
    """
    state, _ = env.reset()
    memory_state = state

    for _ in tqdm(range(START_UPDATE), desc="Filling replay memory"):
        action = get_action(env, agent, epsilon, state)
        state, reward, done, _, _ = env.step(action)

        # Store the transition in the replay memory
        replay_memory.append((memory_state, action, reward, state, done))
        memory_state = state

        if done:
            state, _ = env.reset()
            memory_state = state


def get_action(
    env: gym.Env, agent: DeepQNetwork, epsilon: EpsilonGreedyPolicy, state: np.ndarray
) -> np.int64:
    """
    Get the action to take

    Parameters
    ----------
    env: gym.Env
        Environment to interact with the agent
    agent: DeepQNetwork
        Agent used to get the action
    epsilon: EpsilonGreedyPolicy
        Epsilon greedy policy
    state: np.ndarray
        Current state

    Returns
    -------
    np.int64
        Action to take
    """
    action: np.int64
    if random.uniform(0, 1) < epsilon.get_epsilon():
        # Exploration: choose a random action
        action = env.action_space.sample()
    else:
        # Exploitation: choose the action with the highest Q-value
        observation = (
            torch.as_tensor(state, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        )
        agent.eval()
        with torch.no_grad():
            q_values = agent(observation)
        action = torch.argmax(q_values).item()

    return action


def get_model_path(game_name: str, model_name: str) -> str:
    """
    Get the path of the model for retraining or playing

    Parameters
    ----------
    game_name: str
        Name of the game
    model_name: str
        Name of the model

    Returns
    -------
    str
        Path of the model
    """
    model_path = f"{model_name}.pt"
    if not os.path.exists(os.path.join("models", game_name, f"{model_path}")):
        raise Exception("No model found")

    return model_path


def update_q_network(
    agent: DeepQNetwork,
    target_agent: DeepQNetwork,
    minibatch: dict,
    optimizer: torch.optim,
) -> float:
    """
    Update the Q-network using the minibatch

    Parameters
    ----------
    agent: DeepQNetwork
        Agent used to get the Q-values for the current state
    target_agent: DeepQNetwork
        Target agent used to get the maximum Q-value for the next state
    minibatch: dict
        Minibatch of transitions
    optimizer: torch.optim
        Optimizer used to update the Q-network

    Returns
    -------
    float
        Loss value
    """
    # Unpack the minibatch
    states, actions, rewards, next_states, dones = zip(*minibatch)

    # Convert data to PyTorch tensors
    states = torch.as_tensor(np.array(states), dtype=torch.float32).to(DEVICE)
    actions = torch.as_tensor(actions).to(DEVICE)
    rewards = torch.as_tensor(rewards).to(DEVICE)
    next_states = torch.as_tensor(np.array(next_states), dtype=torch.float32).to(DEVICE)
    not_dones = torch.logical_not(torch.as_tensor(dones).to(DEVICE))

    # Compute Q-values for the current state
    agent.train()
    q_values = agent(states)
    q_values = q_values.gather(1, actions.unsqueeze(1))

    # Compute the target Q-values using the target network
    next_q_values = target_agent(next_states)
    max_next_q_values, _ = next_q_values.max(1)
    target_q_values = rewards + not_dones * GAMMA * max_next_q_values

    # Clip error to be between -1 and 1
    error = (q_values - target_q_values.unsqueeze(1)).clamp(-1, 1)

    # Compute loss
    loss = torch.mean(torch.square(error))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def train_model(
    agent: DeepQNetwork,
    target_agent: DeepQNetwork,
    env: gym.Env,
    optimizer: torch.optim,
    game_name: str,
    model_name: str,
    hours_to_train: int,
    episodes_already_done: int = 0,
    steps_already_done: int = 0,
    hours_already_done: int = 0,
) -> tuple:
    """
    Train the model with the implementation described in the article

    Parameters
    ----------
    agent: DeepQNetwork
        Agent used to get the Q-values for the current state
    target_agent: DeepQNetwork
        Target agent used to get the maximum Q-value for the next state
    env: gym.Env
        Environment to interact with the agent
    optimizer: torch.optim
        Optimizer used to update the Q-network
    game_name: str
        Name of the game
    model_name: str
        Name of the model
    hours_to_train: int
        Number of hours to train
    episodes_already_done: int
        Number of episodes already done
    steps_already_done: int
        Number of steps already done
    hours_already_done: int
        Number of hours already done

    Returns
    -------
    tuple
        Number of episodes, number of steps and number of hours done
    """
    writer = SummaryWriter(
        log_dir=os.path.join(
            "logs",
            f"{model_name}_{time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())}",
        )
    )
    env = AtariWrapper(env)

    epsilon = EpsilonGreedyPolicy(1.0, steps=steps_already_done)
    replay_memory = ReplayMemory(REPLAY_MEMORY_MAXLEN)
    if steps_already_done > 0:
        fill_replay_memory(agent, env, replay_memory, epsilon)

    episodes = episodes_already_done
    steps = steps_already_done
    counter_hours = hours_already_done

    start_time = time.time()
    time_spent = 0
    time_spent_to_save = steps_already_done % 3600
    better_reward = 0.0
    reward_last_100_episodes = deque(maxlen=100)
    length_last_100_episodes = deque(maxlen=100)

    max_seconds = SECOND_PER_HOUR * hours_to_train
    progress_bar = tqdm(total=max_seconds, desc="Training", unit="s")
    while time_spent < max_seconds:
        state, _ = env.reset()
        memory_state = state

        episode_reward: np.float64 = 0.0
        episode_step = 0
        done = False

        while episode_step < STEPS_PER_EPISODE:
            action = get_action(env, agent, epsilon, state)
            state, reward, done, _, _ = env.step(action)

            # Store the transition in the replay memory
            replay_memory.append((memory_state, action, reward, state, done))
            memory_state = state

            if steps % 4 == 0 and len(replay_memory) >= START_UPDATE:
                # Sample a minibatch from the replay memory
                minibatch = replay_memory.sample(MINIBATCH_SIZE)
                loss = update_q_network(agent, target_agent, minibatch, optimizer)
                writer.add_scalar("Loss", loss, steps)

            # Update the target network every C steps
            if (
                steps % UPDATE_TARGET_NETWORK == 0
                and len(replay_memory) >= START_UPDATE
            ):
                target_agent.load_state_dict(agent.state_dict())

            episode_reward += reward
            episode_step += 1
            steps += 1

            if done:
                break

        reward_last_100_episodes.append(episode_reward)
        length_last_100_episodes.append(episode_step)

        writer.add_scalar(
            "Epsilon at the end of the episode", epsilon.get_current_epsilon(), episodes
        )
        writer.add_scalar("Rewards/Episode", episode_reward, episodes)
        writer.add_scalar(
            "Rewards/Last_100_episodes", np.mean(reward_last_100_episodes), episodes
        )
        writer.add_scalar("Lengths/Episode", episode_step, episodes)
        writer.add_scalar(
            "Lengths/Last_100_episodes", np.mean(length_last_100_episodes), episodes
        )

        episodes += 1
        time_spent = int(time.time() - start_time)
        increment = time_spent - progress_bar.n
        progress_bar.update(increment)
        time_spent_to_save += increment

        # Save the model every hour
        if time_spent_to_save >= 3600:
            counter_hours += 1
            time_spent_to_save = time_spent_to_save % 3600
            torch.save(
                {
                    "state_dict": agent.state_dict(),
                    "target_state_dict": target_agent.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "episodes": episodes,
                    "steps": steps,
                    "hours": counter_hours,
                },
                os.path.join("models", game_name, f"{model_name}_{counter_hours}.pt"),
            )

        # Save the best model with the best mean reward over the last 100 episodes
        if counter_hours > 0 and np.mean(reward_last_100_episodes) > better_reward:
            better_reward = np.mean(reward_last_100_episodes)
            torch.save(
                {
                    "state_dict": agent.state_dict(),
                    "target_state_dict": target_agent.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "episodes": episodes,
                    "steps": steps,
                    "hours": counter_hours,
                },
                os.path.join("models", game_name, f"{model_name}_best.pt"),
            )

    progress_bar.close()

    writer.flush()
    writer.close()

    env.close()

    return episodes, steps, counter_hours
