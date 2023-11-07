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
import parameters

logs_path = os.path.join("logs")
if not os.path.exists(logs_path):
    os.makedirs(logs_path)

device = parameters.device
gamma = 0.99
minibatch_size = 32


def fill_replay_memory(
    agent: DeepQNetwork,
    env: gym.Env,
    replay_memory: deque,
    epsilon: EpsilonGreedyPolicy,
    steps_already_done: int,
) -> None:
    state, _ = env.reset()
    memory_state = state
    steps_to_fill = (
        parameters.replay_memory_maxlen
        if steps_already_done > parameters.replay_memory_maxlen
        else steps_already_done
    )

    for _ in tqdm(range(steps_to_fill), desc="Filling replay memory"):
        action: np.int64
        epsilon_value = epsilon.get_epsilon()
        if random.uniform(0, 1) < epsilon_value:
            # Choose a random action
            action = env.action_space.sample()
        else:
            # Choose the action with the highest Q-value
            observation = (
                torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            )
            with torch.no_grad():
                q_values = agent(observation)
            action = torch.argmax(q_values).item()

        state, reward, done, _, _ = env.step(action)

        # Store the transition in the replay memory
        replay_memory.append((memory_state, action, reward, state, done))
        memory_state = state

        if done:
            state, _ = env.reset()
            memory_state = state


def get_model_path(game_name: str, model_name: str) -> str:
    model_path = f"{model_name}.pt"
    if not os.path.exists(os.path.join("models", game_name, f"{model_path}")):
        raise Exception("No model found")

    return model_path


def update_q_network(
    agent: DeepQNetwork,
    target_agent: DeepQNetwork,
    minibatch: dict,
    optimizer: torch.optim,
    gamma: float,
) -> float:
    # Unpack the minibatch
    states, actions, rewards, next_states, dones = zip(*minibatch)

    # Convert data to PyTorch tensors
    states = torch.tensor(np.array(states), dtype=torch.float32).to(parameters.device)
    actions = torch.tensor(actions).to(parameters.device)
    rewards = torch.tensor(rewards).to(parameters.device)
    next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(
        parameters.device
    )
    not_dones = torch.logical_not(torch.tensor(dones).to(parameters.device))

    # Compute Q-values for the current state
    q_values = agent(states)
    q_values = q_values.gather(1, actions.unsqueeze(1))

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
    optimizer: torch.optim,
    game_name: str,
    model_name: str,
    episodes_already_done: int = 0,
    steps_already_done: int = 0,
    hours_already_done: int = 0,
) -> tuple:
    # TODO: better logging -> track with retrain script
    writter = SummaryWriter(
        log_dir=os.path.join(
            "logs", time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        )
    )
    env = AtariWrapper(env)

    epsilon = EpsilonGreedyPolicy(1.0, steps=steps_already_done)
    replay_memory = deque(maxlen=parameters.replay_memory_maxlen)
    if steps_already_done != 0:
        fill_replay_memory(agent, env, replay_memory, epsilon, steps_already_done)

    update_target_network = parameters.update_target_network

    episodes = episodes_already_done
    steps = steps_already_done
    counter_hours = hours_already_done

    start_time = time.time()
    time_spent = 0
    time_spent_to_save = steps_already_done % 3600
    better_reward = 0.0
    reward_last_100_episodes = deque(maxlen=100)

    max_seconds = parameters.seconds_per_training
    progress_bar = tqdm(total=max_seconds, desc="Training", unit="s")
    while time_spent < max_seconds:
        state, _ = env.reset()
        memory_state = state

        episode_reward: np.float64 = 0.0
        episode_step = 0
        done = False
        epsilon_value: float

        while episode_step < parameters.steps_per_episode:
            action: np.int64
            epsilon_value = epsilon.get_epsilon()
            if random.uniform(0, 1) < epsilon_value:
                # Choose a random action
                action = env.action_space.sample()
            else:
                # Choose the action with the highest Q-value
                observation = (
                    torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                )
                with torch.no_grad():
                    q_values = agent(observation)
                action = torch.argmax(q_values).item()

            state, reward, done, _, _ = env.step(action)

            # Store the transition in the replay memory
            replay_memory.append((memory_state, action, reward, state, done))
            memory_state = state

            if steps % 4 == 0 and len(replay_memory) >= parameters.start_update:
                # Sample a minibatch from the replay memory
                minibatch = random.sample(replay_memory, minibatch_size)
                loss = update_q_network(
                    agent, target_agent, minibatch, optimizer, gamma
                )
                writter.add_scalar("Loss", loss, steps)

            # Update the target network every C steps
            if (
                steps % update_target_network == 0
                and len(replay_memory) >= parameters.start_update
            ):
                target_agent.load_state_dict(agent.state_dict())

            episode_reward += reward
            episode_step += 1
            steps += 1

            if done:
                break

        reward_last_100_episodes.append(episode_reward)

        writter.add_scalar(
            "Reward last 100 episodes", np.mean(reward_last_100_episodes), episodes
        )
        writter.add_scalar("Episode reward", episode_reward, episodes)
        writter.add_scalar("Episode length", episode_step, episodes)
        writter.add_scalar(
            "Epsilon at the end of the episodes", epsilon_value, episodes
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

    writter.flush()
    writter.close()

    env.close()

    return episodes, steps, counter_hours
