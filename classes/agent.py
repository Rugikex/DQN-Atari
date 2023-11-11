from collections import deque
import os
import random
import re
import time

import gymnasium as gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from classes.dqn import DeepQNetwork
from classes.env_wrapper import AtariWrapper
from classes.policy import EpsilonGreedyPolicy
from classes.replay_memory import ReplayMemory

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPSILON_FINAL_STEP = 1_000_000
GAMMA = 0.99
MINIBATCH_SIZE = 32
REPLAY_MEMORY_MAXLEN = 600_000
SECOND_PER_HOUR = 3_600
START_UPDATE = 50_000
STEPS_PER_EPISODE = 2_000
UPDATE_TARGET_NETWORK = 10_000


class AtariAgent:
    """
    Agent for Atari games

    Parameters
    ----------
    game_name : str
        Name of the game
    env : gym.Env
        Environment
    play : bool, optional
        Play mode, by default False
    """
    def __init__(self, game_name: str, env: gym.Env, play: bool = False) -> None:
        self.game_name = game_name
        self.env = AtariWrapper(env, play=play)
        self.online_network = DeepQNetwork(env.action_space.n).to(DEVICE)
        self.target_network = DeepQNetwork(env.action_space.n).to(DEVICE)
        self.target_network.load_state_dict(self.online_network.state_dict())
        self.target_network.eval() # Target network is not trained
        self.optimizer = torch.optim.RMSprop(self.online_network.parameters(), lr=0.000_25, alpha=0.95, eps=0.01)
        if play:
            self.policy = EpsilonGreedyPolicy(0.05, 0.05, 0, 0)
        else:
            self.policy = EpsilonGreedyPolicy(1.0, 0.1, START_UPDATE, EPSILON_FINAL_STEP)
        self.replay_memory = ReplayMemory(REPLAY_MEMORY_MAXLEN)
        self.model_name = None
        self.episodes = 0
        self.steps = 0
        self.hours = 0

    def _fill_replay_memory(self) -> None:
        """
        Fill the replay memory with random actions
        Use for retraining
        """
        state, _ = self.env.reset()
        memory_state = state

        for _ in tqdm(range(START_UPDATE), desc="Filling replay memory"):
            action = self._get_action(state)
            state, reward, done, _, _ = self.env.step(action)

            # Store the transition in the replay memory
            self.replay_memory.append((memory_state, action, reward, state, done))
            memory_state = state

            if done:
                state, _ = self.env.reset()
                memory_state = state

    def _get_model_path(self) -> str:
        """
        Get the path of the model

        Returns
        -------
        str
            Path of the model
        """
        model_path = os.path.join("models", self.game_name, f"{self.model_name}.pt")
        if not os.path.exists(model_path):
            # If the model doesn't exist, maybe the user want to load the last model
            model_path = os.path.join("models", self.game_name, f"{self.model_name}_last.pt")
            if not os.path.exists(model_path):
                raise Exception("Model not found")
            self.model_name = f"{self.model_name}_last"
        
        return model_path
    
    def _get_action(self, state: np.ndarray) -> np.int64:
        """
        Get the action to take

        Parameters
        ----------
        state: np.ndarray
            Current state

        Returns
        -------
        np.int64
            Action to take
        """
        action: np.int64
        if random.uniform(0, 1) < self.policy.get_epsilon():
            # Exploration: choose a random action
            action = self.env.action_space.sample()
        else:
            # Exploitation: choose the action with the highest Q-value
            observation = (
                torch.as_tensor(state, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            )
            # Inference part
            self.online_network.eval()
            with torch.no_grad():
                q_values = self.online_network(observation)
            action = torch.argmax(q_values).item()

        return action
    
    def _save_model(self, suffix: str = None) -> None:
        """
        Save the model

        Parameters
        ----------
        suffix : str, optional
            Suffix of the model name, by default None
        """
        if not suffix:
            suffix = self.hours
        torch.save(
            {
                "state_dict": self.online_network.state_dict(),
                "target_state_dict": self.target_network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "episodes": self.episodes,
                "steps": self.steps,
                "hours": self.hours,
            },
            os.path.join("models", self.game_name, f"{self.model_name}_{suffix}.pt"),
        )
    
    def _update_q_network(
        self,
        minibatch: dict,
    ) -> float:
        """
        Update the Q-network using the minibatch

        Parameters
        ----------
        minibatch: dict
            Minibatch of transitions

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
        self.target_network.train()
        q_values = self.online_network(states)
        q_values = q_values.gather(1, actions.unsqueeze(1))

        # Compute the target Q-values using the target network
        next_q_values = self.target_network(next_states)
        max_next_q_values, _ = next_q_values.max(1)
        target_q_values = rewards + not_dones * GAMMA * max_next_q_values

        # Clip error to be between -1 and 1
        error = (q_values - target_q_values.unsqueeze(1)).clamp(-1, 1)

        # Compute loss
        loss = torch.mean(torch.square(error))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def load_model(self, model_name: str, play: bool = False) -> None:
        """
        Load a model

        Parameters
        ----------
        model_name : str
            Name of the model
        """
        self.model_name = model_name
        model_path = self._get_model_path()
        states = torch.load(model_path)
        self.online_network.load_state_dict(states["state_dict"])
        self.target_network.load_state_dict(states["target_state_dict"])
        if self.target_network.training:
            print("Target network is training")
        self.optimizer.load_state_dict(states["optimizer"])
        self.episodes = states["episodes"]
        self.steps = states["steps"]
        self.hours = states["hours"]
        if not play:
            self.policy.decaying_epsilon(self.steps)
            self._fill_replay_memory()

    def train(self, hours_to_train: int, model_name: str = None) -> None:
        """
        Train the agent

        Parameters
        ----------
        hours_to_train : int
            Number of hours to train the agent
        model_name : str, optional
            Name of the model, by default None
        """
        if not model_name:
            list_number_suffix = []
            prefix = "model_v"
            for file in os.listdir(os.path.join("models", self.game_name)):
                # We find model with prefix with regex
                match = re.match(f"^({prefix})([0-9]+)$", file)
                if match:
                    list_number_suffix.append(int(match.group(2)))

            if list_number_suffix:
                model_name = f"{prefix}{max(list_number_suffix) + 1}"
            else:
                model_name = f"{prefix}0"

        self.model_name = model_name

        writer = SummaryWriter(
            log_dir=os.path.join(
                "logs",
                model_name,
                time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()),
            )
        )

        print("=======")
        print(f"Training on {self.game_name} with model {self.model_name} for {hours_to_train} hours\nAlready done: {self.hours} hours, {self.episodes} episodes, {self.steps} steps")
        print("=======")

        time_spent = 0
        time_spent_to_save = self.steps % SECOND_PER_HOUR
        better_reward = 0.0
        reward_last_100_episodes = deque(maxlen=100)
        length_last_100_episodes = deque(maxlen=100)

        max_seconds = hours_to_train * SECOND_PER_HOUR
        progress_bar = tqdm(total=max_seconds, desc="Training", unit="s")
        start_time = time.time()
        while time_spent < max_seconds:
            state, _ = self.env.reset()
            memory_state = state

            episode_reward = 0.0
            episode_step = 0
            done = False

            while episode_step < STEPS_PER_EPISODE:
                action = self._get_action(memory_state)
                state, reward, done, _, _ = self.env.step(action)

                # Store the transition in the replay memory
                self.replay_memory.append((memory_state, action, reward, state, done))
                memory_state = state

                if self.steps % 4 == 0 and len(self.replay_memory) >= START_UPDATE:
                    # Sample a minibatch from the replay memory
                    minibatch = self.replay_memory.sample(MINIBATCH_SIZE)
                    loss = self._update_q_network(minibatch)
                    writer.add_scalar("Loss", loss, self.steps)

                # Update the target network every C steps
                if (
                    self.steps % UPDATE_TARGET_NETWORK == 0
                    and len(self.replay_memory) >= START_UPDATE
                ):
                    self.target_network.load_state_dict(self.online_network.state_dict())

                episode_reward += reward
                episode_step += 1
                self.steps += 1

                if done:
                    break

            reward_last_100_episodes.append(episode_reward)
            length_last_100_episodes.append(episode_step)

            writer.add_scalar(
                "Epsilon at the end of the episode", self.policy.get_current_epsilon(), self.episodes
            )
            writer.add_scalar("Rewards/Episode", episode_reward, self.episodes)
            writer.add_scalar(
                "Rewards/Last_100_episodes", np.mean(reward_last_100_episodes), self.episodes
            )
            writer.add_scalar("Lengths/Episode", episode_step, self.episodes)
            writer.add_scalar(
                "Lengths/Last_100_episodes", np.mean(length_last_100_episodes), self.episodes
            )

            self.episodes += 1
            time_spent = int(time.time() - start_time)
            increment = time_spent - progress_bar.n
            progress_bar.update(increment)
            time_spent_to_save += increment

            # Save the model every hour
            if time_spent_to_save >= SECOND_PER_HOUR:
                self.hours += 1
                self._save_model()

            # Save the best model with the best mean reward over the last 100 episodes
            if self.hours > 0 and np.mean(reward_last_100_episodes) > better_reward:
                better_reward = np.mean(reward_last_100_episodes)
                self._save_model("best")

        progress_bar.close()

        writer.flush()
        writer.close()

        self.env.close()

        self._save_model("last")

    def play(self) -> None:
        """
        Play with the agent
        """
        total_steps = 0
        total_reward = 0.0

        print("=======")
        print(f"Playing {self.game_name} with model {self.model_name}, trained for {self.episodes} episodes, {self.steps} steps, {self.hours} hours")
        print("=======")

        state, _ = self.env.reset()
        while True:
            action = self._get_action(state)

            state, reward, done, _, _ = self.env.step(action)

            total_reward += reward
            total_steps += 1

            if done:
                break

        self.env.close()

        print(f"Total reward: {total_reward} in {total_steps} steps")