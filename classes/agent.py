from collections import deque
import os
import random
import re
import time

import gymnasium as gym
from gymnasium.wrappers.record_video import RecordVideo
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from classes.dqn import DeepQNetwork
from classes.env_wrapper import AtariWrapper
from classes.policy import EpsilonGreedyPolicy
from classes.replay_memory import ReplayMemory

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPSILON_FIRST_STEP = 1_000_000
EPSILON_SECOND_STEP = 10_000_000
GAMMA = 0.99
MINIBATCH_SIZE = 32
REPLAY_MEMORY_MAXLEN = 1_000_000
SECOND_PER_HOUR = 3_600
START_UPDATE = 50_000
STEPS_PER_EPISODE = 2_000
UPDATE_TARGET_NETWORK = 10_000

print(f"Using {DEVICE} device")


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

    def __init__(
        self,
        game_name: str,
        env: gym.Env,
        play: bool = False,
    ) -> None:
        self.game_name = game_name
        self.env = AtariWrapper(env, play=play)
        self.online_network = DeepQNetwork(env.action_space.n).to(DEVICE)
        self.target_network = DeepQNetwork(env.action_space.n).to(DEVICE)
        self.target_network.load_state_dict(self.online_network.state_dict())
        self.target_network.eval()  # Target network is not trained
        self.optimizer = torch.optim.RMSprop(
            self.online_network.parameters(), lr=0.000_25, alpha=0.95, eps=0.01
        )
        if play:
            self.policy = EpsilonGreedyPolicy([0.05], [0])
        else:
            self.policy = EpsilonGreedyPolicy(
                [1.0, 0.1, 0.01],
                [START_UPDATE, EPSILON_FIRST_STEP, EPSILON_SECOND_STEP],
            )
        self.replay_memory = ReplayMemory(REPLAY_MEMORY_MAXLEN)
        self.model_name = None
        self.episodes = 0
        self.steps = 0
        self.hours = 0
        self.time_to_save = 0
        self.best_mean_reward = 0.0

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
            model_path = os.path.join(
                "models", self.game_name, f"{self.model_name}_last.pt"
            )
            if not os.path.exists(model_path):
                raise Exception("Model not found")

        # Get the name of the model without the suffix
        match = re.match(r"^(.*?)(_[0-9]+|_last|_best)?$", self.model_name)
        if match:
            self.model_name = match.group(1)

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
        # Create the folder if it doesn't exist
        if not os.path.exists(os.path.join("models", self.game_name)):
            os.makedirs(os.path.join("models", self.game_name))

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
                "time_to_save": self.time_to_save,
                "best_mean_reward": self.best_mean_reward,
            },
            os.path.join("models", self.game_name, f"{self.model_name}_{suffix}.pt"),
            pickle_protocol=4,
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
        actions = torch.as_tensor(actions, dtype=torch.int64).to(DEVICE)
        rewards = torch.as_tensor(rewards, dtype=torch.float32).to(DEVICE)
        next_states = torch.as_tensor(np.array(next_states), dtype=torch.float32).to(
            DEVICE
        )
        not_dones = torch.logical_not(torch.as_tensor(dones, dtype=torch.bool)).to(
            DEVICE
        )

        # Compute Q-values for the current state
        self.online_network.train()
        self.optimizer.zero_grad()

        q_values = self.online_network(states)
        q_values = q_values.gather(1, actions.unsqueeze(1))

        # Compute the target Q-values using the target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states)
        max_next_q_values, _ = next_q_values.max(1)
        target_q_values = rewards + not_dones * GAMMA * max_next_q_values

        # Compute the loss
        # There is a paragraph in the paper about the optimization of the loss function
        # to clip the error between -1 and 1 but the explanation is not very clear
        # so I use the Huber loss instead with delta=1
        loss = torch.nn.HuberLoss()(q_values, target_q_values.unsqueeze(1))

        # Optimize the model
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
        play : bool, optional
            Play mode, by default False
        """
        self.model_name = model_name
        model_path = self._get_model_path()
        states = torch.load(model_path)
        self.online_network.load_state_dict(states["state_dict"])
        self.target_network.load_state_dict(states["target_state_dict"])
        self.optimizer.load_state_dict(states["optimizer"])
        self.episodes = states["episodes"]
        self.steps = states["steps"]
        self.hours = states["hours"]
        self.time_to_save = states["time_to_save"]
        self.best_mean_reward = states["best_mean_reward"]
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
        if not model_name and not self.model_name:
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

        if not self.model_name:
            self.model_name = model_name

        writer = SummaryWriter(
            log_dir=os.path.join(
                "logs",
                self.model_name,
                time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()),
            )
        )

        print("=======")
        print(
            f"Training on {self.game_name} with model {self.model_name} for {hours_to_train} hours\n"
            f"Already done: {self.hours} hours, {self.episodes} episodes, {self.steps} steps"
        )
        print("=======")

        time_spent = 0
        reward_last_100_episodes = deque(maxlen=100)
        length_last_100_episodes = deque(maxlen=100)

        # - self.time_to_save because we trained for self.time_to_save seconds more than the last hour saved
        max_seconds = hours_to_train * SECOND_PER_HOUR - self.time_to_save
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
                    self.target_network.load_state_dict(
                        self.online_network.state_dict()
                    )

                episode_reward += reward
                episode_step += 1
                self.steps += 1

                if done:
                    break

            reward_last_100_episodes.append(episode_reward)
            length_last_100_episodes.append(episode_step)

            writer.add_scalar(
                "Epsilon at the end of the episode",
                self.policy.get_current_epsilon(),
                self.episodes,
            )
            writer.add_scalar("Rewards/Episode", episode_reward, self.episodes)
            writer.add_scalar(
                "Rewards/Last_100_episodes",
                np.mean(reward_last_100_episodes),
                self.episodes,
            )
            writer.add_scalar("Lengths/Episode", episode_step, self.episodes)
            writer.add_scalar(
                "Lengths/Last_100_episodes",
                np.mean(length_last_100_episodes),
                self.episodes,
            )

            self.episodes += 1
            time_spent = int(time.time() - start_time)
            increment = time_spent - progress_bar.n
            progress_bar.update(increment)
            self.time_to_save += increment

            # Save the model every hour
            if self.time_to_save >= SECOND_PER_HOUR:
                self.time_to_save = self.time_to_save % SECOND_PER_HOUR
                self.hours += 1
                self._save_model()

            # Save the best model with the best mean reward over the last 100 episodes
            if (
                time_spent >= SECOND_PER_HOUR
                and np.mean(reward_last_100_episodes) > self.best_mean_reward
            ):
                self.best_mean_reward = np.mean(reward_last_100_episodes)
                self._save_model("best")

        progress_bar.close()

        writer.flush()
        writer.close()

        self.env.close()

        print("Saving last model... (this may take a while)")
        self._save_model("last")

    def play(self, is_recording: bool) -> None:
        """
        Play with the agent

        Parameters
        ----------
        is_recording : bool
            Whether to record the game or not
        """
        total_steps = 0
        total_clipped_reward = 0.0
        total_unclipped_reward = 0.0

        if is_recording:
            self.env = RecordVideo(
                self.env,
                os.path.join("videos", self.game_name),
                disable_logger=True,
                name_prefix=f"{self.model_name}_{time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())}",
            )
            self.env.metadata["render_fps"] = 30

        print("=======")
        print(
            f"Playing {self.game_name} with model {self.model_name}\n"
            f"Trained for {self.episodes} episodes, {self.steps} steps, {self.hours} hours"
        )
        print("=======")

        state, _ = self.env.reset()

        if is_recording:
            self.env.start_video_recorder()

        while True:
            action = self._get_action(state)

            state, reward, done, _, info = self.env.step(action)

            total_clipped_reward += reward
            total_unclipped_reward += info["real_reward"]
            total_steps += 1

            # Stop the recording after 5_000 steps
            # This is to avoid looping forever
            if is_recording and total_steps == 5_000:
                break

            if done:
                break

        self.env.close()

        print(
            f"Total clipped reward: {total_clipped_reward}, total unclipped reward: {total_unclipped_reward}, total steps: {total_steps}"
        )
