import random

import gymnasium as gym
import numpy as np

from classes.stacked_frames import StackedFrames


class AtariWrapper(gym.Wrapper):
    """
    Wrapper for Atari games

    Parameters
    ----------
    env : gym.Env
        Environment
    skip_frames : int
        Number of frames to skip between each action
    play : bool
        Whether to play the game or train the agent
    """

    def __init__(self, env, skip_frames=4, play=False) -> None:
        super().__init__(env)
        self.skip_frames = skip_frames
        self.play = play
        self.lives = 0
        self.has_to_reset = True
        self.no_op_action = [
            action
            for action, meaning in enumerate(env.unwrapped.get_action_meanings())
            if meaning == "NOOP"
        ][0]
        self.stacked_frames = StackedFrames(4)
        self.previous_state: np.ndarray = None
        self.info = {}

    def reset(self, **kwargs) -> tuple:
        """
        Reset the environment
        Play no-op action between 1 and 30 frames at the beginning of the game
        Only really reset the environment if the game is over

        Returns
        -------
        stacked_frames : np.ndarray
            Stacked frames (4, 84, 84)
        info : dict
            Information about the environment
        """
        if self.has_to_reset:
            self.has_to_reset = False
            state, info = self.env.reset()
            self.lives = info["lives"]
            # Play no-op action between 1 and 30 frames at the beginning of the game
            # for play only
            if self.play:
                for _ in range(random.randint(1, 30)):
                    state, _, done, _, info = self.env.step(self.no_op_action)
                    if info["lives"] != self.lives:
                        done = True
                    if done:
                        state, info = self.env.reset()

            self.stacked_frames.reset(state)
            self.previous_state = state

        else:
            info = self.info

        return self.stacked_frames.get_frames(), info

    def step(self, action) -> tuple:
        """
        Step the environment with the given action
        Repeat the action self.skip_frames times and stack the frames

        Parameters
        ----------
        action : int
            Action to take

        Returns
        -------
        stacked_frames : np.ndarray
            Stacked frames (4, 84, 84)
        reward : float
            Reward
        terminated : bool
            Whether the agent reaches the terminal state
        truncated : bool
            Whether the truncation condition outside the scope of the MDP is satisfied
        info : dict
            Information about the environment
        """
        sum_reward = 0.0
        next_state = self.previous_state
        terminated_episode = False
        has_truncated = False
        for _ in range(self.skip_frames):
            self.previous_state = next_state
            next_state, reward, terminated, truncated, info = self.env.step(action)
            self.has_to_reset = terminated
            sum_reward += reward
            if info["lives"] != self.lives:
                self.lives = info["lives"]
                if not self.play:
                    terminated_episode = True
            if truncated and not self.play:
                has_truncated = True

        self.stacked_frames.append(next_state, self.previous_state)
        self.previous_state = next_state

        self.info = info
        terminated = terminated or terminated_episode or has_truncated
        info["real_reward"] = sum_reward

        return (
            self.stacked_frames.get_frames(),
            np.sign(sum_reward).astype(np.float32),
            terminated,
            has_truncated,
            info,
        )
