import random

import gymnasium as gym
import numpy as np

from classes.stacked_frames import StackedFrames


class AtariWrapper(gym.Wrapper):
    def __init__(self, env, skip_frames=4, play=False):
        super(AtariWrapper, self).__init__(env)
        self.env = env
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

    def reset(self):
        if self.has_to_reset:
            self.has_to_reset = False
            state, info = self.env.reset()
            self.lives = info["lives"]
            # Play no-op action between 1 and 30 frames at the beginning of the game
            for _ in range(random.randint(1, 30)):
                state, _, done, _, info = self.env.step(self.no_op_action)
                if info["lives"] != self.lives:
                    done = True
                if done:
                    state, info = self.env.reset()

        else:
            state, _, _, _, info = self.env.step(self.no_op_action)

        self.stacked_frames.reset(state)
        self.previous_state = state

        return self.stacked_frames.get_frames(), info

    def step(self, action):
        sum_reward = 0.0
        for _ in range(self.skip_frames):
            next_state, reward, done, not_use, info = self.env.step(action)
            self.has_to_reset = done
            self.stacked_frames.append(next_state, self.previous_state)
            self.previous_state = next_state
            sum_reward += reward
            if info["lives"] != self.lives:
                self.lives = info["lives"]
                if self.play:
                    done = self.has_to_reset
                else:
                    done = True

            if done:
                break

        return (
            self.stacked_frames.get_frames(),
            np.sign(sum_reward),
            done,
            not_use,
            info,
        )
