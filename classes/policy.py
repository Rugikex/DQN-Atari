from enum import Enum
from typing import List

from tqdm import tqdm


class DecayState(Enum):
    """
    Decay state

    DECAY_ENDED: Decay ended
    DECAY_NOT_STARTED: Decay not started
    DECAY_FINISH_SEQUENCE: Decay finish sequence
    DECAY_ONGOING: Decay ongoing
    """

    DECAY_ENDED = 1
    DECAY_NOT_STARTED = 2
    DECAY_FINISH_SEQUENCE = 3
    DECAY_ONGOING = 4


class EpsilonGreedyPolicy:
    """
    Epsilon-greedy policy

    Parameters
    ----------
    epsilon_values : List[float]
        List of epsilon values for decay
    steps_to_epsilon_values : List[int]
        List of steps to reach each epsilon value
    """

    def __init__(
        self, epsilon_values: List[float], steps_to_epsilon_values: List[int]
    ) -> None:
        if len(epsilon_values) != len(steps_to_epsilon_values):
            raise ValueError(
                "Length of epsilon values should match the length of steps to epsilon values"
            )

        if len(epsilon_values) == 0:
            raise ValueError("Epsilon values should not be empty")

        self._epsilon = epsilon_values[0]
        self._index = 0
        self._epsilon_values = epsilon_values
        self._steps_to_epsilon_values = steps_to_epsilon_values
        self._current_decay = 0
        self._steps = 0
        self._decay_state = (
            DecayState.DECAY_NOT_STARTED
            if len(epsilon_values) > 1
            else DecayState.DECAY_ENDED
        )

    def _update_decay_state(self) -> None:
        if self._index == len(self._epsilon_values) - 1:
            self._decay_state = DecayState.DECAY_ENDED
        else:
            self._decay_state = DecayState.DECAY_ONGOING
            self._current_decay = (
                self._epsilon_values[self._index]
                - self._epsilon_values[self._index + 1]
            ) / (
                self._steps_to_epsilon_values[self._index + 1]
                - self._steps_to_epsilon_values[self._index]
            )

    def decaying_epsilon(self, steps: int) -> None:
        """
        Decay epsilon based on steps
        Use for retraining the model

        Parameters
        ----------
        steps : int
            Number of steps already done
        """
        steps_to_decay = steps - self._steps_to_epsilon_values[0]
        if (
            steps_to_decay
            >= self._steps_to_epsilon_values[len(self._steps_to_epsilon_values) - 1]
        ):
            self._decay_state = DecayState.DECAY_ENDED
            self._epsilon = self._epsilon_values[len(self._epsilon_values) - 1]
            return

        if steps_to_decay > 0:
            for _ in tqdm(range(steps_to_decay), desc="Decaying epsilon"):
                self.get_epsilon()

    def get_current_epsilon(self) -> float:
        """
        Get current epsilon value
        Use for logging

        Returns
        -------
        float
            Current epsilon value
        """
        return self._epsilon

    def get_epsilon(self) -> float:
        """
        Get epsilon value
        Apply decaying epsilon if needed

        Returns
        -------
        float
            Epsilon value
        """
        if self._decay_state == DecayState.DECAY_ENDED:
            return self._epsilon

        self._steps += 1

        if self._decay_state == DecayState.DECAY_FINISH_SEQUENCE:
            self._index += 1
            self._update_decay_state()

        if self._decay_state == DecayState.DECAY_ONGOING:
            self._epsilon -= self._current_decay
            if self._steps == self._steps_to_epsilon_values[self._index + 1]:
                self._decay_state = DecayState.DECAY_FINISH_SEQUENCE
                self._epsilon = self._epsilon_values[self._index + 1]

        if self._decay_state == DecayState.DECAY_NOT_STARTED:
            if self._steps >= self._steps_to_epsilon_values[self._index]:
                self._update_decay_state()

        return self._epsilon
