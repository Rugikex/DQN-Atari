from random import sample
from typing import List, Tuple

import numpy as np
from numpy.typing import NDArray


class ReplayMemory:
    """
    Replay memory that stores the transitions using numpy arrays for efficient memory management.

    Parameters
    ----------
    max_size: int
        Maximum size of the replay memory
    state_shape: Tuple[int, int, int]
        Shape of the state
    """

    def __init__(
        self, max_size: int, state_shape: Tuple[int, int, int] = (4, 84, 84)
    ) -> None:
        self.max_size: int = max_size
        self.index: int = 0
        self.size: int = 0

        self.states: NDArray[np.uint8] = np.zeros(
            (max_size, *state_shape), dtype=np.uint8
        )
        self.actions: NDArray[np.uint8] = np.zeros(max_size, dtype=np.uint8)
        self.rewards: NDArray[np.int8] = np.zeros(max_size, dtype=np.int8)
        self.dones: NDArray[np.bool_] = np.zeros(max_size, dtype=np.bool_)
        self.last_state: NDArray[np.uint8] = np.zeros(state_shape, dtype=np.uint8)

    def __len__(self) -> int:
        return self.size

    def append(
        self,
        state: NDArray[np.uint8],
        action: np.uint8,
        reward: np.int8,
        next_state: NDArray[np.uint8],
        done: bool,
    ) -> None:
        """
        Append a transition to the replay memory.

        Parameters
        ----------
        state: NDArray[np.uint8]
            Current state
        action: np.uint8
            Action taken
        reward: np.int8
            Reward received
        next_state: NDArray[np.uint8]
            Next state
        done: bool
            Whether the episode is done
        """
        self.states[self.index] = state
        self.actions[self.index] = action
        self.rewards[self.index] = reward
        self.dones[self.index] = done
        self.last_state = next_state

        self.size = min(self.size + 1, self.max_size)
        self.index = (self.index + 1) % self.max_size

    def sample(
        self, batch_size: int
    ) -> Tuple[
        NDArray[np.uint8],
        NDArray[np.uint8],
        NDArray[np.int8],
        NDArray[np.uint8],
        NDArray[np.bool_],
    ]:
        """
        Sample a batch of transitions.

        Parameters
        ----------
        batch_size: int
            Batch size

        Returns
        -------
        Tuple containing batches of states, actions, rewards, next_states, and dones.
        """
        indices: List[int] = sample(range(self.size), batch_size)
        previous_index: int = (self.index - 1) % self.max_size
        if previous_index in indices:
            indices.remove(previous_index)
            indices.append(previous_index)

        batch_states: NDArray[np.uint8] = self.states[indices]
        batch_actions: NDArray[np.uint8] = self.actions[indices]
        batch_rewards: NDArray[np.int8] = self.rewards[indices]
        batch_dones: NDArray[np.bool_] = self.dones[indices]

        batch_next_states: NDArray[np.uint8]
        next_indices: List[int] = [(index + 1) % self.size for index in indices]
        if previous_index in indices:
            batch_next_states = np.concatenate(
                (self.states[next_indices[:-1]], self.last_state[np.newaxis]), axis=0
            )
        else:
            batch_next_states = self.states[next_indices]

        return (
            batch_states,
            batch_actions,
            batch_rewards,
            batch_next_states,
            batch_dones,
        )
