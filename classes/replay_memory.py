from random import sample


class ReplayMemory:
    """
    Replay memory that stores the transitions
    Deque is not used because it is slower with random access

    Parameters
    ----------
    max_size: int
        Maximum size of the replay memory

    Attributes
    ----------
    buffer: list
        List of transitions
    max_size: int
        Maximum size of the replay memory
    index: int
        Index of the last transition
    size: int
        Current size of the replay memory
    last_state: np.ndarray
        Last state of the last transition
    """

    def __init__(self, max_size) -> None:
        self.buffer = [None] * max_size
        self.max_size = max_size
        self.index = 0
        self.size = 0
        self.last_state = None

    def __len__(self) -> int:
        return self.size

    def __getstate__(self):
        return {
            "buffer": self.buffer,
            "max_size": self.max_size,
            "index": self.index,
            "size": self.size,
            "last_state": self.last_state,
        }

    def __setstate__(self, state):
        self.buffer = state["buffer"]
        self.max_size = state["max_size"]
        self.index = state["index"]
        self.size = state["size"]
        self.last_state = state["last_state"]

    def append(self, obj) -> None:
        """
        Append a transition to the replay memory
        Object format: (state, action, reward, next_state, done)
        The object is modified to (state, action, reward, reward, done) because in my
        implementation, the skip frames and the number of frames to stack are the same
        so the next state is the state of the next transition.
        With this modification, the replay memory is more memory efficient.

        Parameters
        ----------
        obj: tuple
            Transition to append
        """
        modified_obj = obj[:3] + obj[4:]
        self.last_state = obj[3]
        self.buffer[self.index] = modified_obj
        self.size = min(self.size + 1, self.max_size)
        self.index = (self.index + 1) % self.max_size

    def sample(self, batch_size: int) -> list:
        """
        Sample a batch of transitions

        Parameters
        ----------
        batch_size: int
            Batch size

        Returns
        -------
        list
            List of random transitions with the original format
        """
        indices = sample(range(self.size), batch_size)
        sampled_transitions = []

        for idx in indices:
            if idx == self.index - 1:
                # If the last transition is sampled, use the last frame of the last transition
                transition = (
                    self.buffer[idx][:3] + (self.last_state,) + self.buffer[idx][3:]
                )
            else:
                # If the last transition is not sampled, use the first frame of the next transition
                next_idx = (idx + 1) % self.max_size
                transition = (
                    self.buffer[idx][:3]
                    + (self.buffer[next_idx][0],)
                    + self.buffer[idx][3:]
                )
            sampled_transitions.append(transition)

        return sampled_transitions
