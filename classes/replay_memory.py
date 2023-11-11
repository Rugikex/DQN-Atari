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
    """

    def __init__(self, max_size) -> None:
        self.buffer = [None] * max_size
        self.max_size = max_size
        self.index = 0
        self.size = 0

    def __len__(self) -> int:
        return self.size

    def append(self, obj) -> None:
        self.buffer[self.index] = obj
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
            List of random transitions
        """
        indices = sample(range(self.size), batch_size)
        return [self.buffer[index] for index in indices]
