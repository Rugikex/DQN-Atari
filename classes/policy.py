from tqdm import tqdm


class EpsilonGreedyPolicy:
    """
    Epsilon-greedy policy

    Parameters
    ----------
    epsilon_init : float
        Initial epsilon value
    epsilon_end : float
        Final epsilon value
    steps_to_start_decay : int
        Number of steps to start decaying epsilon
    steps_to_epsilon_end : int
        Number of steps to reach final epsilon value
    """
    def __init__(
        self,
        epsilon_init: float,
        epsilon_end: float,
        steps_to_start_decay: int,
        steps_to_epsilon_end: int,
    ) -> None:
        self._epsilon = epsilon_init
        self._epsilon_end = epsilon_end
        self._steps = 0
        self._steps_to_start_decay = steps_to_start_decay
        self._steps_to_epsilon_end = steps_to_epsilon_end
        denom = steps_to_epsilon_end - steps_to_start_decay
        if denom == 0:
            self._epsilon_decay = 0
        else:
            self._epsilon_decay = (epsilon_init - epsilon_end) / (
                steps_to_epsilon_end - steps_to_start_decay
            )
        self._is_decay_ended = False

    def decaying_epsilon(self, steps: int) -> None:
        """
        Decaying epsilon
        Use for retraining the model

        Parameters
        ----------
        steps : int
            Number of steps already done
        """
        steps_to_decay = steps - self._steps_to_start_decay
        if steps_to_decay >= self._steps_to_epsilon_end:
            self._is_decay_ended = True
            self._epsilon = self._epsilon_end
        elif steps_to_decay > 0:
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
        if not self._is_decay_ended:
            # Decay epsilon linearly
            if self._steps_to_start_decay <= self._steps < self._steps_to_epsilon_end:
                self._epsilon = self._epsilon - self._epsilon_decay
            elif self._steps == self._steps_to_epsilon_end:
                self._is_decay_ended = True
                self._epsilon = self._epsilon_end
            self._steps += 1

        return self._epsilon
