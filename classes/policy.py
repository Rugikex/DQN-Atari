from tqdm import tqdm

from parameters import EPSILON_FINAL_STEP, START_UPDATE


class EpsilonGreedyPolicy:
    """
    Epsilon-greedy policy

    Parameters
    ----------
    epsilon_init : float
        Initial epsilon value
    epsilon_end : float
        Final epsilon value
    steps : int
        Number of steps already done
    steps_to_start_decay : int
        Number of steps to start decaying epsilon
    steps_to_epsilon_end : int
        Number of steps to reach final epsilon value
    """

    def __init__(
        self,
        epsilon_init: float,
        epsilon_end: float = 0.1,
        steps: int = 0,
        steps_to_start_decay: int = START_UPDATE,
        steps_to_epsilon_end: int = EPSILON_FINAL_STEP,
    ) -> None:
        self._epsilon = epsilon_init
        self._epsilon_end = epsilon_end
        self._steps = steps
        self._steps_to_start_decay = steps_to_start_decay
        self._steps_to_epsilon_end = steps_to_epsilon_end
        self._epsilon_decay = (epsilon_init - epsilon_end) / (
            steps_to_epsilon_end - steps_to_start_decay
        )
        self._is_decay_ended = False

        steps_to_decay = steps - steps_to_start_decay
        if steps_to_decay >= self._steps_to_epsilon_end:
            self._is_decay_ended = True
            self._epsilon = self._epsilon_end
        elif steps_to_decay > 0:
            for _ in tqdm(range(steps_to_decay), desc="Decaying epsilon"):
                self.get_epsilon()

    def get_current_epsilon(self) -> float:
        return self._epsilon

    def get_epsilon(self) -> float:
        if not self._is_decay_ended:
            if self._steps_to_start_decay <= self._steps < self._steps_to_epsilon_end:
                self._epsilon = self._epsilon - self._epsilon_decay
            elif self._steps == self._steps_to_epsilon_end:
                self._is_decay_ended = True
                self._epsilon = self._epsilon_end
            self._steps += 1

        return self._epsilon
