from tqdm import tqdm

import parameters


class EpsilonGreedyPolicy:
    def __init__(
        self,
        epsilon_init: float,
        epsilon_end: float = 0.1,
        steps: int = 0,
        steps_to_start_decay: int = parameters.start_update,
        steps_to_epsilon_end: int = parameters.epsilon_final_step,
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

        steps_to_decay = steps - parameters.replay_memory_maxlen
        if steps_to_decay >= self._steps_to_epsilon_end:
            self._is_decay_ended = True
            self._epsilon = self._epsilon_end
        else:
            for _ in tqdm(range(steps_to_decay), desc="Decaying epsilon"):
                self.get_epsilon()

    def get_epsilon(self) -> float:
        if not self._is_decay_ended:
            if self._steps_to_start_decay <= self._steps < self._steps_to_epsilon_end:
                self._epsilon = self._epsilon - self._epsilon_decay
            elif self._steps == self._steps_to_epsilon_end:
                self._is_decay_ended = True
                self._epsilon = self._epsilon_end
            self._steps += 1

        return self._epsilon
