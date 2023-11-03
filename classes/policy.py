import parameters


class EpsilonGreedyPolicy():
    def __init__(
            self,
            epsilon_init: float,
            epsilon_end: float=0.1,
            steps: int=0,
            steps_to_start_decay: int=parameters.start_update,
            steps_to_epsilon_end: int=1_000_000,
    ) -> None:
        self._epsilon = epsilon_init
        self._epsilon_end = epsilon_end
        self._steps = steps
        self._steps_to_start_decay = steps_to_start_decay
        self._steps_to_epsilon_end = steps_to_epsilon_end
        self._epsilon_decay = (epsilon_init - epsilon_end) / (steps_to_epsilon_end - steps_to_start_decay)
        self._is_decay_ended = False

        for _ in range(steps):
            self.get_epsilon()
            if self._is_decay_ended:
                break

    def get_epsilon(self) -> float:
        if not self._is_decay_ended:
            if self._steps_to_start_decay <= self._steps < self._steps_to_epsilon_end:
                self._epsilon = self._epsilon - self._epsilon_decay
            elif self._steps == self._steps_to_epsilon_end:
                self._is_decay_ended = True
                self._epsilon = self._epsilon_end
            self._steps += 1

        return self._epsilon
