class EpsilonGreedyPolicy():
    def __init__(
            self,
            epsilon: float,
            epsilon_decay: float=0.96,
            epsilon_end: float=0.1,
            epoque_already_played: int=0
    ) -> None:
        self.epsilon = epsilon
        self._epsilon_decay = epsilon_decay
        self._epsilon_end = epsilon_end
        for _ in range(epoque_already_played):
            self.update_epsilon()
            if self.epsilon == self._epsilon_end:
                break

    def get_epsilon(self):
        return self.epsilon
    
    def update_epsilon(self):
        self.epsilon = max(self._epsilon_end, self.epsilon * self._epsilon_decay)
