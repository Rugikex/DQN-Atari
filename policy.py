class EpsilonGreedyPolicy():
    def __init__(self, epsilon: float, epsilon_decay: float=0.99, epsilon_end: float=0.1, epoque_already_played: int=0):
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_end = epsilon_end
        for _ in range(epoque_already_played):
            self.update_epsilon()

    def get_epsilon(self):
        return self.epsilon
    
    def update_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def has_reached_epsilon_end(self):
        return self.epsilon == self.epsilon_end
