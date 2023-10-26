class EpsilonGreedyPolicy():
    def __init__(self, epsilon: float, epsilon_decay: float=0.99, epsilon_end: float=0.1):
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_end = epsilon_end

    def get_epsilon(self):
        return self.epsilon
    
    def update_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
