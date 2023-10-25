class EpsilonGreedyPolicy():
    def __init__(self, epsilon: float, epsilon_decay: float=0.999, epsilon_min: float=0.05):
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

    def get_epsilon(self):
        return self.epsilon
    
    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
