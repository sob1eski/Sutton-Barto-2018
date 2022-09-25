import random

class Player():

    def __init__(
        self,
        k,
        strategy,
        update_type = 'average',
        alpha = 0.1
    ):
        self.n_arms = k
        self.q_values = k * [0.0]
        self.action_counter = k * [0]
        self.strategy = strategy
        self.update_type = update_type
        self.alpha = alpha

    def choose_action(self):
        if self.strategy.type == 'greedy':
            max_q = max(self.q_values)
            max_idx = [
                idx for idx in range(self.n_arms) \
                    if self.q_values[idx] == max_q
            ]
            if len(max_idx) > 1:
                random_idx = random.randint(0, len(max_idx) - 1)
                action = max_idx[random_idx]
            else:
                action = self.q_values.index(max_q)
            self.action_counter[action] += 1
            return action

        elif self.strategy.type == 'eps_greedy':
            epsilon = random.random()
            if epsilon < self.strategy.epsilon:
                action = random.randint(0, self.n_arms - 1)
                self.action_counter[action] += 1
                return action
            else:
                max_q = max(self.q_values)
                max_idx = [
                    idx for idx in range(self.n_arms) \
                        if self.q_values[idx] == max_q
                ]
                if len(max_idx) > 1:
                    random_idx = random.randint(0, len(max_idx) - 1)
                    action = max_idx[random_idx]
                else:
                    action = self.q_values.index(max_q)
                self.action_counter[action] += 1
                return action
    
    def update_q_values(self, action_taken, reward):
        if self.update_type == 'average':
            alpha = 1/self.action_counter[action_taken]
            self.q_values[action_taken] = \
                self.q_values[action_taken] + alpha * \
                    (reward - self.q_values[action_taken])
        elif self.update_type == 'constant':
            self.q_values[action_taken] = \
                self.q_values[action_taken] + self.alpha * \
                    (reward - self.q_values[action_taken])
        elif self.update_type == 'sequence':
            self.q_values[action_taken] = \
                self.q_values[action_taken] + \
                    self.alpha(self.action_counter[action_taken]) * \
                    (reward - self.q_values[action_taken])

class Strategy():

    def __init__(self, type, epsilon = 0.1):
        self.type = type
        self.epsilon = epsilon