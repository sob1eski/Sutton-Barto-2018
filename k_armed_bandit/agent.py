import random
import math

class Player():

    def __init__(
        self,
        k,
        strategy,
        type = 'q_values',
        update_type = 'average',
        alpha = 0.1
    ):
        self.n_arms = k
        self.type = type
        if self.type == 'q_values':
            self.q_values = k * [0.0]
            self.action_counter = k * [0]
        elif self.type == 'gradient':
            self.preferences = k * [0.0]
            self.soft_max = \
                lambda prefrs, action: math.exp(self.preferences[action])\
                    /sum([math.exp(prefrs[i]) for i in range(self.n_arms)])
            self.timestep = 0
            self.baseline = 0
        self.strategy = strategy
        self.update_type = update_type # implement Unbiased Constant-Step-Size Trick
        self.alpha = alpha

    def choose_action(self):
        if self.type == 'q_values':
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
        
        elif self.type == 'gradient':
            probs = [self.soft_max(self.preferences, action) for action\
                 in range(self.n_arms)]
            action = random.choices(
                [i for i in range(self.n_arms)],
                weights = probs
                )
            self.timestep += 1
            return action[0]
    
    def update(self, action_taken, reward):
        if self.type == 'q_values':
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
        elif self.type == 'gradient':
            if self.timestep == 1:
                self.baseline += reward
            else:
                if self.update_type == 'average':
                    self.baseline = self.baseline + (reward - self.baseline) / self.timestep
                elif self.update_type == 'sequence':
                    pass # possible to implement a varying step size
            probs = [self.soft_max(self.preferences, action_taken) for action_taken\
                 in range(self.n_arms)]
            self.preferences = [
                self.preferences[i] + self.alpha * (reward - self.baseline) * (1 - probs[i]) \
                    if i == action_taken else self.preferences[i] - self.alpha * \
                        (reward - self.baseline) * probs[i] for i in range(self.n_arms)
            ]


class Strategy():

    def __init__(self, type, epsilon = 0.1):
        self.type = type
        self.epsilon = epsilon