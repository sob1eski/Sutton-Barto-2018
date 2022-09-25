import random

class SlotMachine():

    def __init__(self, k, stationary = True, init_true_q = 0.0): 
        if stationary:
            self.true_q_values = [
                random.gauss(0, 1) for _ in range(k)
            ]
        elif type(init_true_q) == float:
            self.true_q_values = [
                init_true_q for _ in range(k)
            ]
        elif type(init_true_q) == list:
            self.true_q_values = init_true_q
        
        self.stationary = stationary

    def get_reward(self, action):
        reward = random.gauss(self.true_q_values[action], 1)
        return reward
    
    def add_noise(self):
        if not self.stationary:
            self.true_q_values = [
                v + random.gauss(0, 0.01) for v \
                    in self.true_q_values
            ]
    
    def optimal_action(self):
        max_q = max(self.true_q_values)
        action = self.true_q_values.index(max_q)
        return action