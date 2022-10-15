#%%
from environment import SlotMachine
from agent import Player, Strategy
from playground import Playground
n_runs = 2000
n_timesteps = 10000
k = 10
strategy = Strategy(
    'eps_greedy',
    0.1
)
environment = SlotMachine(
    n_arms = k, 
    timesteps_per_episode = n_timesteps, 
    stationary = False
)
agent_0 = Player(
    n_arms = k, 
    strategy = strategy,
    update_type = 'constant',
    alpha = 0.1
)
agent_1 = Player(
    n_arms = k,
    strategy = strategy,
    update_type = 'average'
)
playground = Playground(
    n_runs = n_runs,
    environment = environment,
    agents = [(agent_0, 'constant'), (agent_1, 'average')]
)
if __name__ == '__main__':
    playground.run_experiment()
    playground.plot_results()