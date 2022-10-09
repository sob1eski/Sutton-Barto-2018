#%%
import matplotlib.pyplot as plt
from environment import SlotMachine
from agent import Player, Strategy
n_runs = 2000
n_timesteps = 10000
k = 10
strategy = Strategy(
    'eps_greedy',
    0.1
)
scores_0 = {
    'rewards': [],
    'optimal_action': []
}
scores_1 = {
    'rewards': [],
    'optimal_action': []
}
for run in range(n_runs):
    environment = SlotMachine(k = k, stationary = False)
    agent_0 = Player(
        k = k, 
        strategy = strategy,
        update_type = 'constant',
        alpha = 0.1
    )
    info_0 = [[], []]
    agent_1 = Player(
        k = k, 
        strategy = strategy,
        update_type = 'average'
    )
    info_1 = [[], []]
    for t in range(n_timesteps):
        action_0 = agent_0.choose_action()
        action_1 = agent_1.choose_action()
        optimal_action = environment.optimal_action()
        info_0[1].append(action_0 == optimal_action)
        info_1[1].append(action_1 == optimal_action)
        reward_0 = environment.get_reward(action_0)
        reward_1 = environment.get_reward(action_1)
        info_0[0].append(reward_0)
        info_1[0].append(reward_1)
        environment.add_noise()
        agent_0.update(action_0, reward_0)
        agent_1.update(action_1, reward_1)
    scores_0['rewards'].append(info_0[0])
    scores_0['optimal_action'].append(info_0[1])
    scores_1['rewards'].append(info_1[0])
    scores_1['optimal_action'].append(info_1[1])

mean_rewards_0 = []
mean_rewards_1 = []
for t in range(n_timesteps):
    mean_rewards_0.append(sum([v[t] for v in scores_0['rewards']])/n_runs)
    mean_rewards_1.append(sum([v[t] for v in scores_1['rewards']])/n_runs)

frac_optimal_0 = []
frac_optimal_1 = []
for t in range(n_timesteps):
    frac_optimal_0.append(sum([v[t] for v in scores_0['optimal_action']])/n_runs)
    frac_optimal_1.append(sum([v[t] for v in scores_1['optimal_action']])/n_runs)

plt.figure()
plt.plot(mean_rewards_0, label = 'constant step')
plt.plot(mean_rewards_1, label = 'average')
plt.legend()

plt.figure()
plt.plot(frac_optimal_0, label = 'constant step')
plt.plot(frac_optimal_1, label = 'average')
plt.legend()