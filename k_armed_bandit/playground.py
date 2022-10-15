#%%
import matplotlib.pyplot as plt
import multiprocessing
import copy

class Playground():

    def __init__(
        self,
        n_runs,
        environment,
        agents
    ):
        self.n_runs = n_runs
        self.env = environment
        self.n_timesteps = self.env.get_timesteps()
        self.agents = [agent[0] for agent in agents] # list of tuples of the form (object of class Player, 'name')
        self.agents_names = [agent[1] for agent in agents]
        self.results = None

    def _choose_actions(self, env, agents, timestep, results):
        optimal_action = env.optimal_action()
        actions = []
        for agent, agent_name in zip(agents, self.agents_names):
            action = agent.choose_action()
            actions.append(action)
            results[agent_name]['optimal_action'][timestep] = \
                action == optimal_action
        return actions

    def _get_rewards(self, env, actions, timestep, results):
        rewards = []
        for i, agent_name in enumerate(self.agents_names):
            reward = env.get_reward(actions[i])
            rewards.append(reward)
            results[agent_name]['rewards'][timestep] = reward
        return rewards 

    def _update_agents(self, agents, actions, rewards):
        for agent, (action, reward) in zip(agents, zip(actions, rewards)):
            agent.update(action, reward)

    def _perform_one_run(self, run):
        # run - unused argument to allow for pool.map usage
        env = copy.deepcopy(self.env)
        agents = copy.deepcopy(self.agents)
        n_timesteps = self.env.get_timesteps()
        results = {}
        for agent_name in self.agents_names:
            results[agent_name] = {
                'rewards': [None for _ in range(n_timesteps)],
                'optimal_action': [None for _ in range(n_timesteps)]
            }
        for timestep in range(n_timesteps):
            actions = self._choose_actions(env, agents, timestep, results)
            rewards = self._get_rewards(env, actions, timestep, results)
            self._update_agents(agents, actions, rewards)
        return results

    def run_experiment(self):
        runs = [i for i in range(self.n_runs)]
        p = multiprocessing.Pool()
        results = p.map(self._perform_one_run, runs)
        self.results = results

    def _preprocess_results(self):
        mean_rewards = {name: [None for _ in range(self.n_timesteps)] for name in self.agents_names}
        optimal_actions = {name: [None for _ in range(self.n_timesteps)] for name in self.agents_names}
        for agent in self.agents_names:
            for t in range(self.n_timesteps):
                mean_rewards[agent][t] = sum(
                    [self.results[run][agent]['rewards'][t] for
                         run in range(self.n_runs)]
                         )/self.n_runs
                optimal_actions[agent][t] = sum(
                    [self.results[run][agent]['optimal_action'][t] for
                         run in range(self.n_runs)]
                         )/self.n_runs
        return mean_rewards, optimal_actions

    def plot_results(self):
        mean_rewards, optimal_actions = self._preprocess_results()
        plt.figure(figsize = (12, 10))
        for agent in self.agents_names:
            plt.plot(mean_rewards[agent], label = agent)
        plt.title(f'Mean rewards over {self.n_runs} runs per agent')
        plt.ylabel('Value')
        plt.xlabel('Timestep')
        plt.legend()
        plt.figure(figsize = (12, 10))
        for agent in self.agents_names:
            plt.plot(optimal_actions[agent], label = agent)
        plt.title(f'Optimal action ratio over {self.n_runs} runs per agent') 
        plt.ylabel('Ratio')
        plt.xlabel('Timestep')
        plt.ylim(bottom = 0, top = 1)
        plt.legend()
        plt.show()