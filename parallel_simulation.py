import numpy as np
import matplotlib.pyplot as plt
import environment as en
import agent as ag
from joblib import Parallel, delayed
from time import time


def simulation_steady_state_parallel(simulation_num, step_num, k, agent_num, environment_type, change_type):
    accuracy = np.zeros((agent_num, step_num))
    regrets = np.zeros((agent_num, step_num))
    replacements = np.zeros((agent_num, step_num))
    for sim in range(simulation_num):
        print(sim + 1)
        bandit = en.Bandit(k)
        agent_list = [ag.meta_UCB1T(k)]
        for i, agent in enumerate(agent_list):
            prev_selected = 0
            regret = 0
            for step in range(step_num):
                # バンディット変化
                if environment_type == "unsteady":
                    if change_type == "constant":
                        if step % 10000 == 0:
                            bandit = en.Bandit(k)
                    else:
                        if np.random.rand() < 0.0001:
                            bandit = en.Bandit(k)
                # 腕の選択
                selected = agent.select_arm()
                if selected == prev_selected:
                    replacement = 0
                else:
                    replacement = 1
                prev_selected = selected
                # 報酬の観測
                reward = bandit.get_reward(int(selected))
                # 価値の更新
                agent.update(selected, reward)
                # accuracy
                accuracy[i][step] += bandit.get_correct(selected)
                # regret
                regret += bandit.get_regret(selected)
                regrets[i][step] += regret
                # replacement
                replacements[i][step] += replacement

    return np.array([accuracy, regrets, replacements])


def plot_graph(data, agent_num, data_type_num, step_num, job_num):
    for i in range(data_type_num):
        graphs = np.zeros((agent_num, step_num))

        for j in range(job_num):
            if data_type_num == 1:
                graphs += data[j]
            else:
                graphs += data[j][i]

        graphs /= simulation_num
        plt.xlabel('step')
        plt.ylabel(graph_titles[i])
        # plt.ylim([0.0, 1.1])
        # plt.xscale("log")
        for l in range(len(graphs)):
            plt.plot(graphs[l], label=labels[l])
        plt.legend(loc="best")
        plt.savefig(graph_titles[i])
        plt.clf()
        # plt.show()


simulation_num = 1000
job_num = 10
simulation_num_per_job = int(simulation_num / job_num)
step_num = 100000
k = 20
agent_num = 1
environment_type = "unsteady"
change_type = "constan"
labels = ["meta_UCB1T"]
# labels = ["500", "750", "1000", "1250", "1500"]
graph_titles = ["accuracy3", "regret3", "replacement_rate3"]

start = time()

data = Parallel(n_jobs=job_num)([delayed(simulation_steady_state_parallel)(simulation_num_per_job, step_num, k, agent_num, environment_type, change_type) for i in range(job_num)])
plot_graph(data, agent_num, len(graph_titles), step_num, job_num)

print('{}秒かかりました'.format(time() - start))