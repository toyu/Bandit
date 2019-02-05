import numpy as np
import matplotlib.pyplot as plt
import environment as en
import agent as ag
from joblib import Parallel, delayed
from time import time


def simulation_parallel(simulation_num, step_num, k, agent_num, is_nonsteady, is_constant):
    accuracy = np.zeros((agent_num, step_num))
    regrets = np.zeros((agent_num, step_num))
    replacements = np.zeros((agent_num, step_num))
    for sim in range(simulation_num):
        print(sim + 1)
        bandit = en.Bandit(k)
        # agent_list = [ag.UCB1T(k), ag.RS(k), ag.meta_UCB1T(k), ag.meta_RS(k), ag.RS_gamma(k)]
        agent_list = [ag.UCB1T(k),
                      ag.RS(k),
                      ag.RS_OPT(k),
                      ag.RS_gamma(k),
                      ag.RS_OPT_gamma(k),
                      ag.meta_bandit(k, agent=ag.UCB1T(k), higher_agent=ag.UCB1T(2), l=500, delta=0, lmd=30),
                      ag.meta_bandit(k, agent=ag.RS(k), higher_agent=ag.RS(2), l=30, delta=0, lmd=30),
                      ag.meta_bandit(k, agent=ag.RS_OPT(k), higher_agent=ag.RS_OPT(2), l=25, delta=0, lmd=30)]

        for agent in agent_list:
            agent.opt_r = bandit.opt_r
        regret_sums = np.zeros(agent_num)
        # prev_selecteds = np.zeros(agent_num)

        for step in range(step_num):
            prev_selected = 0
            regret = 0

            # バンディット変化
            if is_nonsteady:
                if is_constant:
                    if step % 10000 == 0:
                        bandit = en.Bandit(k)
                        for agent in agent_list:
                            agent.opt_r = bandit.opt_r
                else:
                    if np.random.rand() < 0.0001:
                        bandit = en.Bandit(k)
                        for agent in agent_list:
                            agent.opt_r = bandit.opt_r

            for i, agent in enumerate(agent_list):
                # 腕の選択
                selected = agent.select_arm()
                # 前と同じ行動か?
                # if selected == int(prev_selected[i]):
                #     replacement = 0
                # else:
                #     replacement = 1
                #     prev_selecteds[i] = selected
                # 報酬の観測
                reward = bandit.get_reward(int(selected))
                # 価値の更新
                agent.update(selected, reward)
                # accuracy
                accuracy[i][step] += bandit.get_correct(selected)
                # regret
                regret_sums[i] += bandit.get_regret(selected)
                regrets[i][step] += regret_sums[i]
                # replacement
                # replacements[i][step] += replacement

    return np.array([accuracy, regrets])


def plot_graph(data, agent_num, data_type_num, step_num, job_num):
    f = 0
    for i in range(data_type_num):
        graphs = np.zeros((agent_num, step_num))

        for j in range(job_num):
            if data_type_num == 1:
                graphs += data[j]
            else:
                graphs += data[j][i]

        graphs /= simulation_num

        if f == 0:
            f = 1
            file_data = graphs
        else:
            file_data = np.concatenate([file_data, graphs], 0)

        plt.xlabel('step')
        plt.ylabel(y_label[i])
        # plt.ylim([0.0, 1.1])
        # plt.xscale("log")
        for l in range(len(graphs)):
            plt.plot(graphs[l], label=labels[l], alpha=0.8, linewidth=0.8)
        plt.legend(loc="best")
        plt.savefig(graph_titles[i])
        plt.clf()
        # plt.show()

    header = "UCB1T_accuracy,RS_accuracymeta_UCB1T_accuracy,meta_RS_accuracy,RS_γ_accuracy,UCB1T_regret,RS_regret,meta_UCB1T_regret,meta_RS_regret,RS_γ_regret"
    accuracy_header = "UCB1T accuracy,RS accuracy,RS OPT accuracy,RS gamma accuracy,RS OPT gamma accuracy,meta UCB1T accuracy,meta RS accuracy,meta RS OPT accuracy"
    regret_header = ",UCB1T regret,RS regret,RS OPT regret,RS gamma regret,RS OPT gamma regret,meta UCB1T regret,meta RS regret,meta RS OPT regret"
    np.savetxt(file_name, file_data.transpose(), delimiter=",", header=accuracy_header+regret_header)


simulation_num = 1000
job_num = 1
simulation_num_per_job = int(simulation_num / job_num)
step_num = 100000
k = 20
is_nonsteady = 1
is_constant = 0
labels = ["UCB1T", "RS", "RS OPT", "RS gamma", "RS OPT gamma", "meta UCB1T", "meta RS", "meta RS OPT"]
y_label = ["accuracy", "regret"]
graph_titles = ["accuracy3", "regret3"]
file_name = "steady.csv"
file_name = "nonsteady_constant.csv"
file_name = "nonsteady.csv"

print(file_name)
start = time()

data = Parallel(n_jobs=job_num)([delayed(simulation_parallel)(simulation_num_per_job, step_num, k, len(labels), is_nonsteady, is_constant) for i in range(job_num)])
plot_graph(data, len(labels), len(graph_titles), step_num, job_num)

print('{}秒かかりました'.format(time() - start))