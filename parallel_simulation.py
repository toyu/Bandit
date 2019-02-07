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
                      ag.TS(k),
                      ag.RS_gamma(k, gamma=1.0),
                      ag.RS_OPT(k),
                      ag.TS_gamma(k),
                      ag.RS_gamma(k),
                      ag.RS_OPT_gamma(k),
                      ag.meta_bandit(k, agent=ag.UCB1T(k), higher_agent=ag.UCB1T(2), l=500, delta=0, lmd=30),
                      ag.meta_bandit(k, agent=ag.TS(k), higher_agent=ag.TS(2), l=30, delta=0, lmd=30),
                      ag.meta_bandit(k, agent=ag.RS_gamma(k, gamma=1.0), higher_agent=ag.RS_gamma(2, gamma=1.0), l=30, delta=0, lmd=30),
                      ag.meta_bandit(k, agent=ag.RS_OPT(k), higher_agent=ag.RS_OPT(2), l=30, delta=0, lmd=30)]
        # agent_list = [ag.meta_bandit(k, agent=ag.RS_gamma(k, gamma=1.0), higher_agent=ag.RS_gamma(2, gamma=1.0), l=30, delta=0, lmd=30),
        #               ag.meta_bandit(k, agent=ag.TS(k), higher_agent=ag.TS(2), l=30, delta=0, lmd=30)]
        agent_list = [ag.RS_gamma(k),
                      ag.RS_OPT_gamma(k)]

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

        colors = ["g", "b", "r", "m", "c", "#ff7f00", "#a65628", "g", "b", "r", "m"]
        markers = ["o", "s", "^", "v", "<", ">", "p", "D", "x", "1", "2"]
        line_styles = ["dashed", "dashed", "dashed", "dashed", "dashed", "dashed", "dashed", "solid", "solid", "solid", "solid"]


        plt.figure(figsize=(8, 5), dpi=100)
        if i == 1:
            plt.ylim([0, 3000])
        # plt.xscale("log")
        plt.xlabel('step', fontsize=12)
        plt.ylabel(y_label[i], fontsize=12)
        for l in range(len(graphs)):
            plt.plot(graphs[l], alpha=0.8, linewidth=1.2, label=" ", color=colors[l], linestyle=line_styles[l], marker=markers[l], markevery=4999)
        plt.legend(loc="best", frameon=False)
        plt.savefig(graph_titles[i])
        plt.clf()
        # plt.show()

    accuracy_header = "UCB1T accuracy,TS accuracy,RS accuracy,RS OPT accuracy,TS gamma accuracy,RS gamma accuracy,RS OPT gamma accuracy,meta UCB1T accuracy,meta TS accuracy,meta RS accuracy,meta RS OPT accuracy"
    regret_header = ",UCB1T regret,TS regret,RS regret,RS OPT regret,TS gamma regret,RS gamma regret,RS OPT gamma regret,meta UCB1T regret,meta TS regret,meta RS regret,meta RS OPT regret"
    np.savetxt(file_name, file_data.transpose(), delimiter=",", header=accuracy_header+regret_header)


simulation_num = 300
job_num = 3
simulation_num_per_job = int(simulation_num / job_num)
step_num = 100000
k = 20
is_nonsteady = 1
is_constant = 0
agent_num = 11
labels = ["UCB1T", "TS", "RS", "RS OPT", "TS gamma", "RS gamma", "RS OPT gamma", "meta UCB1T", "meta TS", "meta RS", "meta RS OPT"]
y_label = ["accuracy", "regret"]
graph_titles = ["accuracy", "regret"]
file_name = "steady.csv"
file_name = "nonsteady.csv"
# file_name = "non-steady_constant.csv"
file_name = "test.csv"

print(file_name)
start = time()

data = Parallel(n_jobs=job_num)([delayed(simulation_parallel)(simulation_num_per_job, step_num, k, len(labels), is_nonsteady, is_constant) for i in range(job_num)])
plot_graph(data, len(labels), len(graph_titles), step_num, job_num)

print('{}秒かかりました'.format(time() - start))