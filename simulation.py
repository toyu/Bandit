import numpy as np
import matplotlib.pyplot as plt
import environment as en
import agent as ag


def simulation(simulation_num, step_num, k, agent_num, is_unsteady, is_constant):
    labels = ["meta_UCB1T"]
    accuracy = np.zeros((agent_num, step_num))
    regrets = np.zeros((agent_num, step_num))
    replacements = np.zeros((agent_num, step_num))
    for sim in range(simulation_num):
        print(sim + 1)
        bandit = en.Bandit(k)
        agent_list = [ag.meta_RS(k)]
        for i, agent in enumerate(agent_list):
            prev_selected = 0
            regret = 0
            for step in range(step_num):
                # バンディット変化
                if is_unsteady:
                    if is_constant:
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

    accuracy /= simulation_num
    regrets /= simulation_num
    replacements /= simulation_num

    plt.xlabel('steps')
    plt.ylabel('accuracy')
    # plt.xscale("log")
    plt.ylim([0.0, 1.1])
    for i, graph in enumerate(accuracy):
        plt.plot(graph, label=labels[i])
    plt.legend(loc="best")
    plt.savefig("accuracy")
    # plt.show()
    plt.cla()

    plt.xlabel('steps')
    plt.ylabel('regret')
    # plt.xscale("log")
    for i, graph in enumerate(regrets):
        plt.plot(graph, label=labels[i])
    plt.legend(loc="best")
    plt.savefig("regret")
    # plt.show()
    plt.cla()

    plt.xlabel('steps')
    plt.ylabel('replacement_rate')
    # plt.xscale("log")
    for i, graph in enumerate(replacements):
        plt.plot(graph, label=labels[i])
    plt.legend(loc="best")
    plt.savefig("replacement_rate")
    # plt.show()
    plt.cla()


simulation(300, 100000, 20, 1, 1, 1)

from joblib import Parallel, delayed
from time import time

# simulation_num = 1000
# job_num = 10
# simulation_num_per_job = int(simulation_num / job_num)
# episode_num = 1200
# agent_num = 2
# data_type_num = 1
# labels = ["RS_GRC", "QL(e_greedy)"]
# graph_titles = ["reward"]
#
# start = time()
#
# data = Parallel(n_jobs=job_num, verbose=10)([delayed(simulation)(simulation_num_per_job, episode_num, agent_num) for i in range(job_num)])
# plot_graph(data, agent_num, data_type_num, episode_num, job_num)
#
# print('{}秒かかりました'.format(time() - start))