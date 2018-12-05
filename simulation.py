import numpy as np
import matplotlib.pyplot as plt
import environment as en
import agent as ag

def simulation(simulation_num, step_num, k):
    labels = ["UCB1T", "Thompson Sampling", "RS-OPT"]
    labels = ["RS"]
    accuracy = np.zeros((len(labels), step_num))
    regrets = np.zeros((len(labels), step_num))
    for sim in range(simulation_num):
        print(sim + 1)
        bandit = en.Environment(k)
        # agent_list = [ag.UCB1T(bandit, k), ag.TS(bandit, k), ag.RS_OPT(bandit, k)]
        agent_list = [ag.UCB1T(bandit, k)]
        for i, agent in enumerate(agent_list):
            regret = 0
            for step in range(step_num):
                # 腕の選択
                selected = agent.select_arm()
                # 報酬の観測
                reward = bandit.get_reward(int(selected))
                # 価値の更新
                agent.update(selected, reward)
                # accuracy
                accuracy[i][step] += bandit.get_correct(selected)
                # regret
                regret += bandit.get_regret(selected)
                regrets[i][step] += regret

            print(regret)

    accuracy /= simulation_num
    regrets /= simulation_num

    plt.xlabel('steps')
    plt.ylabel('accuracy')
    # plt.xscale("log")
    plt.ylim([0.0, 1.1])
    for i, graph in enumerate(accuracy):
        plt.plot(graph, label=labels[i])
    plt.legend(loc="best")
    plt.savefig("accuracy")
    plt.show()

    plt.xlabel('steps')
    plt.ylabel('regret')
    # plt.xscale("log")
    for i, graph in enumerate(regrets):
        plt.plot(graph, label=labels[i])
    plt.legend(loc="best")
    plt.savefig("regret")
    plt.show()


simulation(10, 1000000, 10)