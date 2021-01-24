import numpy as np
import matplotlib.pyplot as plt
import environment as en
import agent as ag

# シミュレーション
def simulation(simulation_num, step_num, k, agent_num, is_unsteady, is_constant):
    labels = ["meta_UCB1T"]
    # 記録用の配列を用意
    accuracy = np.zeros((agent_num, step_num))
    regrets = np.zeros((agent_num, step_num))
    # 前回選択した腕と今回選択した腕が違うか判定
    replacements = np.zeros((agent_num, step_num))
    
    # シミュレーション
    for sim in range(simulation_num):
        print(sim + 1)
        bandit = en.Bandit(k)
        agent_list = [ag.UCB1T(k), ag.RS(k), ag.meta_UCB1T(k), ag.meta_RS(k), ag.RS_gamma(k)]
        regret_sums = np.zeros(agent_num)
        prev_selecteds = np.zeros(agent_num)
        for step in range(step_num):
            prev_selected = 0
            regret = 0

            # バンディット変化
            if is_unsteady:
                if is_constant:
                    if step % 10000 == 0:
                        bandit = en.Bandit(k)
                else:
                    if np.random.rand() < 0.0001:
                        bandit = en.Bandit(k)

            for i, agent in enumerate(agent_list):
                # 腕の選択
                selected = agent.select_arm()
                # 前と同じ行動か?
                if selected == int(prev_selected[i]):
                    replacement = 0
                else:
                    replacement = 1
                    prev_selecteds[i] = selected
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

    # それぞれの指標のシミュレーションごとの平均を算出
    accuracy /= simulation_num
    regrets /= simulation_num
    replacements /= simulation_num

    # プロット
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
