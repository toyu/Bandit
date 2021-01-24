import numpy as np
import copy as cp
import math
import random

# greedy法
def greedy(values):
    max_values = np.where(values == values.max())
    return np.random.choice(max_values[0])

# エージェントの親クラス
class Agent():
    def __init__(self, k):
        self.arm_counts = np.zeros(k)
        self.arm_rewards = np.zeros(k)
        self.arm_values = np.zeros(k)
        self.opt_r = 0

    def recet_params(self):
        self.value = np.zeros_like(self.value)

    def select_arm(self):
        pass

    def update(self, selected, reward):
        pass

# PS
class PS(Agent):
    def __init__(self, k):
        super().__init__(k)
        self.k = k
        # sorted_index = np.argsort(bandit.probability)[::-1]
        # self.r = (bandit.probability[sorted_index[0]] + bandit.probability[sorted_index[1]]) / 2

    def reset_params(self):
        super().recet_params()

    # 腕を選択
    def select_arm(self):
        if np.max(self.arm_values) > self.opt_r:
            return greedy(self.arm_values)
        else:
            return random.randint(self.k)

    # 価値の更新
    def update(self, selected, reward):
        self.arm_counts[selected] += 1
        self.arm_rewards[selected] += reward
        self.arm_values[selected] = self.arm_rewards[selected] / self.arm_counts[selected]

        
# thompson sampling
class TS(Agent):
    def __init__(self, k):
        super().__init__(k)
        self.sampling = np.ones((k, 2))
        self.alpha = 1
        self.beta = 1
        self.reward_sum = 0
        self.count = 0
        self.arm_w_counts = np.zeros(k)
        self.arm_l_counts = np.zeros(k)

    def recet_params(self):
        super().recet_params()

    # 腕を選択
    def select_arm(self):
        return np.argmax([np.random.beta(n[0], n[1], 1) for n in self.sampling])

    # 価値の更新
    def update(self, selected, reward):
        self.count += 1
        self.reward_sum += reward
        self.arm_w_counts[selected] += reward
        self.arm_l_counts[selected] += 1 - reward
        self.sampling[selected] = (self.arm_w_counts[selected]+self.alpha, self.arm_l_counts[selected]+self.beta)


# 減衰率付き thompson sampling
class TS_gamma(Agent):
    def __init__(self, k, gamma=0.999):
        super().__init__(k)
        self.sampling = np.ones((k, 2))
        self.gamma = gamma
        self.alpha = 1
        self.beta = 1
        self.reward_sum = 0
        self.count = 0
        self.arm_w_counts = np.zeros(k)
        self.arm_l_counts = np.zeros(k)

    def recet_params(self):
        super().recet_params()

    # 腕を選択
    def select_arm(self):
        return np.argmax([np.random.beta(n[0], n[1], 1) for n in self.sampling])

    # 価値の更新
    def update(self, selected, reward):
        self.arm_w_counts *= self.gamma
        self.arm_l_counts *= self.gamma
        self.arm_w_counts[selected] += reward
        self.arm_l_counts[selected] += 1 - reward
        self.arm_rewards[selected] += reward
        self.sampling[selected] = (self.arm_w_counts[selected]+self.alpha, self.arm_l_counts[selected]+self.beta)


# UCB1T
class UCB1T(Agent):
    def __init__(self, k):
        super().__init__(k)
        self.f = 0
        self.reward_sum = 0
        self.arm_rewards_square = np.zeros(k)
        self.k = k
        self.count = 0

    def reset_params(self):
        super().recet_params()

    # 腕を選択
    def select_arm(self):
        if self.f == 0 and np.amin(self.arm_counts) == 0:
            return np.argmin(self.arm_counts)
        else:
            self.f = 1
            return greedy(self.arm_values)

    # 価値の更新
    def update(self, selected, reward):
        self.count += 1
        self.reward_sum += reward
        self.arm_counts[selected] += 1
        self.arm_rewards[selected] += reward
        self.arm_rewards_square[selected] += reward * reward

        # if self.count >= self.k:
        if self.f == 1:
            for i in range(0, self.k):
                ave = self.arm_rewards[i] / self.arm_counts[i]
                variance = self.arm_rewards_square[i] / self.arm_counts[i] - ave * ave
                v = variance + math.sqrt((2.0 * math.log(self.count)) / self.arm_counts[i])
                self.arm_values[i] = ave + math.sqrt((math.log(self.count) / self.arm_counts[i]) * min(0.25, v))

                
# UCB1T + メタバンディット
class meta_UCB1T(Agent):
    def __init__(self, k, l=500, delta=0, lmd=30):
        super().__init__(k)
        self.old_agent = UCB1T(k)
        self.new_agent = None
        self.higher_agent = None
        self.meta_flag = 0
        self.l = l
        self.l_count = 0
        self.delta = delta
        self.lmd = lmd
        self.step = 0
        self.selected = 0
        self.k = k
        self.mT_sum = 0
        self.MT = 0

    def reset_params(self):
        self.meta_flag = 0
        self.l_count = 0
        self.mT_sum = 0
        self.MT = 0

    # 腕を選択
    def select_arm(self):
        if self.meta_flag:
            self.selected = self.higher_agent.select_arm()

            if self.selected == 0:
                return self.old_agent.select_arm()
            else:
                return self.new_agent.select_arm()
        else:
            return self.old_agent.select_arm()

    # 価値の更新
    def update(self, selected, reward):
        self.step += 1

        # 旧エージェントと新エージェントのどちらが優秀か検証している間
        if self.meta_flag:
            self.l_count += 1
            self.higher_agent.update(self.selected, reward)
            self.old_agent.update(selected, reward)
            self.new_agent.update(selected, reward)

            if self.l_count == self.l:
                self.reset_params()

                if greedy(self.higher_agent.arm_rewards) == 1:
                    self.old_agent = self.new_agent

        else:
            self.old_agent.update(selected, reward)
            rt_mean = self.old_agent.reward_sum / self.old_agent.count
            mT = reward - rt_mean + self.delta
            self.mT_sum += mT
            if self.mT_sum > self.MT:
                self.MT = self.mT_sum
            PHT = self.MT - self.mT_sum

            # 環境の変化を検知
            if PHT > self.lmd:
                self.meta_flag = 1
                # 新しいエージェントを作る（価値をリセット）
                self.new_agent = UCB1T(self.k)
                # 旧エージェントと新エージェントのどちらで行動を選択するか選ぶエージェントを生成
                self.higher_agent = UCB1T(2)
