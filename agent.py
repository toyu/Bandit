import numpy as np
import math
import random


def greedy(values):
    max_values = np.where(values == values.max())
    return np.random.choice(max_values[0])


class Agent():
    def __init__(self, k):
        self.arm_counts = np.zeros(k)
        self.arm_rewards = np.zeros(k)
        self.arm_values = np.zeros(k)

    def recet_params(self):
        self.value = np.zeros_like(self.value)

    def select_arm(self):
        pass

    def update(self, selected, reward):
        pass


class PS(Agent):
    def __init__(self, bandit, k):
        super().__init__(k)
        self.k = k
        sorted_index = np.argsort(bandit.probability)[::-1]
        self.r = (bandit.probability[sorted_index[0]] + bandit.probability[sorted_index[1]]) / 2

    def reset_params(self):
        super().recet_params()

    def select_arm(self):
        if np.max(self.arm_values) > self.r:
            return greedy(self.arm_values)
        else:
            return random.randint(self.k)

    def update(self, selected, reward):
        self.arm_counts[selected] += 1
        self.arm_rewards[selected] += reward
        self.arm_values[selected] = self.arm_rewards[selected] / self.arm_counts[selected]


class RS_OPT(Agent):
    def __init__(self, bandit, k):
        super().__init__(k)
        sorted_prob = np.sort(bandit.probability)[::-1]
        self.r = (sorted_prob[0] + sorted_prob[1]) / 2

    def reset_params(self):
        super().recet_params()

    def select_arm(self):
        return greedy(self.arm_values)

    def update(self, selected, reward):
        self.arm_counts[selected] += 1
        self.arm_rewards[selected] += reward
        average = self.arm_rewards[selected] / self.arm_counts[selected]
        self.arm_values[selected] = self.arm_counts[selected] * (average - self.r)


class RS_CH(Agent):
    def __init__(self, k):
        self.arm_num = k
        self.arm_counts = np.ones(k) / 1000
        self.arm_count =np.sum(self.arm_counts)
        self.arm_rewards = np.zeros(k)
        self.arm_values = np.ones(k) / 2

    def reset_params(self):
        super().recet_params()

    def select_arm(self):
        g = greedy(self.arm_values)
        u_array = np.array([])
        r_array = np.array([])
        exceed_indices = []
        f = True

        # for j in range(self.arm_num):
        #     if j == g:
        #         u_array = np.append(u_array, -float('inf'))
        #         r_array = np.append(r_array, -float('inf'))
        #         continue
        #
        #     u = math.exp(-self.arm_counts[j] * ((1 - self.arm_values[j]) * math.log((1 - self.arm_values[j]) / (1 - self.arm_values[g]))) \
        #                                            + self.arm_values[j] * math.log((self.arm_values[j]) / (self.arm_values[g])))
        #
        #     if u == 1:
        #         u_array = np.append(u_array, -float('inf'))
        #         r_array = np.append(r_array, -float('inf'))
        #         continue
        #
        #     r = self.arm_values[g] * ((1 - (self.arm_values[j] / self.arm_values[g]) * u) / (1 - u))
        #
        #     rs_g = (self.arm_counts[g] / self.arm_count) * (self.arm_values[g] - r)
        #     rs_j = (self.arm_counts[j] / self.arm_count) * (self.arm_values[j] - r)
        #
        #     u_array = np.append(u_array, u)
        #     r_array = np.append(r_array, r)
        #     f = False
        #
        #     if rs_g > rs_j:
        #         exceed_indices.append(j)
        #
        # print(u_array)
        # if f:
        #     return np.random.randint(self.arm_num)
        #
        # if len(exceed_indices) == 0:
        #     return g
        # elif len(exceed_indices) == 1:
        #     return exceed_indices[0]
        # else:
        #     exceed_u_array = u_array[exceed_indices]
        #     max_u = np.amax(exceed_u_array)
        #     exceed_indices = np.array(exceed_indices)
        #     max_u_indices = exceed_indices[list(np.where(exceed_u_array == max_u)[0])]
        #     if len(max_u_indices) == 1:
        #         return max_u_indices[0]
        #     else:
        #         exceed_r_array = r_array[list(max_u_indices)]
        #         max_r = np.amax(exceed_r_array)
        #         max_r_indices = max_u_indices[list(np.where(exceed_r_array == max_r)[0])]
        #
        #         if len(max_r_indices) == 1:
        #             return max_r_indices[0]
        #         else:
        #             return max_r_indices[np.random.randint(len(max_r_indices))]

        # --------------------------------------------------------------------------------------------------------------

        for j in range(self.arm_num):
            if j == g:
                u_array = np.append(u_array, -float('inf'))
                r_array = np.append(r_array, -float('inf'))
                continue

            u = math.exp(-self.arm_counts[j] * ((1 - self.arm_values[j]) * math.log((1 - self.arm_values[j]) / (1 - self.arm_values[g]))) \
                                                   + self.arm_values[j] * math.log(self.arm_values[j] / self.arm_values[g]))

            if u == 1:
                u_array = np.append(u_array, -float('inf'))
                r_array = np.append(r_array, -float('inf'))
                continue

            u_array = np.append(u_array, u)

            r = self.arm_values[g] * ((1 - (self.arm_values[j] / self.arm_values[g]) * u) / (1 - u))
            r_array = np.append(r_array, r)
            f = False

        if f:
            return np.random.randint(self.arm_num)

        r_max = np.amax(r_array)

        for j in range(self.arm_num):
            if j == g:
                continue

            rs_g = (self.arm_counts[g] / self.arm_count) * (self.arm_values[g] - r_max)
            rs_j = (self.arm_counts[j] / self.arm_count) * (self.arm_values[j] - r_max)

            if rs_j > rs_g:
                exceed_indices.append(j)

        if len(exceed_indices) == 0:
            return g
        elif len(exceed_indices) == 1:
            return exceed_indices[0]
        else:
            exceed_u_array = u_array[exceed_indices]
            max_u = np.amax(exceed_u_array)
            exceed_indices = np.array(exceed_indices)
            max_u_indices = exceed_indices[list(np.where(exceed_u_array == max_u)[0])]
            if len(max_u_indices) == 1:
                return max_u_indices[0]
            else:
                exceed_r_array = r_array[list(max_u_indices)]
                max_r = np.amax(exceed_r_array)
                max_r_indices = max_u_indices[list(np.where(exceed_r_array == max_r)[0])]

                if len(max_r_indices) == 1:
                    return max_r_indices[0]
                else:
                    return max_r_indices[np.random.randint(len(max_r_indices))]

    def update(self, selected, reward):
        self.arm_count += 1
        a = 1 / (1 + self.arm_counts[selected])
        self.arm_counts[selected] += 1
        self.arm_rewards[selected] += reward
        self.arm_values[selected] = (1 - a) * self.arm_values[selected] + a * reward


class TS(Agent):
    def __init__(self, k):
        super().__init__(k)
        self.sampling = np.ones((k, 2))

    def recet_params(self):
        super().recet_params()

    def select_arm(self):
        return np.argmax([np.random.beta(n[0], n[1], 1) for n in self.sampling])

    def update(self, selected, reward):
        self.arm_counts[selected] += 1
        self.arm_rewards[selected] += reward
        self.sampling[selected] = (self.arm_rewards[selected]+1,\
                                   self.arm_counts[selected]-self.arm_rewards[selected]+1)


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

    def select_arm(self):
        if np.amin(self.arm_counts) == 0:
            return np.argmin(self.arm_counts)
        else:
            self.f = 1
            return greedy(self.arm_values)

    def update(self, selected, reward):
        self.count += 1
        self.arm_counts[selected] += 1
        self.reward_sum += reward
        self.arm_rewards[selected] += reward
        self.arm_rewards_square[selected] += reward * reward

        # if self.count >= self.k:
        if self.f == 1:
            for i in range(0, len(self.arm_rewards)):
                ave = self.arm_rewards[i] / self.arm_counts[i]
                variance = self.arm_rewards_square[i] / self.arm_counts[i] - ave * ave
                v = variance + math.sqrt((2.0 * math.log(self.count)) / self.arm_counts[i])
                self.arm_values[i] = ave + math.sqrt((math.log(self.count) / self.arm_counts[i]) * min(0.25, v))


class meta_UCB1T(Agent):
    def __init__(self, k, l=500, delta=0.1, lmd=30):
        super().__init__(k)
        self.old_agent = UCB1T(k)
        self.new_agent = None
        self.meta_flag = 0
        self.l = l
        self.l_count = 0
        self.delta = delta
        self.lmd = lmd
        self.reward_sum = 0
        self.step = 0
        self.arm_counts = np.zeros(2)
        self.arm_rewards = np.zeros(2)
        self.arm_values = np.zeros(2)
        self.arm_rewards_square = np.zeros(2)
        self.selected = 0
        self.k = k
        self.count = 0
        self.mT_sum = 0
        self.mT_sum_array = np.array([])

    def reset_params(self):
        self.meta_flag = 0
        self.l_count = 0
        self.count = 0
        self.arm_counts = np.zeros(2)
        self.arm_rewards = np.zeros(2)
        self.arm_values = np.zeros(2)
        self.arm_rewards_square = np.zeros(2)
        self.mT_sum = 0
        self.mT_sum_array = np.zeros([])

    def select_arm(self):
        if self.meta_flag:
            if self.l_count <= 1:
                self.selected = self.l_count
            else:
                self.selected = greedy(self.arm_values)

            if self.selected == 0:
                return self.old_agent.select_arm()
            else:
                return self.new_agent.select_arm()
        else:
            return self.old_agent.select_arm()

    def update(self, selected, reward):
        self.reward_sum += reward
        self.step += 1

        if self.meta_flag:
            self.l_count += 1
            self.count += 1
            self.arm_counts[self.selected] += 1
            self.arm_rewards[self.selected] += reward
            self.arm_rewards_square[self.selected] += reward * reward

            if self.l_count >= 2:
                for i in range(0, 2):
                    ave = self.arm_rewards[i] / self.arm_counts[i]
                    variance = self.arm_rewards_square[i] / self.arm_counts[i] - ave * ave
                    v = variance + math.sqrt((2.0 * math.log(self.count)) / self.arm_counts[i])
                    self.arm_values[i] = ave + math.sqrt((math.log(self.count) / self.arm_counts[i]) * min(0.25, v))

            self.old_agent.update(selected, reward)
            self.new_agent.update(selected, reward)

            if self.l_count == self.l:
                self.reset_params()

                if greedy(self.arm_rewards) == 1:
                    self.old_agent = self.new_agent

        else:
            self.old_agent.update(selected, reward)
            # max_arm = greedy(self.old_agent.arm_rewards)
            # rt_mean =  self.old_agent.arm_rewards[max_arm] /  self.old_agent.arm_counts[max_arm]
            rt_mean = self.old_agent.reward_sum / self.old_agent.count
            mT = reward - rt_mean
            self.mT_sum += mT
            self.mT_sum_array = np.append(self.mT_sum_array, self.mT_sum)
            MT = np.amax(self.mT_sum_array)
            PHT = MT - self.mT_sum

            if PHT > self.lmd and self.step > 500:
            # if self.step % 10000 == 0:
            #     print(self.step)
                self.meta_flag = 1
                self.new_agent = UCB1T(self.k)
