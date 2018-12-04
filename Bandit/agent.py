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
    def __init__(self, bandit, k):
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
    def __init__(self, bandit, k):
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
    def __init__(self, bandit, k):
        super().__init__(k)
        self.arm_rewards_square = np.zeros(k)
        self.k = k
        self.count = 0

    def reset_params(self):
        super().recet_params()

    def select_arm(self):
        if self.count < self.k:
            return self.count
        else:
            return greedy(self.arm_values)

    def update(self, selected, reward):
        self.count += 1
        self.arm_counts[selected] += 1
        self.arm_rewards[selected] += reward
        self.arm_rewards_square[selected] += reward * reward

        if self.count >= self.k:
            for i in range(0, len(self.arm_rewards)):
                ave = self.arm_rewards[i] / self.arm_counts[i]
                variance = self.arm_rewards_square[i] / self.arm_counts[i] - ave * ave
                v = variance + math.sqrt((2.0 * math.log(self.count)) / self.arm_counts[i])
                self.arm_values[i] = ave + math.sqrt((math.log(self.count) / self.arm_counts[i]) * min(0.25, v))