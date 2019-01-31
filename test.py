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