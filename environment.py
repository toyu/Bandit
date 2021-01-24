import numpy as np


# バンディット問題の環境
class Bandit(object):
    # 腕の報酬確率を設定
    def __init__(self, k):
        while True:
            self.probability = np.asarray(np.random.rand(k))
            if len(self.probability) == len(set(self.probability)):
                break

        self.correct_arm = np.argmax(self.probability)
        self.max_prob_arm = np.amax(self.probability)
        sorted_prob = np.sort(self.probability)[::-1]
        self.opt_r = (sorted_prob[0] + sorted_prob[1]) / 2

    # 報酬を返す
    def get_reward(self, selected):
        if self.probability[selected] >= np.random.rand():
            return 1
        else:
            return 0

    # 正解の腕を選んだか判定する
    def get_correct(self, selected):
        if self.correct_arm == selected:
            return 1
        else:
            return 0

    # regret を返す
    def get_regret(self, selected):
        return self.max_prob_arm - self.probability[selected]
