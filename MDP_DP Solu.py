import copy
from math import gamma
import numpy as np


class CliffWalkingEnv:
    def __init__(self, nrow=4, ncol=12):
        """
        :param nrow: Number of rows of grid map
        :param ncol: Number of columns of grid map
        """
        self.ncol = ncol
        self.nrow = nrow
        self.P = self.createP()

    def createP(self):
        """
        Origin point locates in the up-left corner of the map, whose positive direction of x-axis is to the right and
        positive direction of y-axis is to the downward
        """
        P = [[[] for _ in range(4)] for _ in range(self.nrow * self.ncol)]
        change = [[0, -1], [0, 1], [-1, 0], [1, 0]]  # Coordinate change of actions :up, down, left, right
        for i in range(self.nrow):
            for j in range(self.ncol):
                for a in range(4):
                    # Process the situation when agent locates in (j, i) and chooses action 'a'
                    if i == self.nrow - 1 and j > 0:
                        P[i * self.ncol + j][a].append([1, i * self.nrow + j, 0])
                        continue
                    next_x = max(min(j + change[a][0], self.ncol - 1), 0)
                    next_y = max(min(i + change[a][1], self.nrow - 1), 0)
                    next_state = next_y * self.ncol + next_x
                    reward = -1
                    if next_y == self.nrow - 1 and next_x > 0 and next_x != self.ncol - 1:
                        reward = -100
                    P[i * self.ncol + j][a].append([1, next_state, reward])
        return P


class PolicyIter:
    def __init__(self, env, gamma, thre):
        self.env = env
        self.gamma = gamma
        self.thre = thre
        self.tot = self.env.ncol * self.env.nrow
        self.pi = [[0.25, 0.25, 0.25, 0.25] for _ in range(self.tot)]
        self.V = [0] * self.tot

    def policy_eval(self):
        cnt = 1
        while True:
            delta = 0.
            tmp_V = [0] * self.tot
            for s in range(self.tot):
                tmp_sum_1 = 0.
                for a in range(4):
                    tmp_sum_2 = np.sum([p * self.V[next_state] for (p, next_state, _) in self.env.P[s][a]])
                    r = self.env.P[s][a][0][2]
                    tmp_sum_1 += (r + self.gamma * tmp_sum_2) * self.pi[s][a]
                tmp_V[s] = tmp_sum_1
                delta = max(delta, abs(tmp_V[s] - self.V[s]))
            self.V = tmp_V
            if delta <= self.thre:
                break
            cnt += 1
        print(f'Policy finished in {cnt} iterations')

    def policy_impro(self):
        for s in range(self.tot):
            qsa_list = []
            for a in range(4):
                tmp = np.sum([p * self.V[next_state] for (p, next_state, _) in self.env.P[s][a]])
                r = self.env.P[s][a][0][2]
                qsa_list.append(r + self.gamma * tmp)
            maxn = max(qsa_list)
            tmp = qsa_list.count(maxn)
            self.pi[s] = [1 / tmp if q == maxn else 0 for q in qsa_list]

    def policy_iter(self):
        while True:
            pi_old = copy.deepcopy(self.pi)
            self.policy_eval()
            self.policy_impro()
            if pi_old == self.pi:
                break


if __name__ == '__main__':
    """
    Hyperparameter Settings
    """
    theta = 1e-3
    gamma = 0.9
    """
    Coding
    """
    env = CliffWalkingEnv()
    agent = PolicyIter(env, gamma, theta)
    agent.policy_iter()
    pass
