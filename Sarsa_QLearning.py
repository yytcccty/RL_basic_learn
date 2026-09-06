import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


class CliffWalkingEnv:
    def __init__(self, ncol, nrow):
        """
        :param ncol: Number of columns in the grid
        :param nrow: Number of rows in the grid
        (self.x_cor, self.y_cor): Current state s=(x, y), coordinate of the agent
        Origin point (0,0) is located in the top left corner
        Positive x direction is to the right
        Positive y direction is to the downwards
        """
        self.ncol = ncol
        self.nrow = nrow
        self.x_cor = 0
        self.y_cor = nrow - 1

    def interact(self, action):
        """
        :param action: Given action
        :return: (s_next, reward, done)
                s_next: Next state
                reward: Reward of (s_cur, action)
                done: Whether the episode has ended
        """
        change = [[0, -1], [0, 1], [-1, 0], [1, 0]]
        self.x_cor = min(max(self.x_cor + change[action][0], 0), self.ncol - 1)
        self.y_cor = min(max(self.y_cor + change[action][1], 0), self.nrow - 1)
        s_next = self.y_cor * self.ncol + self.x_cor
        reward = -1
        done = False
        if self.y_cor == self.nrow - 1 and self.x_cor != 0:
            done = True
            if self.x_cor != self.ncol - 1:
                reward = -100
        return s_next, reward, done

    def reset(self):
        self.x_cor = 0
        self.y_cor = nrow - 1
        return self.y_cor * self.ncol + self.x_cor


class Sarsa:
    def __init__(self, n_steps, alpha, gamma, eps, ncol, nrow, n_act=4):
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.n_act = n_act
        self.Q = np.zeros((nrow * ncol, n_act))
        self.n_step = n_step

        self.r_list = []
        self.a_list = []
        self.s_list = []
        self.rng = np.random.default_rng(seed=41)

    def perform_action(self, s_cur):
        """
        :param s_cur: Current state
        :return: Action
        """
        if self.rng.random() < self.eps:
            return self.rng.integers(self.n_act)
        else:
            return np.argmax(self.Q[s_cur])

    def update(self, s_cur, a_cur, r, s_next, a_next, done):
        """
        :param s_cur: Current state
        :param a_cur: Current action
        :param r: Reward r=f(s_cur, a_cur, s_next)
        :param s_next: The next state
        :param a_next: The next action
        :param done: Whether the episode has ended
        Update Q_sa using n-step Temporal Difference Formulation
        """
        self.s_list.append(s_cur)
        self.r_list.append(r)
        self.a_list.append(a_cur)
        if len(self.s_list) == self.n_step:
            tmp_G = self.Q[s_next, a_next]
            for i in reversed(range(self.n_step)):
                tmp_G = self.gamma * tmp_G + self.r_list[i]
                if done and i > 0:
                    s = self.s_list[i]
                    a = self.a_list[i]
                    self.Q[s, a] += self.alpha * (tmp_G - self.Q[s, a])
            s = self.s_list.pop(0)
            a = self.a_list.pop(0)
            self.r_list.pop(0)
            self.Q[s, a] += self.alpha * (tmp_G - self.Q[s, a])

        if done:
            self.s_list = []
            self.r_list = []
            self.a_list = []


class Q_Learning:
    def __init__(self, alpha, gamma, eps, ncol, nrow, n_act=4):
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.n_act = n_act
        self.Q = np.zeros((nrow * ncol, n_act))

        self.rng = np.random.default_rng(seed=91)

    def perform_action(self, s_cur):
        """
        :param s_cur: Current state
        :return: Action
        """
        if self.rng.random() < self.eps:
            return self.rng.integers(self.n_act)
        else:
            return np.argmax(self.Q[s_cur])

    def update(self, s_cur, a_cur, r, s_next):
        self.Q[s_cur, a_cur] += self.alpha * (r + self.gamma * np.max(self.Q[s_next, :]) - self.Q[s_cur, a_cur])


def printPolicy(agent, env, disaster=None, end=None):
    action_meaning = ['^', 'v', '<', '>']
    for i in range(env.nrow):
        for j in range(env.ncol):
            tmp = i * env.ncol + j
            if tmp in disaster:
                print('****', end=' ')
            elif tmp in end:
                print('EEEE', end=' ')
            else:
                tmp_str = ''
                tmp_q_s = agent.Q[tmp, :]
                tmp_maxn = np.max(tmp_q_s)
                for k in range(4):
                    tmp_str += 'o' if tmp_q_s[k] != tmp_maxn else action_meaning[k]
                print(tmp_str, end=' ')
        print()


if __name__ == '__main__':
    """
    Hyperparameter settings
    """
    ncol, nrow = 12, 4
    alpha = 0.1
    gamma = 0.9
    eps = 0.1
    episodes = 500
    n_step = 5
    alg_name = 'QLearning'  # ['Sarsa', 'QLearning']
    """
    Coding
    """
    env = CliffWalkingEnv(ncol, nrow)


    retn_list = []
    if alg_name == 'Sarsa':
        agent = Sarsa(n_step, alpha, gamma, eps, ncol, nrow)
        for cnt in tqdm(range(episodes), desc=f'Episodes'):
            s_cur = env.reset()
            a_cur = agent.perform_action(s_cur)
            tmp_retn = 0.
            while True:
                (s_next, reward, done) = env.interact(a_cur)
                a_next = agent.perform_action(s_next)
                agent.update(s_cur, a_cur, reward, s_next, a_next, done)
                tmp_retn += reward  # No discount here
                s_cur = s_next
                a_cur = a_next
                if done is True:
                    break
            retn_list.append(tmp_retn)

    elif alg_name == 'QLearning':
        agent = Q_Learning(alpha, gamma, eps, ncol, nrow)
        for cnt in tqdm(range(episodes), desc=f'Episodes'):
            s_cur = env.reset()
            tmp_retn = 0.
            while True:
                a_cur = agent.perform_action(s_cur)
                (s_next, reward, done) = env.interact(a_cur)
                agent.update(s_cur, a_cur, reward, s_next)
                tmp_retn += reward  # No discount here
                s_cur = s_next
                if done is True:
                    break
            retn_list.append(tmp_retn)

    print(f'{alg_name} policy result: ')
    printPolicy(agent, env, disaster=range(37, 47), end=[47])

    plt.plot(retn_list),
    plt.xlabel('Episode')
    plt.ylabel('Reward Sum')
    plt.show()

    pass
