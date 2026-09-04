import numpy as np
import numpy.random as random
import matplotlib.pyplot as plt


class MAB_Bernoulli:
    def __init__(self, K):
        self.K = K
        self.rng = random.default_rng(seed=42)

        self.probs = self.rng.uniform(size=K)
        self.best_id = np.argmax(self.probs)
        self.best_prob = self.probs[self.best_id]

    def step(self, k):
        if random.random() < self.probs[k]:
            return 1
        return 0


class Solver:
    def __init__(self, machine):
        self.machine = machine
        self.counts = np.zeros(machine.K)
        self.regret = 0
        self.actions = []
        self.regrets = []

    def run_one_step(self):  # Policy
        raise NotImplementedError

    def update_regret(self, k):
        self.regret += self.machine.best_prob - self.machine.probs[k]
        self.regrets.append(self.regret)

    def run(self, timestep):
        for _ in range(timestep):
            k = self.run_one_step()
            self.counts[k] += 1
            self.actions.append(k)
            self.update_regret(k)


class EpsilonGreedy(Solver):  # Adopt Fixed Epsilon-Greedy Policy
    def __init__(self, machine, eps=0.01, init_prob=1.0):
        super().__init__(machine)
        self.eps = eps
        self.estimates = np.array([init_prob] * self.machine.K)

    def run_one_step(self):
        if random.random() < self.eps:
            k = random.randint(self.machine.K)
        else:
            k = np.argmax(self.estimates)
        r = self.machine.step(k)
        self.estimates[k] += (r - self.estimates[k]) / (self.counts[k] + 1)
        return k


class DecayEpsilonGreedy(Solver):  # Adopt Time-varied Epsilon Greedy Policy
    def __init__(self, machine, init_prob=1.0):
        super().__init__(machine)
        self.timestamp = 0
        self.estimates = np.array([init_prob] * self.machine.K)

    def run_one_step(self):
        self.timestamp += 1
        if random.random() < 1 / self.timestamp:
            k = random.randint(self.machine.K)
        else:
            k = np.argmax(self.estimates)
        r = self.machine.step(k)
        self.estimates[k] += (r - self.estimates[k]) / (self.counts[k] + 1)
        return k


class UCB(Solver):  # Adopt UCB policy
    def __init__(self, machine, init_prob=1.0, c=1.0):
        super().__init__(machine)
        self.estimates = np.array([init_prob] * self.machine.K)
        self.timestamp = 0
        self.c = c

    def run_one_step(self):
        self.timestamp += 1
        UCB_list = self.estimates + self.c * np.sqrt(np.log(self.c) / (2 * self.counts + 1))
        k = np.argmax(UCB_list)
        r = self.machine.step(k)
        self.estimates[k] += (r - self.estimates[k]) / (self.counts[k] + 1)
        return k


class ThompsonSampling(Solver):  # Adopt ThompsonSampling Policy
    def __init__(self, machine):
        super().__init__(machine)
        self._a = np.ones(self.machine.K)
        self._b = np.ones(self.machine.K)

    def run_one_step(self):
        samples = random.beta(self._a, self._b)
        k = np.argmax(samples)
        r = self.machine.step(k)
        self._a[k] += r
        self._b[k] += 1 - r
        return k


def plot_results(solvers, names):
    for idx, sovler in enumerate(solvers):
        step_lis = range(len(sovler.regrets))
        plt.plot(step_lis, sovler.regrets, label=names[idx])
    plt.xlabel('Time step')
    plt.ylabel('Cumulative regrets')
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == '__main__':
    K = 10  # Number of Bandit Arms
    attempts = 5000  # Number of the Times of Pulling Arms
    Eps = [1e-4, 0.01, 0.25, 0.5]  # Epsilon Value(s) for Epsilon-Greedy Algorithm
    c = 1.0  # Coefficient Value for UCB Algorithm

    machine = MAB_Bernoulli(K)
    print(
        f'\nThe highest winning probability is \033[1;33m#{machine.best_id}\033[0m with the probability of \033[1;32m{machine.best_prob}\033[0m')

    mySolver_list = [EpsilonGreedy(machine, eps=ep) for ep in Eps] + [DecayEpsilonGreedy(machine)] + [
        UCB(machine, c)] + [ThompsonSampling(machine)]
    mySolver_name_list = [f'Fixed Epsilon={ep}' for ep in Eps] + ['Decay Epsilon'] + [f'UCB c={c}'] + [
        'Thompson Sampling']

    # mySolver_list = [DecayEpsilonGreedy(machine)]
    # mySolver_name_list = ['Decay Epsilon']

    for mySolver in mySolver_list:
        mySolver.run(attempts)

    index = np.argmin([ele.regret for ele in mySolver_list])
    print(f'\033[1;34m{mySolver_name_list[index]}\033[0m achieves the lowest cumulative regret')
    plot_results(mySolver_list, mySolver_name_list)
