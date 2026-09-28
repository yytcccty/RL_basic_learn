import numpy as np
import matplotlib.pyplot as plt
import gym
from scipy.stats import truncnorm
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CEM:
    def __init__(self, n_sequence, elite_ratio, fake_env, upper_bound, lower_bound):
        self.n_sequence = n_sequence
        self.elite_ratio = elite_ratio
        self.fake_env = fake_env
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound

    def optimize(self, state, init_mean, init_var):
        mean, var = init_mean, init_var
        X = truncnorm(-2, 2, loc=np.ones_like(mean), scale=np.zeros_like(mean))
        state = np.tile(state, (self.n_sequence, 1))
        for _ in range(5):
            lb_dist = mean - self.lower_bound
            ub_dist = self.upper_bound - mean
            constrained_var = np.minimum(np.minimum(np.square(ub_dist / 2), np.square(lb_dist)), var)
            action_sequences = [X.rvs() for _ in range(self.n_sequence)] * np.sqrt(constrained_var) + mean
            returns = self.fake_env.propogate(state, action_sequences)
            elites = action_sequences[np.argsort(returns)][-int(self.elite_ratio * self.n_sequence):]
            new_mean = np.mean(elites, axis=0)
            new_var = np.var(elites, axis=0)
            mean = 0.1 * mean + 0.9 * new_mean
            var = 0.1 * var + 0.1 * new_var

        return mean


class EnsembleModel(nn.Module):
    """Environment Ensemble Model"""

    def __init__(self, state_dim, action_dim, ensemble_size, lr=1e-3):
        super(EnsembleModel, self).__init__()
        self._output_dim = (state_dim + 1) * 2
        self._max_logvar = nn.Parameter((torch.ones(1, self._output_dim // 2).float() / 2).to(device), requires_grad=False)
        self._min_logvar = nn.Parameter((-torch.ones(1, self._output_dim // 2).float() * 10).to(device), requires_grad=False)


if __name__ == '__main__':
    pass
