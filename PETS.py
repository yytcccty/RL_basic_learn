import numpy as np
import matplotlib.pyplot as plt
import gym
from scipy.stats import truncnorm
import torch
import torch.nn as nn
import torch.nn.functional as F
import collections

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ReplayerBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def size(self):
        return len(self.buffer)

    def return_all_samples(self):
        all_eles = list(self.buffer)
        return zip(*all_eles)


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


class FCLayer(nn.Module):
    """Ensemble_size independent linear networks"""

    def __init__(self, input_dim, output_dim, ensemble_size, activation):
        super(FCLayer, self).__init__()
        self.inpit_dim = input_dim
        self._output_dim = output_dim
        self.activation = activation
        self.weights = nn.Parameter(torch.Tensor(ensemble_size, input_dim, output_dim)).to(device)
        self.bias = nn.Parameter(torch.Tensor(ensemble_size, output_dim)).to(device)

    def forward(self, x):
        return self.activation(torch.add(torch.bmm(x, self.weights), self.bias.unsqueeze(1)))


def init_weights(m):
    """Initialize weights of FCLayer using truncated normal distribution
       Set bias of FCLayer to zero"""
    if isinstance(m, FCLayer):
        std = 1 / (2 * np.sqrt(m.inpit_dim))
        nn.init.trunc_normal_(m.weights, std=std, a=-2 * std, b=2 * std)
        nn.init.zeros_(m.bias)


class EnsembleModel(nn.Module):
    """Environment Ensemble Model"""

    def __init__(self, state_dim, action_dim, ensemble_size, lr=1e-3):
        super(EnsembleModel, self).__init__()
        self._output_dim = (state_dim + 1) * 2
        self._max_logvar = nn.Parameter((torch.ones(1, self._output_dim // 2).float() / 2).to(device),
                                        requires_grad=False)
        self._min_logvar = nn.Parameter((-torch.ones(1, self._output_dim // 2).float() * 10).to(device),
                                        requires_grad=False)
        self.layer1 = FCLayer(state_dim + action_dim, 200, ensemble_size, nn.SiLU())
        self.layer2 = FCLayer(200, 200, ensemble_size, nn.SiLU())
        self.layer3 = FCLayer(200, 200, ensemble_size, nn.SiLU())
        self.layer4 = FCLayer(200, 200, ensemble_size, nn.SiLU())
        self.layer5 = FCLayer(200, self._output_dim, ensemble_size, nn.Identity())
        self.apply(init_weights)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)


class PETS:
    def __init__(self, env, replay_buffer, num_episodes):
        self._env = env
        self._env_pool = replay_buffer
        self.num_episodes = num_episodes

    def explore(self):
        obs, _ = self._env.reset()
        done = False
        episode_retn = 0
        while not done:
            action = self._env.action_space.sample()
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            self._env_pool.add(obs, action, reward, next_obs, done)
            obs = next_obs
            episode_retn += reward
        return episode_retn


    def train(self):
        retn_list = []
        explore_retn = self.explore()
        print(f'Episode 1 retun: {explore_retn}')
        retn_list.append(explore_retn)
        for i in range(self.num_episodes -1):


        return retn_list


if __name__ == '__main__':
    """
    Hyperparameter Settings
    """
    buffer_size = 1e5
    n_sequences = 50
    elite_ratio = 0.2
    num_episodes = 10

    env_name = "Pendulum-v1"
    """
    Coding
    """
    env = gym.make(env_name)
    replay_buffer = ReplayerBuffer(capacity=buffer_size)
    pets = PETS(env, replay_buffer, num_episodes)
    retn_list = pets.train()

    plt.plot(retn_list)
    plt.xlabel("Episodes")
    plt.ylabel("Reward Sum")
    plt.show()

    pass
