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
        return list(zip(*self.buffer))


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


class EnsembleDynamicModel:
    def __init__(self, state_dim, action_dim, num_network=5):
        self.model = EnsembleModel(state_dim, action_dim, ensemble_size=num_network)

    def train(self, inputs, labels, batch_size=64, mat_iter=20, holdout_ratio=0.1):
        permutn = np.random.permutation(inputs.shape[0])
        inputs, labels = inputs[permutn], labels[permutn]
        num_holdout = int(inputs.shape[0] * holdout_ratio)
        train_inputs, train_labels = inputs[num_holdout:], labels[num_holdout:]  # training set
        holdout_inputs, holdout_labels = inputs[:num_holdout], labels[:num_holdout]  # validation set

        holdout_inputs = torch.from_numpy(holdout_inputs).float().to(device)
        holdout_labels = torch.from_numpy(holdout_labels).float().to(device)



class PETS:
    def __init__(self, env, replay_buffer, num_episodes):
        self._env = env
        self._env_pool = replay_buffer
        self.num_episodes = num_episodes
        obs_dim = env.observation_space.shape[0]
        self._action_dim = env.action_space.shape[0]
        self._model = EnsembleDynamicModel(obs_dim, self._action_dim)

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

    def train_model(self):
        env_samples = self._env_pool.return_all_samples()
        obs = np.array(env_samples[0])
        actions = np.array(env_samples[1])
        rewards = np.array(env_samples[2]).reshape(-1, 1)
        next_obs = np.array(env_samples[3])
        next_obs = np.array(env_samples[3])
        dones = np.array(env_samples[4])
        inputs = np.concatenate((obs, actions), axis=1)
        labels = np.concatenate((rewards, next_obs), axis=1)
        self._model.train(inputs, labels)

    def train(self):
        retn_list = []
        explore_retn = self.explore()
        print(f'Episode 1 return: {explore_retn}')
        retn_list.append(explore_retn)
        for i in range(self.num_episodes - 1):
            self.train_model()
            episode_retn = self.mpc()
            retn_list.append(episode_retn)
            print(f'Episode {i + 2} return: {episode_retn}')

        return retn_list


if __name__ == '__main__':
    """
    Hyperparameter Settings
    """
    buffer_size = 100000
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
