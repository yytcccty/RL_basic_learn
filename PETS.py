import numpy as np

if not hasattr(np, "bool8"):
    np.bool8 = np.bool_
import matplotlib.pyplot as plt
import gym
from scipy.stats import truncnorm
import torch
import torch.nn as nn
import torch.nn.functional as F
import collections
import itertools

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ReplayBuffer:
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
        X = truncnorm(-2, 2, loc=np.zeros_like(mean), scale=np.ones_like(var))
        state = np.tile(state, (self.n_sequence, 1))
        for _ in range(5):
            lb_dist = mean - self.lower_bound
            ub_dist = self.upper_bound - mean
            constrained_var = np.minimum(np.minimum(np.square(ub_dist / 2), np.square(lb_dist / 2)), var)
            action_sequences = [X.rvs() for _ in range(self.n_sequence)] * np.sqrt(constrained_var) + mean
            returns = self.fake_env.propagate(state, action_sequences)[:, 0]
            elites = action_sequences[np.argsort(returns)][-int(self.elite_ratio * self.n_sequence):]
            new_mean = np.mean(elites, axis=0)
            new_var = np.var(elites, axis=0)
            mean = 0.1 * mean + 0.9 * new_mean
            var = 0.1 * var + 0.9 * new_var

        return mean


class FCLayer(nn.Module):
    """Ensemble_size independent linear networks"""

    def __init__(self, input_dim, output_dim, ensemble_size, activation):
        super(FCLayer, self).__init__()
        self.input_dim = input_dim
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
        std = 1 / (2 * np.sqrt(m.input_dim))
        nn.init.trunc_normal_(m.weights, std=std, a=-2 * std, b=2 * std)
        nn.init.zeros_(m.bias)
        # m.bias.data.fill_(0.0)


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

    def forward(self, x, return_logvar=False):
        ret = self.layer5(self.layer4(self.layer3(self.layer2(self.layer1(x)))))
        mean = ret[:, :, :self._output_dim // 2]
        logvar = self._max_logvar - F.softplus(self._max_logvar - ret[:, :, self._output_dim // 2:])
        logvar = self._min_logvar + F.softplus(logvar - self._min_logvar)
        return mean, logvar if return_logvar else torch.exp(logvar)

    def loss(self, mean, logvar, labels, use_var_loss=True):
        inverse_var = torch.exp(-logvar)
        if use_var_loss:
            mse_loss = torch.mean(torch.pow(mean - labels, 2) * inverse_var, dim=(1, 2))
            var_loss = torch.mean(logvar, dim=(1, 2))
            total_loss = torch.sum(mse_loss + var_loss)
        else:
            mse_loss = torch.mean(torch.pow(mean - labels, 2), dim=(1, 2))
            total_loss = torch.sum(mse_loss)
        return total_loss, mse_loss

    def train(self, loss):
        self.optimizer.zero_grad()
        loss += 0.01 * torch.sum(self._max_logvar) - 0.01 * torch.sum(
            self._min_logvar)
        loss.backward()
        self.optimizer.step()


class EnsembleDynamicsModel:
    def __init__(self, state_dim, action_dim, num_network=5):
        self.model = EnsembleModel(state_dim, action_dim, ensemble_size=num_network)
        self._num_network = num_network
        self._epoch_since_last_update = 0

    def _save_best(self, loss, threshold=0.1):
        updated = False
        for i in range(len(loss)):
            if self._snapshots[i] - loss[i] > threshold * self._snapshots[i]:
                self._snapshots[i] = loss[i]
                updated = True
        self._epoch_since_last_update = 0 if updated else self._epoch_since_last_update + 1
        return self._epoch_since_last_update > 5

    def train(self, inputs, labels, batch_size=64, max_iter=20, holdout_ratio=0.1):
        permutn = np.random.permutation(inputs.shape[0])
        inputs, labels = inputs[permutn], labels[permutn]
        num_holdout = int(inputs.shape[0] * holdout_ratio)
        train_inputs, train_labels = inputs[num_holdout:], labels[num_holdout:]  # training set
        holdout_inputs, holdout_labels = inputs[:num_holdout], labels[:num_holdout]  # validation set

        holdout_inputs = torch.from_numpy(holdout_inputs).float().to(device)
        holdout_labels = torch.from_numpy(holdout_labels).float().to(device)
        holdout_inputs = holdout_inputs[None, :, :].repeat([self._num_network, 1, 1])
        holdout_labels = holdout_labels[None, :, :].repeat([self._num_network, 1, 1])

        self._snapshots = [1e10] * self._num_network

        for epoch in itertools.count():
            train_index = np.vstack([np.random.permutation(train_inputs.shape[0]) for _ in range(self._num_network)])

            for batch_start_pos in range(0, train_inputs.shape[0], batch_size):
                batch_index = train_index[:, batch_start_pos:batch_start_pos + batch_size]
                train_input = torch.from_numpy(train_inputs[batch_index]).float().to(device)
                train_label = torch.from_numpy(train_labels[batch_index]).float().to(device)

                mean, logvar = self.model(train_input, return_logvar=True)
                loss, _ = self.model.loss(mean, logvar, train_label)
                self.model.train(loss)

            with torch.no_grad():
                mean, logvar = self.model(holdout_inputs, return_logvar=True)
                _, holdout_loss = self.model.loss(mean, logvar, holdout_labels, use_var_loss=False)
                holdout_loss = holdout_loss.cpu()
                break_condition = self._save_best(holdout_loss)
                if break_condition or epoch > max_iter:
                    break

    def predict(self, inputs, batch_size=64):
        mean, var = [], []
        for i in range(0, inputs.shape[0], batch_size):
            batch_input = torch.from_numpy(inputs[i:i + batch_size]).float().to(device)
            batch_input = batch_input[None, :, :].repeat([self._num_network, 1, 1])
            cur_mean, cur_var = self.model(batch_input, return_logvar=False)
            mean.append(cur_mean.cpu().detach().numpy())
            var.append(cur_var.cpu().detach().numpy())
        return np.hstack(mean), np.hstack(var)


class FakeEnv:
    def __init__(self, model):
        self.model = model

    def step(self, obs, action):
        inputs = np.concatenate((obs, action), axis=-1)
        ensemble_model_means, ensemble_model_vars = self.model.predict(inputs)
        ensemble_model_means[:, :, 1:] += obs.numpy()
        ensemble_model_std = np.sqrt(ensemble_model_vars)
        ensemble_samples = np.random.normal(size=ensemble_model_means.shape) * ensemble_model_std + ensemble_model_means

        num_models, batch_size, _ = ensemble_model_means.shape
        models_to_use = np.random.choice(num_models, size=batch_size)
        samples = ensemble_samples[models_to_use, np.arange(0, batch_size)]
        rewards = samples[:, :1]
        next_obs = samples[:, 1:]
        return rewards, next_obs

    def propagate(self, obs, actions):
        with torch.no_grad():
            obs = np.copy(obs)
            total_reward = np.expand_dims(np.zeros(obs.shape[0]), axis=-1)
            obs = torch.as_tensor(obs)
            actions = torch.as_tensor(actions)
            for i in range(actions.shape[1]):
                action = actions[:, i].unsqueeze(-1)
                rewards, next_obs = self.step(obs, action)
                total_reward += rewards
                obs = torch.as_tensor(next_obs)
            return total_reward


class PETS:
    def __init__(self, env, replay_buffer, num_episodes, plan_horizon, n_sequence, elite_ratio, ):
        self._env = env
        self._env_pool = replay_buffer
        self.num_episodes = num_episodes
        obs_dim = env.observation_space.shape[0]
        self._action_dim = env.action_space.shape[0]
        self._model = EnsembleDynamicsModel(obs_dim, self._action_dim)
        self._fake_env = FakeEnv(self._model)
        self.plan_horizon = plan_horizon
        self.upper_bound = env.action_space.high[0]
        self.lower_bound = env.action_space.low[0]
        self._cem = CEM(n_sequence, elite_ratio, self._fake_env, self.upper_bound, self.lower_bound)

    def explore(self):
        obs, _ = self._env.reset()
        done = False
        episode_retn = 0
        while not done:
            action = self._env.action_space.sample()
            next_obs, reward, terminated, truncated, info = self._env.step(action)
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
        inputs = np.concatenate((obs, actions), axis=1)
        labels = np.concatenate((rewards, next_obs - obs), axis=1)
        self._model.train(inputs, labels)

    def mpc(self):
        mean = np.tile((self.upper_bound + self.lower_bound) / 2, self.plan_horizon)
        var = np.tile((self.upper_bound - self.lower_bound) ** 2 / 16, self.plan_horizon)
        obs, _ = self._env.reset()
        done = False
        episode_retn = 0

        while not done:
            actions = self._cem.optimize(obs, mean, var)
            action = actions[:self._action_dim]
            next_obs, reward, terminated, truncated, info = self._env.step(action)
            done = terminated or truncated
            self._env_pool.add(obs, action, reward, next_obs, done)
            obs = next_obs
            episode_retn += reward
            mean = np.concatenate((np.copy(actions)[self._action_dim:],
                                   np.zeros(self._action_dim)))  # Sliding window for warm-up
        return episode_retn

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
    plan_horizon = 25
    env_name = "Pendulum-v1"
    """
    Coding
    """
    env = gym.make(env_name)
    replay_buffer = ReplayBuffer(capacity=buffer_size)
    pets = PETS(env, replay_buffer, num_episodes, plan_horizon, n_sequences, elite_ratio)
    retn_list = pets.train()

    plt.plot(retn_list)
    plt.xlabel("Episodes")
    plt.ylabel("Reward Sum")
    plt.show()

    pass













import numpy as np
from scipy.stats import truncnorm
import gym
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
import collections
import matplotlib.pyplot as plt

if not hasattr(np, "bool8"):
    np.bool8 = np.bool_


class CEM:
    def __init__(self, n_sequence, elite_ratio, fake_env, upper_bound,
                 lower_bound):
        self.n_sequence = n_sequence
        self.elite_ratio = elite_ratio
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound
        self.fake_env = fake_env

    def optimize(self, state, init_mean, init_var):
        mean, var = init_mean, init_var
        X = truncnorm(-2, 2, loc=np.zeros_like(mean), scale=np.ones_like(var))
        state = np.tile(state, (self.n_sequence, 1))
        for _ in range(5):
            lb_dist, ub_dist = mean - self.lower_bound, self.upper_bound - mean
            constrained_var = np.minimum(np.minimum(np.square(lb_dist / 2), np.square(ub_dist / 2)), var)
            action_sequences = [X.rvs() for _ in range(self.n_sequence)] * np.sqrt(constrained_var) + mean
            returns = self.fake_env.propagate(state, action_sequences)[:, 0]
            elites = action_sequences[np.argsort(returns)][-int(self.elite_ratio * self.n_sequence):]
            new_mean = np.mean(elites, axis=0)
            new_var = np.var(elites, axis=0)
            mean = 0.1 * mean + 0.9 * new_mean
            var = 0.1 * var + 0.9 * new_var

        return mean


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def init_weights(m):
    ''' 初始化模型权重 '''
    if isinstance(m, FCLayer):
        std = 1 / (2 * np.sqrt(m._input_dim))
        nn.init.trunc_normal_(m.weight, std=std, a=-2 * std, b=2 * std)
        nn.init.zeros_(m.bias)
        # m.bias.data.fill_(0.0)


class FCLayer(nn.Module):
    ''' 集成之后的全连接层 '''

    def __init__(self, input_dim, output_dim, ensemble_size, activation):
        super(FCLayer, self).__init__()
        self._input_dim, self._output_dim = input_dim, output_dim
        self.weight = nn.Parameter(torch.Tensor(ensemble_size, input_dim, output_dim).to(device))
        self.activation = activation
        self.bias = nn.Parameter(torch.Tensor(ensemble_size, output_dim).to(device))

    def forward(self, x):
        return self.activation(torch.add(torch.bmm(x, self.weight), self.bias.unsqueeze(1)))


class EnsembleModel(nn.Module):
    ''' 环境模型集成 '''

    def __init__(self, state_dim, action_dim, ensemble_size=5, learning_rate=1e-3):
        super(EnsembleModel, self).__init__()
        self._output_dim = (state_dim + 1) * 2
        self._max_logvar = nn.Parameter((torch.ones((1, self._output_dim // 2)).float() / 2).to(device),
                                        requires_grad=False)
        self._min_logvar = nn.Parameter((-torch.ones((1, self._output_dim // 2)).float() * 10).to(device),
                                        requires_grad=False)
        self.layer1 = FCLayer(state_dim + action_dim, 200, ensemble_size, nn.SiLU())
        self.layer2 = FCLayer(200, 200, ensemble_size, nn.SiLU())
        self.layer3 = FCLayer(200, 200, ensemble_size, nn.SiLU())
        self.layer4 = FCLayer(200, 200, ensemble_size, nn.SiLU())
        self.layer5 = FCLayer(200, self._output_dim, ensemble_size, nn.Identity())
        self.apply(init_weights)  # 初始化环境模型中的参数
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x, return_log_var=False):
        ret = self.layer5(self.layer4(self.layer3(self.layer2(self.layer1(x)))))
        mean = ret[:, :, :self._output_dim // 2]
        logvar = self._max_logvar - F.softplus(self._max_logvar - ret[:, :, self._output_dim // 2:])
        logvar = self._min_logvar + F.softplus(logvar - self._min_logvar)
        return mean, logvar if return_log_var else torch.exp(logvar)

    def loss(self, mean, logvar, labels, use_var_loss=True):
        inverse_var = torch.exp(-logvar)
        if use_var_loss:
            mse_loss = torch.mean(torch.pow(mean - labels, 2) * inverse_var, dim=(1, 2))
            var_loss = torch.mean(logvar, dim=(1, 2))
            total_loss = torch.sum(mse_loss + var_loss)
        else:
            mse_loss = torch.mean(torch.pow(mean - labels, 2), dim=(1, 2))
            total_loss = torch.sum(mse_loss)
        return total_loss, mse_loss

    def train(self, loss):
        self.optimizer.zero_grad()
        loss += 0.01 * torch.sum(self._max_logvar) - 0.01 * torch.sum(
            self._min_logvar)
        loss.backward()
        self.optimizer.step()


class EnsembleDynamicsModel:
    ''' 环境模型集成,加入精细化的训练 '''

    def __init__(self, state_dim, action_dim, num_network=5):
        self._num_network = num_network
        self.model = EnsembleModel(state_dim, action_dim, ensemble_size=num_network)
        self._epoch_since_last_update = 0

    def train(self, inputs, labels, batch_size=64, holdout_ratio=0.1, max_iter=20):
        # 设置训练集与验证集
        permutation = np.random.permutation(inputs.shape[0])
        inputs, labels = inputs[permutation], labels[permutation]
        num_holdout = int(inputs.shape[0] * holdout_ratio)
        train_inputs, train_labels = inputs[num_holdout:], labels[num_holdout:]
        holdout_inputs, holdout_labels = inputs[:num_holdout], labels[:num_holdout]

        holdout_inputs = torch.from_numpy(holdout_inputs).float().to(device)
        holdout_labels = torch.from_numpy(holdout_labels).float().to(device)
        holdout_inputs = holdout_inputs[None, :, :].repeat([self._num_network, 1, 1])
        holdout_labels = holdout_labels[None, :, :].repeat([self._num_network, 1, 1])

        self._snapshots = [1e10] * self._num_network

        for epoch in itertools.count():
            # 定义每一个网络的训练数据
            train_index = np.vstack([np.random.permutation(train_inputs.shape[0]) for _ in range(self._num_network)])
            # 所有真实数据都用来训练
            for batch_start_pos in range(0, train_inputs.shape[0], batch_size):
                batch_index = train_index[:, batch_start_pos:batch_start_pos + batch_size]
                train_input = torch.from_numpy(train_inputs[batch_index]).float().to(device)
                train_label = torch.from_numpy(train_labels[batch_index]).float().to(device)

                mean, logvar = self.model(train_input, return_log_var=True)
                loss, _ = self.model.loss(mean, logvar, train_label)
                self.model.train(loss)

            with torch.no_grad():
                mean, logvar = self.model(holdout_inputs, return_log_var=True)
                _, holdout_losses = self.model.loss(mean, logvar, holdout_labels, use_var_loss=False)
                holdout_losses = holdout_losses.cpu()
                break_condition = self._save_best(holdout_losses)
                if break_condition or epoch > max_iter:  # 结束训练
                    break

    def _save_best(self, losses, threshold=0.1):
        updated = False
        for i in range(len(losses)):
            if self._snapshots[i] - losses[i] > threshold * self._snapshots[i]:
                self._snapshots[i] = losses[i]
                updated = True
        self._epoch_since_last_update = 0 if updated else self._epoch_since_last_update + 1
        return self._epoch_since_last_update > 5

    def predict(self, inputs, batch_size=64):
        mean, var = [], []
        for i in range(0, inputs.shape[0], batch_size):
            input = torch.from_numpy(inputs[i:i + batch_size]).float().to(device)
            cur_mean, cur_var = self.model(input[None, :, :].repeat([self._num_network, 1, 1]), return_log_var=False)
            mean.append(cur_mean.cpu().detach().numpy())
            var.append(cur_var.cpu().detach().numpy())
        return np.hstack(mean), np.hstack(var)


class FakeEnv:
    def __init__(self, model):
        self.model = model

    def step(self, obs, act):
        inputs = np.concatenate((obs, act), axis=-1)
        ensemble_model_means, ensemble_model_vars = self.model.predict(inputs)
        ensemble_model_means[:, :, 1:] += obs.numpy()
        ensemble_model_stds = np.sqrt(ensemble_model_vars)
        ensemble_samples = ensemble_model_means + np.random.normal(size=ensemble_model_means.shape) * ensemble_model_stds

        num_models, batch_size, _ = ensemble_model_means.shape
        models_to_use = np.random.choice([i for i in range(self.model._num_network)], size=batch_size)
        samples = ensemble_samples[models_to_use, np.arange(0, batch_size)]
        rewards, next_obs = samples[:, :1], samples[:, 1:]
        return rewards, next_obs

    def propagate(self, obs, actions):
        with torch.no_grad():
            obs = np.copy(obs)
            total_reward = np.expand_dims(np.zeros(obs.shape[0]), axis=-1)
            obs, actions = torch.as_tensor(obs), torch.as_tensor(actions)
            for i in range(actions.shape[1]):
                action = torch.unsqueeze(actions[:, i], 1)
                rewards, next_obs = self.step(obs, action)
                total_reward += rewards
                obs = torch.as_tensor(next_obs)
            return total_reward


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def size(self):
        return len(self.buffer)

    def return_all_samples(self):
        return list(zip(*self.buffer))


class PETS:
    ''' PETS算法 '''

    def __init__(self, env, replay_buffer, n_sequence, elite_ratio,plan_horizon, num_episodes):
        self._env = env
        self._env_pool = replay_buffer
        obs_dim = env.observation_space.shape[0]
        self._action_dim = env.action_space.shape[0]
        self._model = EnsembleDynamicsModel(obs_dim, self._action_dim)
        self._fake_env = FakeEnv(self._model)
        self.upper_bound = env.action_space.high[0]
        self.lower_bound = env.action_space.low[0]
        self._cem = CEM(n_sequence, elite_ratio, self._fake_env,self.upper_bound, self.lower_bound)
        self.plan_horizon = plan_horizon
        self.num_episodes = num_episodes

    def train_model(self):
        env_samples = self._env_pool.return_all_samples()
        obs = np.array(env_samples[0])
        actions = np.array(env_samples[1])
        rewards = np.array(env_samples[2]).reshape(-1, 1)
        next_obs = np.array(env_samples[3])
        inputs = np.concatenate((obs, actions), axis=-1)
        labels = np.concatenate((rewards, next_obs - obs), axis=-1)
        self._model.train(inputs, labels)

    def mpc(self):
        mean = np.tile((self.upper_bound + self.lower_bound) / 2.0,
                       self.plan_horizon)
        var = np.tile(
            np.square(self.upper_bound - self.lower_bound) / 16,
            self.plan_horizon)
        (obs, _), done, episode_return = self._env.reset(), False, 0
        while not done:
            actions = self._cem.optimize(obs, mean, var)
            action = actions[:self._action_dim]  # 选取第一个动作
            next_obs, reward, terminated, truncated, info = self._env.step(action)
            done = terminated or truncated
            self._env_pool.add(obs, action, reward, next_obs, done)
            obs = next_obs
            episode_return += reward
            mean = np.concatenate([
                np.copy(actions)[self._action_dim:],
                np.zeros(self._action_dim)
            ])
        return episode_return

    def explore(self):
        (obs, _), done, episode_return = self._env.reset(), False, 0
        while not done:
            action = self._env.action_space.sample()
            next_obs, reward, terminated, truncated, info = self._env.step(action)
            done = terminated or truncated
            self._env_pool.add(obs, action, reward, next_obs, done)
            obs = next_obs
            episode_return += reward
        return episode_return

    def train(self):
        return_list = []
        explore_return = self.explore()  # 先进行随机策略的探索来收集一条序列的数据
        print('episode: 1, return: %d' % explore_return)
        return_list.append(explore_return)

        for i_episode in range(self.num_episodes - 1):
            self.train_model()
            episode_return = self.mpc()
            return_list.append(episode_return)
            print('episode: %d, return: %d' % (i_episode + 2, episode_return))
        return return_list


buffer_size = 100000
n_sequence = 50
elite_ratio = 0.2
plan_horizon = 25
num_episodes = 10
env_name = 'Pendulum-v1'
env = gym.make(env_name)

replay_buffer = ReplayBuffer(buffer_size)
pets = PETS(env, replay_buffer, n_sequence, elite_ratio, plan_horizon, num_episodes)
return_list = pets.train()

episodes_list = list(range(len(return_list)))
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('PETS on {}'.format(env_name))
plt.show()
