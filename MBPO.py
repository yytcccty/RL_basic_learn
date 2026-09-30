import numpy as np
import matplotlib.pyplot as plt
import gym
import torch
import collections
import random
from Basic_DRL.SAC import SAC_Continuous
from PETS import EnsembleDynamicsModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EnsembleDynamicsModel_Onestep(EnsembleDynamicsModel):
    def predict(self, inputs, batch_size=64):
        inputs = np.tile(inputs, (self._num_network, 1, 1))
        inputs = torch.from_numpy(inputs).float().to(device)
        mean, var = self.model(inputs, return_logvar=False)
        return mean.detach().cpu().numpy(), var.detach().cpu().numpy()


class FakeEnv:
    def __init__(self, model):
        self.model = model

    def step(self, obs, action):
        inputs = np.concatenate((obs, action), axis=-1)
        ensemble_model_means, ensemble_model_vars = self.model.predict(inputs)
        ensemble_model_means[:, :, 1:] += obs
        ensemble_model_std = np.sqrt(ensemble_model_vars)
        ensemble_samples = np.random.normal(size=ensemble_model_means.shape) * ensemble_model_std + ensemble_model_means

        num_models, batch_size, _ = ensemble_model_means.shape
        models_to_use = np.random.choice(num_models, size=batch_size)
        samples = ensemble_samples[models_to_use, np.arange(0, batch_size)]
        rewards = samples[:, :1][0][0]
        next_obs = samples[:, 1:][0]
        return rewards, next_obs


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def size(self):
        return len(self.buffer)

    def sample(self, batch_size):
        if batch_size > len(self.buffer):
            return self.return_all_samples()
        else:
            transitions = random.sample(self.buffer, batch_size)
            return list(zip(*transitions))

    def return_all_samples(self):
        return list(zip(*self.buffer))


class MBPO:
    def __init__(self, env, agent, fake_env, env_pool, model_pool, real_ratio, rollout_length, rollout_batch_size,
                 num_episodes):
        self.env = env
        self.agent = agent
        self.fake_env = fake_env
        self.env_pool = env_pool
        self.model_pool = model_pool
        self.real_ratio = real_ratio
        self.rollout_length = rollout_length
        self.rollout_batch_size = rollout_batch_size
        self.num_episodes = num_episodes

    def train(self):
        retn_list = []
        explore_retn = self.explore()
        print(f'Episode 1 return: {explore_retn}')
        retn_list.append(explore_retn)

        for i in range(self.num_episodes - 1):
            obs, _ = self.env.reset()
            done = False
            episode_retn = 0
            step = 0
            while not done:
                if step % 50 == 0:
                    self.train_model()
                    self.rollout_model()
                action = self.agent.take_action(obs)
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                self.env_pool.add(obs, action, reward, next_obs, done)
                episode_retn += reward
                obs = next_obs
                self.update_agent()
                step += 1
            retn_list.append(episode_retn)
            print(f'Episode {i + 2} return: {episode_retn}')
        return retn_list

    def explore(self):
        obs, _ = self.env.reset()
        done = False
        episode_retn = 0
        while not done:
            action = self.env.action_space.sample()
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            self.env_pool.add(obs, action, reward, next_obs, done)
            obs = next_obs
            episode_retn += reward
        return episode_retn

    def train_model(self):
        env_samples = self.env_pool.return_all_samples()
        obs = np.array(env_samples[0])
        actions = np.array(env_samples[1])
        rewards = np.array(env_samples[2]).reshape(-1, 1)
        next_obs = np.array(env_samples[3])
        inputs = np.concatenate((obs, actions), axis=1)
        labels = np.concatenate((rewards, next_obs - obs), axis=1)
        self.fake_env.model.train(inputs, labels)

    def rollout_model(self):
        env_samples = self.env_pool.sample(self.rollout_batch_size)
        observations = np.array(env_samples[0])
        for obs in observations:
            for _ in range(self.rollout_length):
                action = self.agent.take_action(obs)
                reward, next_obs = self.fake_env.step(obs, action)
                self.model_pool.add(obs, action, reward, next_obs, False)
                obs = next_obs

    def update_agent(self, policy_train_batch_size=64):
        env_batch_size = int(policy_train_batch_size * self.real_ratio)
        model_batch_size = policy_train_batch_size - env_batch_size
        for _ in range(10):
            env_samples = self.env_pool.sample(env_batch_size)
            env_obs = np.array(env_samples[0])
            env_actions = np.array(env_samples[1])
            env_rewards = np.array(env_samples[2])
            env_next_obs = np.array(env_samples[3])
            env_dones = np.array(env_samples[4])
            if self.model_pool.size() > 0:
                model_samples = self.model_pool.sample(model_batch_size)
                model_obs = np.array(model_samples[0])
                model_actions = np.array(model_samples[1])
                model_rewards = np.array(model_samples[2])
                model_next_obs = np.array(model_samples[3])
                model_dones = np.array(model_samples[4])
                obs = np.concatenate((env_obs, model_obs), axis=0)
                actions = np.concatenate((env_actions, model_actions), axis=0)
                rewards = np.concatenate((env_rewards, model_rewards), axis=0)
                next_obs = np.concatenate((env_next_obs, model_next_obs), axis=0)
                dones = np.concatenate((env_dones, model_dones), axis=0)
            else:
                obs, actions, rewards, next_obs, dones = env_obs, env_actions, env_rewards, env_next_obs, env_dones
            trail_info = {
                'states': obs,
                'actions': actions,
                'rewards': rewards,
                'next_states': next_obs,
                'dones': dones
            }
            self.agent.update(trail_info)


if __name__ == '__main__':
    env_name = "Pendulum-v1"
    lr_actor = 5e-4
    lr_critic = 5e-3
    lr_alpha = 1e-3
    hidden_dim = 128
    gamma = 0.98
    tau = 0.005
    buffer_size = 10000
    target_entropy = -1
    rollout_batch_size = 1000
    rollout_length = 1
    real_ratio = 0.5
    n_episodes = 20
    """
    Codings
    """
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_bound = env.action_space.high[0]
    model_pool_size = rollout_batch_size * rollout_length
    agent = SAC_Continuous(state_dim, hidden_dim, action_dim, action_bound, gamma, tau, lr_actor, lr_critic, lr_alpha,
                           device, target_entropy)
    model = EnsembleDynamicsModel_Onestep(state_dim, action_dim, )
    fake_env = FakeEnv(model)
    env_pool = ReplayBuffer(buffer_size)
    model_pool = ReplayBuffer(model_pool_size)
    mbpo = MBPO(env, agent, fake_env, env_pool, model_pool, real_ratio, rollout_length, rollout_batch_size, n_episodes)

    retn_list = mbpo.train()

    plt.plot(retn_list)
    plt.xlabel("Episodes")
    plt.ylabel("Reward Sum")
    plt.show()
