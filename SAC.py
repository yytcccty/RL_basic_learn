import numpy as np
from torch import nn

import rl_utils

if not hasattr(np, 'bool_8'):
    np.bool_8 = np.bool_
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import gym


class Actor(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound):
        super(Actor, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc_mu = torch.nn.Linear(hidden_dim, action_dim)
        self.fc_sigma = torch.nn.Linear(hidden_dim, action_dim)
        self.action_bound = action_bound

    def forward(self, x):
        x = F.relu(self.fc1(x))
        mu = self.fc_mu(x)
        sigma = F.softplus(self.fc_sigma(x))
        dist = torch.distributions.Normal(mu, sigma)

        normal_sample = dist.rsample()
        log_prob = dist.log_prob(normal_sample)

        action = torch.tanh(normal_sample)
        action = action * self.action_bound



        return action

class Critic(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Critic, self).__init__()

    def forward(self, state):


class SAC_Continuous:
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound, gamma, tau, lr_actor, lr_critic, lr_alpha,
                 device, target_entropy):
        self.actor = Actor(state_dim, hidden_dim, action_dim, action_bound).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)

        self.critic_1 = Critic(state_dim, hidden_dim, action_dim).to(device)
        self.critic_2 = Critic(state_dim, hidden_dim, action_dim).to(device)
        self.critic_1.optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=lr_critic)
        self.critic_2.optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=lr_alpha)

        self.critic_1_target = Critic(state_dim, hidden_dim, action_dim).to(device)
        self.critic_2_target = Critic(state_dim, hidden_dim, action_dim).to(device)
        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())

        self.log_alpha = torch.tensor(torch.log(0.01), dtype=torch.float, requires_grad=True)
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr_alpha)

        self.tau = tau
        self.gamma = gamma
        self.target_entropy = target_entropy
        self.device = device


if __name__ == '__main__':
    """
    Hyperparameter Settings
    """
    lr_actor = 1e-3
    lr_critic = 1e-3
    gamma = 0.95
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env_name = 'Pendulum-v1'  # CartPole Pendulum
    hidden_dim = 128
    lr_alpha = 1e-3
    episodes = 500
    buffer_size = 10000
    batch_size = 64
    minimal_size = 1000
    tau = 0.05
    """
    Coding
    """
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_bound = env.action_space.high[0]
    target_entropy = -env.action_space.shape[0]

    replay_buffer = rl_utils.ReplayBuffer(buffer_size)
    agent = SAC_Continuous(state_dim, hidden_dim, action_dim, action_bound, gamma, tau, lr_actor, lr_critic, lr_alpha,
                           device, target_entropy)
    retn_list = rl_utils.train_off_policy_agent(env, agent, episodes, replay_buffer, minimal_size, batch_size)

    plt.plot(retn_list)
    plt.xlabel('Episodes')
    plt.ylabel('Reward Sum')
    plt.grid(True)
    plt.show()

    mv_return = rl_utils.moving_average(retn_list, 9)
    plt.plot(mv_return)
    plt.xlabel('Episodes')
    plt.ylabel('Converted Reward')
