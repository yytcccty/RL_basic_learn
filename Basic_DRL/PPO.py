import numpy as np
import rl_utils
if not hasattr(np, "bool8"):
    np.bool8 = np.bool_
import matplotlib.pyplot as plt
import gym
import torch
import torch.nn.functional as F


class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=-1)
        return x


class PolicyNetContinuous(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNetContinuous, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc_mu = torch.nn.Linear(hidden_dim, action_dim)
        self.fc_sigma = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        mu = 2.0 * torch.tanh(self.fc_mu(x))
        sigma = F.softplus(self.fc_sigma(x))
        return mu, sigma


class Value_net(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(Value_net, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def compute_GAE(lamda, gamma, td_error):
    """
    Using Generalized Advantage Estimation (GAE) to calculate advantages of the sampled path
    :param lamda: Hyperparameter of GAE
    :param gamma: Hyperparameter of RL
    :param td_error: The temporal difference error in the sampled path
    :return: Generalized advantage of all states in the sampled path
    """
    td_delta = td_error.detach().numpy()
    advan_GAE_list = []
    tmp_GAE = 0.
    for delta in td_delta[::-1]:
        tmp_GAE = tmp_GAE * gamma * lamda + delta
        advan_GAE_list.append(tmp_GAE)
    advan_GAE_list.reverse()
    return torch.tensor(np.array(advan_GAE_list), dtype=torch.float)


class PPO:
    """
    Suitable for discrete action space
    """

    def __init__(self, state_space, action_space, hidden_dim, eps, gamma, lr_critic, lr_actor, lamda, device, rounds):
        self.actor = PolicyNet(state_space.shape[0], hidden_dim, action_space.n).to(device)
        self.critic = Value_net(state_space.shape[0], hidden_dim).to(device)
        self.gamma = gamma
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.lamda = lamda
        self.device = device
        self.eps = eps
        self.rounds = rounds

    def take_action(self, state):
        state = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)
        prob_dist = self.actor(state)
        action_dist = torch.distributions.Categorical(prob_dist)
        action = action_dist.sample()
        return action.item()

    def update(self, trail_info):
        states = torch.tensor(np.array(trail_info['states']), dtype=torch.float).to(self.device)
        actions = torch.tensor(np.array(trail_info['actions'])).view(-1, 1).to(self.device)
        rewards = torch.tensor(np.array(trail_info['rewards']), dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(np.array(trail_info['next_states']), dtype=torch.float).to(self.device)
        dones = torch.tensor(np.array(trail_info['dones']), dtype=torch.float).view(-1, 1).to(self.device)

        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)

        td_error = td_target - self.critic(states)  # advantages of each state in the trail
        advan_GAE = rl_utils.compute_advantage(self.gamma, self.lamda, td_error.cpu()).to(device)
        old_pi_log = torch.log(self.actor(states).gather(1, actions)).detach()

        for _ in range(self.rounds):
            new_action_log = torch.log(self.actor(states).gather(1, actions))
            ratio = torch.exp(new_action_log - old_pi_log)
            clipped_ratio = torch.clamp(ratio, 1 - self.eps, 1 + self.eps)
            actor_loss = -torch.min(ratio * advan_GAE, clipped_ratio * advan_GAE).mean()  # Update actor using PPO-clip
            critic_loss = F.mse_loss(td_target.detach(), self.critic(states))
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.critic_optimizer.step()
            self.actor_optimizer.step()


class PPOContinuous:
    """
    Suitable for continuous action space
    """

    def __init__(self, state_space, action_space, hidden_dim, eps, gamma, lr_critic, lr_actor, lamda, device, rounds):
        self.actor = PolicyNetContinuous(state_space.shape[0], hidden_dim, action_space.shape[0]).to(device)
        self.critic = Value_net(state_space.shape[0], hidden_dim).to(device)
        self.gamma = gamma
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.lamda = lamda
        self.device = device
        self.eps = eps
        self.rounds = rounds

    def take_action(self, state):
        state = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)
        mu, sigma = self.actor(state)
        action_dist = torch.distributions.Normal(mu, sigma)
        action = action_dist.sample()
        return [action.item()]

    def update(self, trail_info):
        states = torch.tensor(np.array(trail_info['states']), dtype=torch.float).to(self.device)
        actions = torch.tensor(np.array(trail_info['actions'])).view(-1, 1).to(self.device)
        rewards = torch.tensor(np.array(trail_info['rewards']), dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(np.array(trail_info['next_states']), dtype=torch.float).to(self.device)
        dones = torch.tensor(np.array(trail_info['dones']), dtype=torch.float).view(-1, 1).to(self.device)

        rewards = (rewards + 8.0) / 8.0
        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)

        td_error = td_target - self.critic(states)  # advantages of each state in the trail
        advan_GAE = rl_utils.compute_advantage(self.gamma, self.lamda, td_error.cpu()).to(device)
        mu, sigma = self.actor(states)
        old_pi_log = torch.distributions.Normal(mu.detach(), sigma.detach()).log_prob(actions)

        for _ in range(self.rounds):
            mu, sigma = self.actor(states)
            new_action_log = torch.distributions.Normal(mu, sigma).log_prob(actions)
            ratio = torch.exp(new_action_log - old_pi_log)
            clipped_ratio = torch.clamp(ratio, 1 - self.eps, 1 + self.eps)
            actor_loss = -torch.min(ratio * advan_GAE, clipped_ratio * advan_GAE).mean()  # Update actor using PPO-clip
            critic_loss = F.mse_loss(td_target.detach(), self.critic(states))
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.critic_optimizer.step()
            self.actor_optimizer.step()


if __name__ == '__main__':
    """
    Hyperparameter Settings:
    """
    env_name = "Pendulum-v1"  # CartPole-v1, Pendulum-v1
    if "Pendulum" in env_name:
        gamma = 0.9
        lamda = 0.9
        episodes = 2000
        lr_critic = 5e-3
        lr_actor = 1e-4
    else:
        gamma = 0.98
        lamda = 0.95
        episodes = 500
        lr_critic = 1e-2
        lr_actor = 1e-3
    rounds = 10
    alpha = 0.5
    eps = 0.2
    hidden_dim = 128
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    """
    Codings
    """
    env = gym.make(env_name)
    state_space = env.observation_space
    action_space = env.action_space
    if "Pendulum" in env_name:
        agent = PPOContinuous(state_space, action_space, hidden_dim, eps, gamma, lr_critic, lr_actor, lamda, device, rounds)
    else:
        agent = PPO(state_space, action_space, hidden_dim, eps, gamma, lr_critic, lr_actor, lamda, device, rounds)
    retn_list = rl_utils.train_on_policy_agent(env, agent, episodes)

    plt.plot(retn_list)
    plt.xlabel("Episodes")
    plt.ylabel("Reward Sum")
    plt.show()

    mv_return = rl_utils.moving_average(retn_list, 9)
    plt.plot(mv_return)
    plt.xlabel("Episodes")
    plt.ylabel("Converted Reward")
    plt.show()

    pass
