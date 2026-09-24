import numpy as np
import rl_utils

if not hasattr(np, 'bool_8'):
    np.bool8 = np.bool_
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import gym


class ActorContinuous(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound):
        super(ActorContinuous, self).__init__()
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
        log_prob -= torch.log(1 - torch.tanh(normal_sample).pow(2) + 1e-7)
        action = action * self.action_bound
        return action, log_prob


class CriticContinuous(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(CriticContinuous, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = torch.nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc_out(x)
        return x


class ActorDiscrete(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(ActorDiscrete, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=1)
        return x


class CriticDiscrete(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(CriticDiscrete, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = self.fc2(x)
        return x


class SAC_Continuous:
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound, gamma, tau, lr_actor, lr_critic, lr_alpha,
                 device, target_entropy):
        self.actor = ActorContinuous(state_dim, hidden_dim, action_dim, action_bound).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)

        self.critic_1 = CriticContinuous(state_dim, hidden_dim, action_dim).to(device)
        self.critic_2 = CriticContinuous(state_dim, hidden_dim, action_dim).to(device)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=lr_critic)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=lr_alpha)

        self.critic_1_target = CriticContinuous(state_dim, hidden_dim, action_dim).to(device)
        self.critic_2_target = CriticContinuous(state_dim, hidden_dim, action_dim).to(device)
        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())

        self.log_alpha = torch.tensor(np.log(0.01), dtype=torch.float, requires_grad=True)
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr_alpha)

        self.tau = tau
        self.gamma = gamma
        self.target_entropy = target_entropy
        self.device = device

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float, device=self.device)
        action, _ = self.actor(state)
        return [action.item()]

    def calc_target(self, rewards, dones, next_states):
        next_actions, log_probs = self.actor(next_states)
        q_value = torch.min(self.critic_1_target(next_states, next_actions),
                            self.critic_2_target(next_states, next_actions))
        target = rewards + self.gamma * (q_value - self.log_alpha.exp() * log_probs) * (1 - dones)
        return target

    def soft_update(self, net, target_net):
        for para, old_para in zip(net.parameters(), target_net.parameters()):
            old_para.data.copy_(para.data * self.tau + old_para.data * (1 - self.tau))

    def update(self, trail_info):
        states = torch.tensor(np.array(trail_info['states']), dtype=torch.float).to(self.device)
        actions = torch.tensor(np.array(trail_info['actions']), dtype=torch.float).view(-1, 1).to(self.device)
        rewards = torch.tensor(np.array(trail_info['rewards']), dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(np.array(trail_info['next_states']), dtype=torch.float).to(self.device)
        dones = torch.tensor(np.array(trail_info['dones']), dtype=torch.float).view(-1, 1).to(self.device)
        # Calculate critic loss
        rewards = (rewards + 8.0) / 8.0
        td_target = self.calc_target(rewards, dones, next_states)
        critic_loss_1 = F.mse_loss(td_target.detach(), self.critic_1(states, actions))
        critic_loss_2 = F.mse_loss(td_target.detach(), self.critic_2(states, actions))
        self.critic_1_optimizer.zero_grad()
        critic_loss_1.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.zero_grad()
        critic_loss_2.backward()
        self.critic_2_optimizer.step()
        # Calculate actor loss
        actions, log_probs = self.actor(states)
        q_value = torch.min(self.critic_1(states, actions), self.critic_2(states, actions))
        actor_loss = torch.mean(self.log_alpha.exp() * log_probs - q_value)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        # Update temperature
        alpha_loss = torch.mean(-self.log_alpha.exp() * (log_probs.detach() + self.target_entropy))
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        self.soft_update(self.critic_1, self.critic_1_target)
        self.soft_update(self.critic_2, self.critic_2_target)


class SAC:
    def __init__(self, state_dim, hidden_dim, action_dim, gamma, tau, lr_actor, lr_critic, lr_alpha,
                 device, target_entropy):
        self.actor = ActorDiscrete(state_dim, hidden_dim, action_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)

        self.critic_1 = CriticDiscrete(state_dim, hidden_dim, action_dim).to(device)
        self.critic_2 = CriticDiscrete(state_dim, hidden_dim, action_dim).to(device)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=lr_critic)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=lr_alpha)

        self.critic_1_target = CriticDiscrete(state_dim, hidden_dim, action_dim).to(device)
        self.critic_2_target = CriticDiscrete(state_dim, hidden_dim, action_dim).to(device)
        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())

        self.log_alpha = torch.tensor(np.log(0.01), dtype=torch.float, requires_grad=True)
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr_alpha)

        self.tau = tau
        self.gamma = gamma
        self.target_entropy = target_entropy
        self.device = device

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float, device=self.device)
        probs = self.actor(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item()

    def calc_target(self, rewards, dones, next_states):
        next_probs = self.actor(next_states)
        next_log_probs = torch.log(next_probs + 1e-8)
        entropy = -torch.sum(next_probs * next_log_probs, dim=1, keepdim=True)
        q_1_value = self.critic_1_target(next_states)
        q_2_value = self.critic_2_target(next_states)
        q_min_value = torch.sum(next_probs * torch.min(q_1_value, q_2_value), dim=1, keepdim=True)
        q_next = q_min_value + entropy * self.log_alpha.exp()
        target = rewards + self.gamma * q_next * (1 - dones)
        return target

    def soft_update(self, net, target_net):
        for para, old_para in zip(net.parameters(), target_net.parameters()):
            old_para.data.copy_(para.data * self.tau + old_para.data * (1 - self.tau))

    def update(self, trail_info):
        states = torch.tensor(np.array(trail_info['states']), dtype=torch.float).to(self.device)
        actions = torch.tensor(np.array(trail_info['actions'])).view(-1, 1).to(self.device)
        rewards = torch.tensor(np.array(trail_info['rewards']), dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(np.array(trail_info['next_states']), dtype=torch.float).to(self.device)
        dones = torch.tensor(np.array(trail_info['dones']), dtype=torch.float).view(-1, 1).to(self.device)
        # Calculate critic loss
        rewards = (rewards + 8.0) / 8.0
        td_target = self.calc_target(rewards, dones, next_states)
        critic_loss_1 = F.mse_loss(td_target.detach(), self.critic_1(states).gather(1, actions))
        critic_loss_2 = F.mse_loss(td_target.detach(), self.critic_2(states).gather(1, actions))
        self.critic_1_optimizer.zero_grad()
        critic_loss_1.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.zero_grad()
        critic_loss_2.backward()
        self.critic_2_optimizer.step()
        # Calculate actor loss
        probs = self.actor(states)
        log_prbs = torch.log(probs + 1e-8)
        entropy = -torch.sum(probs * log_prbs, dim=1, keepdim=True)
        q_value = torch.min(self.critic_1(states), self.critic_2(states))

        actor_loss = torch.mean(-entropy * self.log_alpha.exp() - torch.sum(probs * q_value, dim=1, keepdim=True))
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        # Update temperature
        alpha_loss = torch.mean(self.log_alpha.exp() * (entropy.detach() - self.target_entropy))
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        self.soft_update(self.critic_1, self.critic_1_target)
        self.soft_update(self.critic_2, self.critic_2_target)


if __name__ == '__main__':
    """
    Hyperparameter Settings
    """
    lr_actor = 3e-4
    lr_critic = 3e-3
    lr_alpha = 3e-4
    gamma = 0.99
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env_name = 'CartPole-v1'  # CartPole Pendulum
    hidden_dim = 128

    episodes = 100
    buffer_size = 100000
    batch_size = 64
    minimal_size = 1000
    tau = 0.005
    """
    Coding
    """
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    if 'CartPole' in env_name:
        lr_actor = 1e-3
        lr_critic = 1e-3
        lr_alpha = 1e-3
        episodes = 300
        gamma = 0.98
        buffer_size = 10000
        minimal_size = 500
        target_entropy = -1.
        action_dim = env.action_space.n
        agent = SAC(state_dim, hidden_dim, action_dim, gamma, tau, lr_actor, lr_critic,
                    lr_alpha, device, target_entropy)
    else:
        target_entropy = -env.action_space.shape[0]
        action_bound = env.action_space.high[0]
        action_dim = env.action_space.shape[0]
        agent = SAC_Continuous(state_dim, hidden_dim, action_dim, action_bound, gamma, tau, lr_actor, lr_critic,
                               lr_alpha, device, target_entropy)
        action_dim = env.action_space.shape[0]

    replay_buffer = rl_utils.ReplayBuffer(buffer_size)

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
    plt.grid(True)
    plt.show()
