import numpy as np

if not hasattr(np, "bool8"):
    np.bool8 = np.bool_
import matplotlib.pyplot as plt
import gym
import torch
import torch.nn.functional as F
import rl_utils


class Actor(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound):
        super(Actor, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)
        self.action_bound = action_bound

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.tanh(self.fc2(x))
        return x * self.action_bound


class Critic(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc_out(x)


class DDPG:
    def __init__(self, state_dim, action_dim, hidden_dim, gamma, tau, action_bound, device, lr_actor, lr_critic, sigma):
        self.actor = Actor(state_dim, hidden_dim, action_dim, action_bound).to(device)
        self.critc = Critic(state_dim, hidden_dim, action_dim).to(device)
        self.actor_target = Actor(state_dim, hidden_dim, action_dim, action_bound).to(device)
        self.critc_target = Critic(state_dim, hidden_dim, action_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critc_target.load_state_dict(self.critc.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr_actor)
        self.critc_optimizer = torch.optim.Adam(self.critc.parameters(), lr_critic)
        self.gamma = gamma
        self.tau = tau
        self.sigma = sigma
        self.device = device
        self.action_dim = action_dim

    def take_action(self, state):
        state = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)
        action = self.actor(state).item()
        noise = np.random.randn(self.action_dim) * self.sigma
        return action + noise

    def soft_update(self, net, target_net):
        for para, old_para in zip(net.parameters(), target_net.parameters()):
            old_para.data.copy_(para.data * self.tau + old_para.data * (1 - self.tau))

    def update(self, trail_info):
        rewards = torch.tensor(np.array(trail_info['rewards']), dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(np.array(trail_info['next_states']), dtype=torch.float).to(self.device)
        states = torch.tensor(np.array(trail_info['states']), dtype=torch.float).to(self.device)
        actions = torch.tensor(np.array(trail_info['actions']), dtype=torch.float).view(-1, 1).to(self.device)
        dones = torch.tensor(np.array(trail_info['dones']), dtype=torch.float).view(-1, 1).to(self.device)

        q_target = rewards + self.gamma * self.critc_target(next_states, self.actor_target(next_states)) * (1 - dones)
        critic_loss = torch.mean(F.mse_loss(q_target, self.critc(states, actions)))
        self.critc_optimizer.zero_grad()
        critic_loss.backward()
        self.critc_optimizer.step()

        actor_loss = -torch.mean(self.critc(states, self.actor(states)))
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.soft_update(self.critc, self.critc_target)
        self.soft_update(self.actor, self.actor_target)


if __name__ == '__main__':
    """
    Hyperparameter settings
    """
    gamma = 0.98
    tau = 0.005
    lr_actor = 3e-4
    lr_critic = 3e-3
    hidden_dim = 64
    episodes = 200
    capacity = 10000
    minimal_size = 1000
    batch_size = 64
    env_name = 'Pendulum-v1'
    sigma = 0.01
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    """
    Coding
    """
    env = gym.make(env_name)
    replay_buffer = rl_utils.ReplayBuffer(capacity)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_bound = env.action_space.high[0]
    agent = DDPG(state_dim, action_dim, hidden_dim, gamma, tau, action_bound, device, lr_actor, lr_critic, sigma)
    retn_list = rl_utils.train_off_policy_agent(env, agent, episodes, replay_buffer, minimal_size, batch_size)

    plt.plot(retn_list)
    plt.xlabel("Episodes")
    plt.ylabel("Reward Sum")
    plt.show()

    mv_retun = rl_utils.moving_average(retn_list, 9)
    plt.plot(mv_retun)
    plt.xlabel("Episodes")
    plt.ylabel("Converted Reward")
    plt.show()

    pass
