import numpy as np
if not hasattr(np, "bool8"):
    np.bool8 = np.bool_
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import gym
import rl_utils


class Policy_net(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Policy_net, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=-1)
        return x


class Value_net(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(Value_net, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class ActorCritic:
    def __init__(self, state_dim, hidden_dim, action_dim, lr_actor, lr_critic, gamma, device):
        self.actor = Policy_net(state_dim, hidden_dim, action_dim).to(device)
        self.critic = Value_net(state_dim, hidden_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.gamma = gamma
        self.device = device

    def take_action(self, state):
        state = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def update(self, trail_info):
        rewards = torch.tensor(np.array(trail_info['rewards']), dtype=torch.float).view(-1, 1).to(self.device)
        states_next = torch.tensor(np.array(trail_info['next_states']), dtype=torch.float).to(self.device)
        states = torch.tensor(np.array(trail_info['states']), dtype=torch.float).to(self.device)
        actions = torch.tensor(np.array(trail_info['actions'])).view(-1, 1).to(self.device)
        dones = torch.tensor(np.array(trail_info['dones']), dtype=torch.float).view(-1, 1).to(self.device)

        td_target = rewards + self.gamma * self.critic(states_next) * (1 - dones)

        td_error = td_target - self.critic(states)
        log_probs = torch.log(self.actor(states).gather(1, actions))
        actor_loss = -torch.mean(log_probs * td_error.detach())
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        critic_loss = F.mse_loss(td_target.detach(), self.critic(states))
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()


if __name__ == '__main__':
    """
    Hyperparameter settings
    """
    lr_actor = 1e-3
    lr_critic = 1e-2
    episodes = 1000
    hidden_dim = 128
    gamma = 0.98
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env_name = "CartPole-v0"
    """
    Codings
    """
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = ActorCritic(state_dim, hidden_dim, action_dim, lr_actor, lr_critic, gamma, device)
    return_list = rl_utils.train_on_policy_agent(env, agent, episodes)

    plt.plot(return_list)
    plt.xlabel("Episodes")
    plt.ylabel("Reward Sum")
    plt.show()

    mv_retun = rl_utils.moving_average(return_list, 9)
    plt.plot(mv_retun)
    plt.xlabel("Episodes")
    plt.ylabel("Converted Reward")
    plt.show()

    pass
