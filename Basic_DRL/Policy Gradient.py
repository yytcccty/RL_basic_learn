import numpy as np

import rl_utils

if not hasattr(np, "bool8"):
    np.bool8 = np.bool_
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import gym
from tqdm import tqdm


class Policy_net(torch.nn.Module):
    def __init__(self, state_dim, hidden_size, action_dim):
        super(Policy_net, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.softmax(self.fc2(x), dim=1)
        return x


class REINFORCE:
    def __init__(self, state_dim, hidden_size, action_dim, gamma, lr, device):
        self.gamma = gamma
        self.device = device
        self.policy = Policy_net(state_dim, hidden_size, action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

    def take_action(self, state):
        state = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)
        probs = self.policy(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def update(self, trail_info):
        states = trail_info['states']
        actions = trail_info['actions']
        rewards = trail_info['rewards']

        self.optimizer.zero_grad()
        G = 0.
        tmp_len = len(states) - 1
        for i in range(tmp_len, -1, -1):
            G = G * self.gamma + rewards[i]
            state = torch.tensor(np.array([states[i]]), dtype=torch.float).to(self.device)
            action = torch.tensor(np.array([actions[i]])).view(-1, 1).to(self.device)
            log_prob = torch.log(self.policy(state)).gather(1, action)
            loss = -log_prob * G
            loss.backward()
        self.optimizer.step()


if __name__ == '__main__':
    """
    Hyperparameter settings
    """
    lr = 1e-3
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
    agent = REINFORCE(state_dim, hidden_dim, action_dim, gamma, lr, device)

    retn_list = []
    for _ in tqdm(range(episodes), desc="Episodes"):
        state, _ = env.reset()
        tmp_sum = 0.
        trail_dict = {
            'states': [],
            'actions': [],
            'rewards': [],
        }
        while True:
            action = agent.take_action(state)
            next_state, reward, terminal, truncated, info = env.step(action)
            trail_dict['states'].append(state)
            trail_dict['actions'].append(action)
            trail_dict['rewards'].append(reward)
            state = next_state
            tmp_sum += reward
            if terminal or truncated:
                break
        agent.update(trail_dict)
        retn_list.append(tmp_sum)

    plt.plot(retn_list)
    plt.xlabel('Episodes')
    plt.ylabel('Reward Sum')
    plt.grid(True)
    plt.show()

    mv_return = rl_utils.moving_average(retn_list, 5)
    plt.plot(mv_return)
    plt.xlabel('Episodes')
    plt.ylabel('Converted Reward')
    plt.show()

    pass
