import collections
import numpy as np

if not hasattr(np, "bool8"):
    np.bool8 = np.bool_
import matplotlib.pyplot as plt
import random
import rl_utils
from tqdm import tqdm
import torch
import torch.nn.functional as F
import gym


class replay_buffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def size(self):
        return len(self.buffer)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        tmp = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*tmp)
        return np.array(states), actions, rewards, np.array(next_states), dones


class Qnet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Qnet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class DQN:
    def __init__(self, state_dim, hidden_dim, action_dim, lr, gamma, device, eps, target_update):
        self.action_dim = action_dim
        self.q_net = Qnet(state_dim, hidden_dim, action_dim).to(device)
        self.target_net = Qnet(state_dim, hidden_dim, action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.gamma = gamma
        self.eps = eps
        self.target_update = target_update
        self.device = device

        self.cnt = 0
        self.rng = np.random.default_rng(seed=91)

    def perform_action(self, state):
        if self.rng.random() < self.eps:
            return self.rng.integers(self.action_dim)
        else:
            state = torch.tensor(state, dtype=torch.float).to(self.device)
            return self.q_net(state).argmax().item()

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).to(self.device).view(-1, 1)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).to(self.device).view(-1, 1)

        q_values = self.q_net(states).gather(1, actions)    # Q(s,a)
        q_nex_max_value = self.target_net(next_states).max(1)[0].view(-1, 1)    # \max_a Q(s_next, a)

        q_target = rewards + self.gamma * q_nex_max_value * (1 - dones)
        dqn_loss = F.mse_loss(q_values, q_target).mean()
        self.optimizer.zero_grad()
        dqn_loss.backward()
        self.optimizer.step()

        self.cnt += 1
        if self.cnt % self.target_update == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())


if __name__ == '__main__':
    """
    Hyperparameter settings
    """
    episodes = 400
    gamma = 0.98
    eps = 0.05

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lr = 2e-3
    hidden_dim = 128
    batch_size = 64
    buffer_size = 10000
    target_update = 10
    minimal_size = 500
    # torch.manual_seed(1)
    """
    Coding
    """
    env_name = 'CartPole-v0'
    env = gym.make(env_name)
    replay_buffer = replay_buffer(buffer_size)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = DQN(state_dim, hidden_dim, action_dim, lr, gamma, device, eps, target_update)

    retn_list = []
    for cnt in tqdm(range(episodes), desc='Episodes'):
        state, _ = env.reset()
        tmp_sum = 0.
        while True:
            action = agent.perform_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            replay_buffer.add(state, action, reward, next_state, terminated)
            if replay_buffer.size() > minimal_size:
                batch_s, batch_a, batch_r, batch_s_next, batch_done = replay_buffer.sample(batch_size)
                transition_dict = {
                    'states': batch_s,
                    'actions': batch_a,
                    'rewards': batch_r,
                    'next_states': batch_s_next,
                    'dones': batch_done
                }
                agent.update(transition_dict)
            state = next_state
            tmp_sum += reward
            if terminated or truncated:
                break
        retn_list.append(tmp_sum)

    plt.plot(retn_list)
    plt.xlabel('Episodes')
    plt.ylabel('Reward Sum')
    plt.show()
    pass
