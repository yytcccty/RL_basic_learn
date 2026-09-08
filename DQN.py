import collections
import numpy as np
from matplotlib.lines import lineStyles

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
    def __init__(self, state_dim, hidden_dim, action_dim, lr, gamma, device, eps, target_update, dqn_type='vanilla'):
        self.action_dim = action_dim
        self.q_net = Qnet(state_dim, hidden_dim, action_dim).to(device)
        self.target_net = Qnet(state_dim, hidden_dim, action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.gamma = gamma
        self.eps = eps
        self.target_update = target_update
        self.device = device
        self.dqn_type = dqn_type

        self.cnt = 0
        self.rng = np.random.default_rng(seed=91)

    def max_q_value(self, state):
        state = torch.tensor(state, dtype=torch.float).to(self.device)
        return self.q_net(state).max().item()

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

        q_values = self.q_net(states).gather(1, actions)  # Q(s,a)
        if self.dqn_type == 'vanilla':
            q_nex_max_value = self.target_net(next_states).max(1)[0].view(-1, 1)  # \max_a Q(s_next, a)
        elif self.dqn_type == 'DoubleDQN':
            a_next_max_value = self.q_net(next_states).argmax(1).view(-1, 1)
            q_nex_max_value = self.target_net(next_states).gather(1, a_next_max_value)

        q_target = rewards + self.gamma * q_nex_max_value * (1 - dones)
        dqn_loss = F.mse_loss(q_values, q_target).mean()
        self.optimizer.zero_grad()
        dqn_loss.backward()
        self.optimizer.step()

        self.cnt += 1
        if self.cnt % self.target_update == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())


def train_DQN(agent, env, episodes, replay_buffer, minimal_size, buffer_size):
    retn_list = []
    max_q_list = []
    max_q = 0.
    for _ in tqdm(range(episodes), desc='Episodes'):
        state, _ = env.reset()
        tmp_sum = 0.
        while True:
            action = agent.perform_action(state)
            action_con = dis_to_con(action, env, agent.action_dim)
            next_state, reward, terminated, truncated, info = env.step([action_con])
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
            max_q = agent.max_q_value(state) * 0.005 + max_q * 0.995
            max_q_list.append(max_q)
            state = next_state
            tmp_sum += reward
            if terminated or truncated:
                break
        retn_list.append(tmp_sum)

    return retn_list, max_q_list


def dis_to_con(discrete_act, env, action_dim):
    action_lower = env.action_space.low[0]
    action_upper = env.action_space.high[0]
    return (action_upper - action_lower) * discrete_act / (action_dim - 1) + action_lower


if __name__ == '__main__':
    """
    Hyperparameter settings
    """
    episodes = 200
    gamma = 0.98
    eps = 0.05

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lr = 2e-3
    hidden_dim = 128
    batch_size = 64
    buffer_size = 5000
    target_update = 50
    minimal_size = 1000
    # torch.manual_seed(1)
    env_name = 'Pendulum-v1'
    """
    Coding
    """
    env = gym.make(env_name)
    replay_buffer = rl_utils.ReplayBuffer(buffer_size)
    state_dim = env.observation_space.shape[0]
    action_dim = 11
    agent = DQN(state_dim, hidden_dim, action_dim, lr, gamma, device, eps, target_update, dqn_type='DoubleDQN')

    retn_list, max_q_list = train_DQN(agent, env, episodes, replay_buffer, minimal_size, buffer_size)

    mv_return = rl_utils.moving_average(retn_list, 5)
    plt.plot(mv_return)
    plt.xlabel('Episodes')
    plt.ylabel('Converted Reward')
    plt.show()

    plt.plot(max_q_list)
    plt.axhline(0, color='orange', ls='--')
    plt.axhline(10, color='r', ls='--')
    plt.xlabel('Episodes')
    plt.ylabel('Max Q Value')
    plt.show()

    pass
