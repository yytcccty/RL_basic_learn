# This script needs another requirements.txt (requirements_for_IPPO.txt)
import torch
import torch.nn.functional as F
import numpy as np
import rl_utils
from tqdm import tqdm
import matplotlib.pyplot as plt

from ma_gym.envs.combat.combat import Combat


class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc2(F.relu(self.fc1(x))))
        return F.softmax(self.fc3(x), dim=1)


class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc2(F.relu(self.fc1(x))))
        return self.fc3(x)


class PPO:
    """ PPO算法,采用截断方式 """

    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                 lmbda, eps, gamma, device):
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr)
        self.gamma = gamma
        self.lmbda = lmbda
        self.eps = eps  # PPO中截断范围的参数
        self.device = device

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(
            self.device)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)
        td_target = rewards + self.gamma * self.critic(next_states) * (1 -
                                                                       dones)
        td_delta = td_target - self.critic(states)
        advantage = rl_utils.compute_advantage(self.gamma, self.lmbda,
                                               td_delta.cpu()).to(self.device)
        old_log_probs = torch.log(self.actor(states).gather(1,
                                                            actions)).detach()

        log_probs = torch.log(self.actor(states).gather(1, actions))
        ratio = torch.exp(log_probs - old_log_probs)
        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1 - self.eps,
                            1 + self.eps) * advantage  # 截断
        actor_loss = torch.mean(-torch.min(surr1, surr2))  # PPO损失函数
        critic_loss = torch.mean(
            F.mse_loss(self.critic(states), td_target.detach()))
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.step()


if __name__ == '__main__':
    actor_lr = 3e-4
    critic_lr = 1e-3
    hidden_dim = 64
    num_episodes = 100000
    gamma = 0.99
    lmbda = 0.97
    eps = 0.2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    team_size = 2
    grid_size = (15, 15)
    env = Combat(grid_shape=grid_size, n_agents=team_size, n_opponents=team_size)
    state_dim = env.observation_space[0].shape[0]
    action_dim = env.action_space[0].n

    agent = PPO(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda, eps, gamma, device)

    win_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc=f"Iteration: {i}") as pbar:
            for i_episode in range(int(num_episodes / 10)):
                trail_info_1 = dict(states=[], actions=[], rewards=[], next_states=[], dones=[])
                trail_info_2 = dict(states=[], actions=[], rewards=[], next_states=[], dones=[])
                s = env.reset()
                terminal = False
                while not terminal:
                    a_1 = agent.take_action(s[0])
                    a_2 = agent.take_action(s[1])
                    next_s, r, done, info = env.step([a_1, a_2])
                    trail_info_1['states'].append(s[0])
                    trail_info_1['actions'].append(a_1)
                    trail_info_1['rewards'].append(r[0] + 100 if info["win"] else r[0] - 0.1)
                    trail_info_1['next_states'].append(next_s[0])
                    trail_info_1['dones'].append(False)

                    trail_info_2['states'].append(s[1])
                    trail_info_2['actions'].append(a_2)
                    trail_info_2['rewards'].append(r[1] + 100 if info["win"] else r[1] - 0.1)
                    trail_info_2['next_states'].append(next_s[1])
                    trail_info_2['dones'].append(False)

                    s = next_s
                    terminal = all(done)
                win_list.append(1 if info["win"] else 0)
                agent.update(trail_info_1)
                agent.update(trail_info_2)
                if (i_episode + 1) % 100 == 0:
                    pbar.set_postfix({
                        "episode": f"{i * num_episodes / 10 + i_episode + 1}",
                        "avg win rate": f"{np.mean(win_list[-100:])}"
                    })
                pbar.update(1)

    win_array = np.array(win_list)
    win_array = np.mean(win_array.reshape(-1, 100), axis=1).flatten()
    xx = np.arange(win_array.shape[0]) * 100
    plt.plot(xx, win_array)
    plt.xlabel("Episode")
    plt.ylabel("Win rate")
    plt.show()
    pass
