from Basic_DRL.PPO import PPO, PolicyNet
import rl_utils
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import gym
from tqdm import tqdm


def sample_expert_data(agent, env, n_episodes):
    states = []
    actions = []
    for _ in range(n_episodes):
        state, _ = env.reset()
        done = False
        while not done:
            action = agent.take_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            states.append(state)
            actions.append(action)
            state = next_state
    return np.array(states), np.array(actions)


class BehaviorClone:
    def __init__(self, state_dim, hidden_dim, action_dim, lr, device):
        self.policy_net = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)
        self.device = device

    def take_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        prob = self.policy_net(state)
        dist = torch.distributions.Categorical(prob)
        action = dist.sample()
        return action.item()

    def learn(self, expert_s, expert_a):
        feature = torch.from_numpy(expert_s).float().to(self.device)
        label = torch.from_numpy(expert_a).view(-1, 1).to(self.device)
        loss = -torch.mean(torch.log(self.policy_net(feature).gather(1, label)))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


def test_BC(agent_bc, env, n_episode):
    retn = []
    for _ in range(n_episode):
        tmp_return = 0
        state, _ = env.reset()
        done = False
        while not done:
            action = agent_bc.take_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            tmp_return += reward
            state = next_state
        retn.append(tmp_return)
    return np.mean(retn)


class Discriminator(torch.nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(Discriminator, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, states, actions):
        x = torch.cat((states, actions), dim=1)
        x = F.relu(self.fc1(x))
        return F.sigmoid(self.fc2(x))


class GAIL:
    def __init__(self, agent, state_dim, action_dim, hidden_dim, lr_d, device):
        self.device = device
        self.agent = agent
        self.discriminator = Discriminator(state_dim, action_dim, hidden_dim).to(device)
        self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=lr_d)

    def learn(self, expert_s, expert_a, trail_info):
        expert_s = torch.from_numpy(expert_s).float().to(self.device)
        expert_a = torch.from_numpy(expert_a).to(self.device)
        agent_s = torch.tensor(np.array(trail_info['states']), dtype=torch.float).to(self.device)
        agent_a = torch.tensor(np.array(trail_info['actions'])).to(self.device)

        expert_a = F.one_hot(expert_a, num_classes=2).float()
        agent_a = F.one_hot(agent_a, num_classes=2).float()

        expert_probs = self.discriminator(expert_s, expert_a)
        agent_probs = self.discriminator(agent_s, agent_a)
        d_loss = -torch.log(1. - expert_probs).mean() - torch.log(agent_probs).mean()
        self.discriminator_optimizer.zero_grad()
        d_loss.backward()
        self.discriminator_optimizer.step()

        trail_info['rewards'] = -torch.log(agent_probs).detach().cpu().numpy()
        self.agent.update(trail_info)


if __name__ == '__main__':
    method = 'GAIL'  # BC GAIL
    """
    Hyperparameters of PP0
    """
    actor_lr = 1e-3
    critic_lr = 1e-2
    num_episodes = 250
    hidden_dim = 128
    gamma = 0.98
    lmbda = 0.95
    epochs = 10
    eps = 0.2
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    env_name = 'CartPole-v1'
    """
    PPO definition
    """
    env = gym.make(env_name)
    state_space = env.observation_space
    action_space = env.action_space
    agent = PPO(state_space, action_space, hidden_dim, eps, gamma, critic_lr, actor_lr, lmbda, device, epochs)
    """
    Sample expert data
    """
    n_episodes = 1
    n_samples = 30
    expert_s, expert_a = sample_expert_data(agent, env, n_episodes)
    idx = np.random.randint(0, expert_s.shape[0], n_samples)
    expert_s = expert_s[idx]
    expert_a = expert_a[idx]
    if method == 'BC':
        retn_list = rl_utils.train_on_policy_agent(env, agent, num_episodes)
        """
        Behavior Cloning
        """
        lr_BC = 1e-3
        BC_agent = BehaviorClone(state_space.shape[0], hidden_dim, action_space.n, lr_BC, device)
        batch_size = 64
        n_iterations = 1000
        retn_list = []
        with tqdm(total=n_iterations, desc="Iterations") as pbar:
            for i in range(n_iterations):
                idx = np.random.randint(0, expert_s.shape[0], batch_size)
                batch_s = expert_s[idx]
                batch_a = expert_a[idx]
                BC_agent.learn(batch_s, batch_a)
                retn = test_BC(BC_agent, env, 5)
                retn_list.append(retn)

                if (i + 1) % 10 == 0:
                    pbar.set_postfix({"Return avg": retn})
                pbar.update(1)
    else:
        lr_d = 1e-3
        gail = GAIL(agent, state_space.shape[0], action_space.n, hidden_dim, lr_d, device)
        n_episodes = 500
        retn_list = []
        with tqdm(total=n_episodes, desc="Iterations") as pbar:
            for i in range(n_episodes):
                state, _ = env.reset()
                done = False
                tmp_return = 0.
                transition_dict = {'states': [], 'actions': [], 'next_states': [], 'dones': []}
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    transition_dict['states'].append(state)
                    transition_dict['actions'].append(action)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['dones'].append(done)
                    state = next_state
                    tmp_return += reward
                retn_list.append(tmp_return)
                gail.learn(expert_s, expert_a, transition_dict)
                if (i + 1) % 10 == 0:
                    pbar.set_postfix({"Return avg": np.mean(retn_list[-10:])})
                pbar.update(1)

    """
    Plot
    """
    plt.plot(retn_list)
    plt.xlabel("Iterations")
    plt.ylabel("Return avg")
    plt.grid(True)
    plt.show()
    pass
