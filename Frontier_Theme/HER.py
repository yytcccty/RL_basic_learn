import collections
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import random
from Basic_DRL.DDPG import DDPG


class WorldEnv:
    def __init__(self):
        self.distance_threshold = 0.15
        self.action_bound = 1

    def reset(self):
        self.goal = np.array([4.0 + random.uniform(-0.5, 0.5), 4.0 + random.uniform(-0.5, 0.5)])
        self.state = np.array([0, 0])
        self.count = 0
        return np.hstack((self.state, self.goal))

    def step(self, action):
        action = np.clip(action, -self.action_bound, self.action_bound)
        x = max(min(self.state[0] + action[0], 5), 0)
        y = max(min(self.state[1] + action[1], 5), 0)
        self.state = np.array([x, y])
        self.count += 1

        dist = np.sqrt(np.sum(np.square(self.state - self.goal)))
        reward = 0 if dist <= self.distance_threshold else -1
        if dist <= self.distance_threshold or self.count == 50:
            done = True
        else:
            done = False
        return np.hstack((self.state, self.goal)), reward, done


class PolicyNet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, action_dim)
        self.action_bound = action_bound

    def forward(self, x):
        x = F.relu(self.fc2(F.relu(self.fc1(x))))
        return torch.tanh(self.fc3(x)) * self.action_bound


class QValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(QValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x, a):
        cat = torch.cat([x, a], dim=1)
        x = F.relu(self.fc2(F.relu(self.fc1(cat))))
        return self.fc3(x)


class DDPG_HER(DDPG):
    def take_action(self, state):
        state = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)
        action = self.actor(state).detach().cpu().numpy()[0]
        noise = np.random.randn(self.action_dim) * self.sigma
        return action + noise

    def update(self, trail_info):
        rewards = torch.tensor(np.array(trail_info['rewards']), dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(np.array(trail_info['next_states']), dtype=torch.float).to(self.device)
        states = torch.tensor(np.array(trail_info['states']), dtype=torch.float).to(self.device)
        actions = torch.tensor(np.array(trail_info['actions']), dtype=torch.float).to(self.device)
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


class Trajectory:
    def __init__(self, init):
        self.states = [init]
        self.actions = []
        self.rewards = []
        self.dones = []
        self.length = 0

    def store_step(self, state, action, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.length += 1


class ReplayBuffer_Trajectory:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add_trajectory(self, trajectory):
        self.buffer.append(trajectory)

    def size(self):
        return len(self.buffer)

    def sample(self, batch_size, use_her, her_ratio=0.8, distance_threshold=0.15):
        batch = dict(states=[], actions=[], next_states=[], rewards=[], dones=[])
        for _ in range(batch_size):
            traj = random.sample(self.buffer, 1)[0]
            step = np.random.randint(traj.length)
            state = traj.states[step]
            action = traj.actions[step]
            next_state = traj.states[step + 1]
            reward = traj.rewards[step]
            done = traj.dones[step]

            if use_her and np.random.uniform() <= her_ratio:
                step_goal = np.random.randint(step + 1, traj.length + 1)
                goal = traj.states[step_goal][:2]
                dist = np.sqrt(np.sum(np.square(next_state[:2] - goal)))
                reward = 0 if dist <= distance_threshold else 0
                done = True if dist <= distance_threshold else False
                state = np.hstack((state[:2], goal))
                next_state = np.hstack((next_state[:2], goal))

            batch["states"].append(state)
            batch["actions"].append(action)
            batch["next_states"].append(next_state)
            batch["rewards"].append(reward)
            batch["dones"].append(done)

        return batch


if __name__ == '__main__':
    """
    Hyperparameter settings
    """
    gamma = 0.98
    tau = 0.005
    lr_actor = 1e-3
    lr_critic = 1e-3
    hidden_dim = 128
    state_dim = 4
    action_dim = 2
    action_bound = 1
    episodes = 2000
    buffer_size = 10000
    minimal_traj = 200
    n_train = 20
    batch_size = 256
    sigma = 0.1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    """
    Codings
    """
    env = WorldEnv()
    replay_buffer = ReplayBuffer_Trajectory(buffer_size)
    agent = DDPG_HER(state_dim, action_dim, hidden_dim, gamma, tau, action_bound, device, lr_actor, lr_critic, sigma)
    return_list = []
    for i in range(10):
        with tqdm(total=int(episodes / 10), desc=f"Iteration {i}") as pbar:
            for i_episodes in range(int(episodes / 10)):
                episode_return = 0
                state = env.reset()
                done = False
                traj = Trajectory(state)
                while not done:
                    action = agent.take_action(state)
                    state, reward, done = env.step(action)
                    traj.store_step(state, action, reward, done)
                    episode_return += reward
                replay_buffer.add_trajectory(traj)
                return_list.append(episode_return)

                if replay_buffer.size() >= minimal_traj:
                    for _ in range(n_train):
                        trail_info = replay_buffer.sample(batch_size, True)
                        agent.update(trail_info)
                if (i_episodes + 1) % 10 == 0:
                    pbar.set_postfix({
                        "episode": f"{i * episodes / 10 + i_episodes + 1}",
                        "return": np.mean(return_list[-10:])
                    })
                pbar.update(1)

    plt.plot(return_list)
    plt.xlabel("Episodes")
    plt.ylabel("Reward Sum")
    plt.show()

    pass
