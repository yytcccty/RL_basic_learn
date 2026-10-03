import numpy as np
import matplotlib.pyplot as plt
import gym
from Basic_DRL.SAC import ActorContinuous, CriticContinuous, SAC_Continuous
import torch
import torch.nn.functional as F
import rl_utils
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CQL:
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound, tau, lr_actor, lr_critic, lr_alpha,
                 target_entropy, gamma, beta, num_random):
        self.actor = ActorContinuous(state_dim, hidden_dim, action_dim, action_bound).to(device)
        self.actor_optimizer = torch.optim.Adam(params=self.actor.parameters(), lr=lr_actor)

        self.critic_1 = CriticContinuous(state_dim, hidden_dim, action_dim).to(device)
        self.critic_2 = CriticContinuous(state_dim, hidden_dim, action_dim).to(device)
        self.critic_1_optimizer = torch.optim.Adam(params=self.critic_1.parameters(), lr=lr_critic)
        self.critic_2_optimizer = torch.optim.Adam(params=self.critic_2.parameters(), lr=lr_critic)

        self.critic_1_target = CriticContinuous(state_dim, hidden_dim, action_dim).to(device)
        self.critic_2_target = CriticContinuous(state_dim, hidden_dim, action_dim).to(device)
        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())

        self.log_alpha = torch.tensor(np.log(0.01), dtype=torch.float, requires_grad=True)
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr_alpha)
        self.target_entropy = target_entropy

        self.tau = tau
        self.gamma = gamma
        self.beta = beta
        self.num_random = num_random

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(device)
        action = self.actor(state)[0]
        return [action.item()]

    def soft_update(self, net, target_net):
        for para, old_para in zip(net.parameters(), target_net.parameters()):
            old_para.data.copy_(para.data * self.tau + old_para.data * (1 - self.tau))

    def update(self, trail_info):
        states = torch.tensor(np.array(trail_info['states']), dtype=torch.float).to(device)
        actions = torch.tensor(np.array(trail_info['actions']), dtype=torch.float).view(-1, 1).to(device)
        rewards = torch.tensor(np.array(trail_info['rewards']), dtype=torch.float).view(-1, 1).to(device)
        next_states = torch.tensor(np.array(trail_info['next_states']), dtype=torch.float).to(device)
        dones = torch.tensor(np.array(trail_info['dones']), dtype=torch.float).view(-1, 1).to(device)
        rewards = (rewards + 8.0) / 8.0
        next_actions, log_probs = self.actor(next_states)
        q_value = torch.min(self.critic_1_target(next_states, next_actions),
                            self.critic_2_target(next_states, next_actions))
        td_target = rewards + self.gamma * (q_value - self.log_alpha.exp() * log_probs) * (1 - dones)
        critic_loss_1 = F.mse_loss(td_target.detach(), self.critic_1(states, actions))
        critic_loss_2 = F.mse_loss(td_target.detach(), self.critic_2(states, actions))
        # Update critic networks
        batch_size = states.shape[0]
        random_unif_actions = torch.rand([batch_size * self.num_random, actions.shape[-1]], dtype=torch.float).uniform_(
            -1, 1).to(device)
        random_unif_log_pi = np.log(0.5 ** next_actions.shape[-1])

        tmp_states = states.unsqueeze(1).repeat(1, self.num_random, 1).view(-1, states.shape[-1])
        tmp_next_states = next_states.unsqueeze(1).repeat(1, self.num_random, 1).view(-1, states.shape[-1])

        random_curr_actions, random_curr_log_pi = self.actor(tmp_states)
        random_next_actions, random_next_log_pi = self.actor(tmp_next_states)

        q1_unif = self.critic_1(tmp_states, random_unif_actions).view(-1, self.num_random, 1)
        q1_curr = self.critic_1(tmp_states, random_curr_actions).view(-1, self.num_random, 1)
        q1_next = self.critic_1(tmp_states, random_next_actions).view(-1, self.num_random, 1)
        q2_unif = self.critic_2(tmp_states, random_unif_actions).view(-1, self.num_random, 1)
        q2_curr = self.critic_2(tmp_states, random_curr_actions).view(-1, self.num_random, 1)
        q2_next = self.critic_2(tmp_states, random_next_actions).view(-1, self.num_random, 1)

        q1_cat = torch.cat([
            q1_unif - random_unif_log_pi,
            q1_curr - random_curr_log_pi.detach().view(-1, self.num_random, 1),
            q1_next - random_next_log_pi.detach().view(-1, self.num_random, 1)
        ], dim=1)
        q2_cat = torch.cat([
            q2_unif - random_unif_log_pi,
            q2_curr - random_curr_log_pi.detach().view(-1, self.num_random, 1),
            q2_next - random_next_log_pi.detach().view(-1, self.num_random, 1)
        ], dim=1)

        qf1_loss_1 = torch.logsumexp(q1_cat, dim=1).mean()
        qf1_loss_2 = self.critic_1(states, actions).mean()
        qf2_loss_1 = torch.logsumexp(q2_cat, dim=1).mean()
        qf2_loss_2 = self.critic_2(states, actions).mean()
        qf1_loss = self.beta * (qf1_loss_1 - qf1_loss_2) + critic_loss_1
        qf2_loss = self.beta * (qf2_loss_1 - qf2_loss_2) + critic_loss_2

        self.critic_1_optimizer.zero_grad()
        qf1_loss.backward(retain_graph=True)
        self.critic_1_optimizer.step()

        self.critic_2_optimizer.zero_grad()
        qf2_loss.backward(retain_graph=True)
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


if __name__ == '__main__':
    lr_actor = 3e-4
    lr_critic = 3e-3
    lr_alpha = 3e-4
    gamma = 0.99

    env_name = 'Pendulum-v1'  # CartPole Pendulum
    hidden_dim = 128
    episodes = 100
    buffer_size = 100000
    batch_size = 64
    minimal_size = 1000
    tau = 0.005
    seed = 42

    env = gym.make(env_name)

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    env.reset(seed=seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    state_dim = env.observation_space.shape[0]
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

    num_epochs = 100
    num_trains_per_epoch = 500
    num_random = 5
    beta = 5.0

    retn_list = []
    agent = CQL(state_dim, hidden_dim, action_dim, action_bound, tau, lr_actor, lr_critic, lr_alpha, target_entropy,
                gamma, beta, num_random)
    for i in range(10):
        with tqdm(total=int(num_epochs / 10), desc="Iteration %d" % i) as pbar:
            for i_epoch in range(int(num_epochs / 10)):
                epoch_return = 0
                state, _ = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_action, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    epoch_return += reward
                    state = next_action
                retn_list.append(epoch_return)

                for _ in range(num_trains_per_epoch):
                    b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                    trail_info = {
                        'states': b_s,
                        'actions': b_a,
                        'next_states': b_ns,
                        'rewards': b_r,
                        'dones': b_d
                    }
                    agent.update(trail_info)
                if (i_epoch + 1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_epochs / 10 * i + i_epoch + 1),
                                      'return': '%.3f' % np.mean(retn_list[-10:])})
                pbar.update(1)

    plt.plot(retn_list)
    plt.xlabel('Episodes')
    plt.ylabel('Reward Sum')
    plt.grid(True)
    plt.show()
