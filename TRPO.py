import numpy as np

if not hasattr(np, "bool8"):
    np.bool8 = np.bool_
import matplotlib.pyplot as plt
import copy
import gym
import rl_utils
import torch
import torch.nn.functional as F


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


def compute_GAE(lamda, gamma, td_error):
    """
    Using Generalized Advantage Estimation (GAE) to calculate advantages of the sampled path
    :param lamda: Hyperparameter of GAE
    :param gamma: Hyperparameter of RL
    :param td_error: The temporal difference error in the sampled path
    :return: Generalized advantage of all states in the sampled path
    """
    td_delta = td_error.detach().numpy()
    advan_GAE_list = []
    tmp_GAE = 0.
    for delta in td_delta[::-1]:
        tmp_GAE = tmp_GAE * gamma * lamda + delta
        advan_GAE_list.append(tmp_GAE)
    advan_GAE_list.reverse()
    return torch.tensor(np.array(advan_GAE_list), dtype=torch.float)


class TRPO:
    def __init__(self, state_dim, action_dim, hidden_dim, kl_delta, gamma, alpha, lr_critic, lamda, device):
        self.actor = Policy_net(state_dim, hidden_dim, action_dim).to(device)
        self.critic = Value_net(state_dim, hidden_dim).to(device)
        self.kl_delta = kl_delta
        self.gamma = gamma
        self.alpha = alpha
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.lamda = lamda
        self.device = device

    def take_action(self, state):
        state = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)
        prob_dist = self.actor(state)
        action_dist = torch.distributions.Categorical(prob_dist)
        action = action_dist.sample()
        return action.item()

    def compute_surrogate_obj(self, old_log_prob, advantages, actions, states, actor):
        """
        Compute the estimation of the surrogate objective.

        :param old_log_prob: The :math:`\pi_{\theta_k}(a \mid s)` term.
        :param advantages: The :math:`A^{\pi_{\theta_k}}(s, a)` term.
        :param actions: Actions taken by the agent.
        :param states: States visited by the agent.
        :param actor: The :math:`\theta\ \ '` network.
        :return: The estimation of :math:`\dfrac{\pi_{\theta\ \ '}(a \mid s)}{\pi_{\theta_k}(a \mid s)} A^{\pi_{\theta_k}}(s,a)`.
        """
        log_probs = torch.log(actor(states).gather(1, actions))
        ration = torch.exp(log_probs - old_log_prob)
        return torch.mean(ration * advantages)

    def Hessian_vector_product(self,states, old_action_dist, vector):


    def CG(self, g, H):

    def update(self, trail_info):
        states = torch.tensor(np.array(trail_info['states']), dtype=torch.float).to(self.device)
        actions = torch.tensor(np.array(trail_info['actions'])).view(-1, 1).to(self.device)
        rewards = torch.tensor(np.array(trail_info['rewards']), dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(np.array(trail_info['next_states']), dtype=torch.float).to(self.device)
        dones = torch.tensor(np.array(trail_info['dones']), dtype=torch.float).view(-1, 1).to(self.device)

        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)

        """
        Update critic network
        """
        critic_loss = F.mse_loss(td_target.detach(), self.critic(states))
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        """
        Update actor network
        """
        td_error = td_target - self.critic(states)  # advantages of each state in the trail
        advan_GAE = compute_GAE(self.lamda, self.gamma, td_error.cpu()).to(device)
        old_pi_log = torch.log(self.actor(states).gather(1, actions)).detach()
        old_action_dist = torch.distributions.Categorical(self.actor(states).detach())

        surrogate_obj = self.compute_surrogate_obj(old_pi_log, advan_GAE, actions, states, self.actor)
        # calculate gradient g
        grads = torch.autograd.grad(surrogate_obj, self.actor.parameters())
        g = torch.cat([grad.view(-1) for grad in grads]).detach()

        # compute x=H^-1*g using CG

        pass


if __name__ == '__main__':
    """
    Hyperparameter Settings:
    """
    gamma = 0.98
    alpha = 0.5
    lamda = 0.95
    lr_critic = 1e-2
    kl_delta = 5e-4
    episodes = 500
    hidden_dim = 128
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env_name = "CartPole-v1"
    """
    Codings
    """
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = TRPO(state_dim, action_dim, hidden_dim, kl_delta, gamma, alpha, lr_critic, lamda, device)
    retn_list = rl_utils.train_on_policy_agent(env, agent, episodes)

    plt.plot(retn_list)
    plt.xlabel("Episodes")
    plt.ylabel("Reward Sum")
    plt.show()

    mv_return = rl_utils.moving_average(retn_list, 9)
    plt.plot(mv_return)
    plt.xlabel("Episodes")
    plt.ylabel("Converted Reward")
    plt.show()

    pass
