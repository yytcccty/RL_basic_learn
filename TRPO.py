import numpy as np

if not hasattr(np, "bool8"):
    np.bool8 = np.bool_
import matplotlib.pyplot as plt
import copy
import gym
import rl_utils
import torch
import torch.nn.functional as F


class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=-1)
        return x


class PolicyNetContinuous(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNetContinuous, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc_mu = torch.nn.Linear(hidden_dim, action_dim)
        self.fc_sigma = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        mu = 2.0 * torch.tanh(self.fc_mu(x))
        sigma = F.softplus(self.fc_sigma(x))
        return mu, sigma


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
    """
    Suitable for discrete action
    """
    def __init__(self, state_dim, action_dim, hidden_dim, kl_delta, gamma, alpha, lr_critic, lamda, device):
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
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

        :param old_log_prob: The :math:`\log{\pi_{\theta_k}(a \mid s)}` term.
        :param advantages: The :math:`A^{\pi_{\theta_k}}(s, a)` term.
        :param actions: Actions taken by the agent.
        :param states: States visited by the agent.
        :param actor: The :math:`\theta\ \ '` network.
        :return: The estimation of :math:`\dfrac{\pi_{\theta\ \ '}(a \mid s)}{\pi_{\theta_k}(a \mid s)} A^{\pi_{\theta_k}}(s,a)`.
        """
        log_probs = torch.log(actor(states).gather(1, actions))
        ration = torch.exp(log_probs - old_log_prob)
        return torch.mean(ration * advantages)

    def Hessian_vector_product(self, states, old_action_dist, vector):
        r"""
        Calculate :math:`Hp` where :math:`H=\mathbb{E}[D_{KL}(\pi_{\theta_k}(\cdot\mid s), \pi_{\theta\ '}(\cdot\mid s))]`
        :param states: States visited by the agent.
        :param old_action_dist: The :math:`\pi_{\theta_k}(\cdot\mid s)` term.
        :param vector: Vector :math:`p`.
        :return: The :math:`Hp` term.
        """
        new_action_dist = torch.distributions.Categorical(self.actor(states))
        kl = torch.mean(torch.distributions.kl.kl_divergence(old_action_dist, new_action_dist))
        grad1 = torch.autograd.grad(kl, self.actor.parameters(), create_graph=True)
        grad1_vector = torch.cat([grad.view(-1) for grad in grad1])
        grad1_vector_product = torch.dot(grad1_vector, vector)
        grad2 = torch.autograd.grad(grad1_vector_product, self.actor.parameters())
        grad2_vector = torch.cat([grad.view(-1) for grad in grad2])
        return grad2_vector

    def CG(self, g, states, old_action_dist):
        """
        Compute :math:`x=H^{-1}g` using Conjugate Gradient method
        :param g: Vector :math:`g`
        :param states: States visited by the agent.
        :param old_action_dist: The :math:`\pi_{\theta_k}(\cdot\mid s)` term.
        :return: :math:`x` solved by CG
        """
        x = torch.zeros_like(g)
        d = r = g.clone()
        r_dot = torch.dot(r, r)
        for _ in range(10):
            if r_dot <= 1e-10:
                break
            Hd = self.Hessian_vector_product(states, old_action_dist, d)
            alpha = r_dot / torch.dot(d, Hd)
            x += alpha * d
            r -= alpha * Hd
            beta = torch.dot(r, r) / r_dot
            d = r + beta * d
            r_dot = torch.dot(r, r)
        return x

    def line_search(self, search_vector, states, old_action_dist, old_log_prob, advantages, actions):
        r"""
        Find parameters to update actor network using the formula:

        :math:`\theta\ '=\theta_k+ \alpha^i \sqrt{\frac{2\delta}{x^T Hx}}x`,
        where :math:`i` is the minimal integer making :math:`\theta\ '` better than :math:`\theta_k`
        while satisfying KL constraint.
        :param search_vector: The :math:`\sqrt{\frac{2\delta}{x^T Hx}}` term.
        :param states: States visited by the agent.
        :param old_action_dist: The :math:`\pi_{\theta_k}(\cdot\mid s)` term.
        :param old_log_prob: The :math:`\log{\pi_{\theta_k}(a \mid s)}` term.
        :param advantages: The :math:`A^{\pi_{\theta_k}}(s, a)` term.
        :param actions: Actions taken by the agent.
        :return: The :math:`\theta\ '` term.
        """
        old_paras = torch.nn.utils.convert_parameters.parameters_to_vector(self.actor.parameters())
        old_obj = self.compute_surrogate_obj(old_log_prob, advantages, actions, states, self.actor)
        for i in range(15):
            new_actor = copy.deepcopy(self.actor)
            new_paras = old_paras + self.alpha ** i * search_vector
            torch.nn.utils.convert_parameters.vector_to_parameters(new_paras, new_actor.parameters())
            new_obj = self.compute_surrogate_obj(old_log_prob, advantages, actions, states, new_actor)
            new_action_dist = torch.distributions.Categorical(new_actor(states))
            kl_dist = torch.mean(torch.distributions.kl.kl_divergence(old_action_dist, new_action_dist))
            if new_obj > old_obj and kl_dist < self.kl_delta:
                return new_paras
        return old_paras

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
        x = self.CG(g, states, old_action_dist)

        # Update actor using Line Search
        Hx = self.Hessian_vector_product(states, old_action_dist, x)
        search_vector = torch.sqrt(2 * self.kl_delta / (torch.dot(x, Hx) + 1e-8)) * x
        new_paras = self.line_search(search_vector, states, old_action_dist, old_pi_log, advan_GAE, actions)
        torch.nn.utils.convert_parameters.vector_to_parameters(new_paras, self.actor.parameters())


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
