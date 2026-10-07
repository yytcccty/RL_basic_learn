import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import random
import rl_utils

from multiagent.environment import MultiAgentEnv
import multiagent.scenarios as scenarios


def make_env(scenario_name):
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    world = scenario.make_world()
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    return env


def onehot_from_logits(logits, eps=0.01):
    argmax_acs = (logits == logits.max(dim=1, keepdim=True)[0]).float()
    rand_acs = torch.autograd.Variable(
        torch.eye(logits.shape[1])[np.random.choice(range(logits.shape[1]), size=logits.shape[0])],
        requires_grad=False).to(logits.device)
    return torch.stack([argmax_acs[i] if r > eps else rand_acs[i] for (i, r) in enumerate(torch.rand(logits.shape[0]))])


def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape)
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature):
    y = logits + sample_gumbel(logits.shape).to(logits.device)
    return F.softmax(y / temperature, dim=1)


def gumbel_softmax(logits, temperature=1.0):
    y = gumbel_softmax_sample(logits, temperature)
    y_hard = onehot_from_logits(logits)
    y = (y_hard.to(logits.device) - y).detach() + y
    return y


class TwoLayerNet(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, hidden_size)
        self.fc3 = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class DDPG:
    def __init__(self, state_dim, action_dim, hidden_dim, critic_input_dim, device, lr_actor, lr_critic):
        self.actor = TwoLayerNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = TwoLayerNet(critic_input_dim, hidden_dim, 1).to(device)
        self.actor_target = TwoLayerNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic_target = TwoLayerNet(critic_input_dim, hidden_dim, 1).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr_critic)

    def take_action(self, state, explore=False):
        action = self.actor(state)
        if explore:
            action = gumbel_softmax(action)
        else:
            action = onehot_from_logits(action)
        return action.detach().cpu().numpy()[0]

    def soft_update(self, net, target_net, tau):
        for para, old_para in zip(net.parameters(), target_net.parameters()):
            old_para.data.copy_(para.data * tau + old_para.data * (1.0 - tau))


def evaluate(env_id, maddpg, n_episode=10, episode_length=25):
    env = make_env(env_id)
    returns = np.zeros(len(env.agents))
    for _ in range(n_episode):
        obs = env.reset()
        for t_i in range(episode_length):
            actions = maddpg.take_action(obs, explore=False)
            obs, rew, done, info = env.step(actions)
            rew = np.array(rew)
            returns += rew / n_episode
    return returns.tolist()


class MADDPG:
    def __init__(self, env, state_dims, action_dims, hidden_dim, critic_input_dim, device, lr_actor, lr_critic, tau):
        self.agents = []
        for i in range(len(env.agents)):
            self.agents.append(
                DDPG(state_dims[i], action_dims[i], hidden_dim, critic_input_dim, device, lr_actor, lr_critic))
        self.env = env
        self.device = device
        self.tau = tau

    def take_action(self, states, explore):
        states = [torch.tensor([states[i]], dtype=torch.float, device=self.device) for i in range(len(env.agents))]
        actions = [self.agents[i].take_action(states[i], explore) for i in range(len(self.env.agents))]
        return actions

    def update_all_targets(self):
        for agt in self.agents:
            agt.soft_update(agt.actor, agt.actor_target, self.tau)
            agt.soft_update(agt.critic, agt.critic_target, self.tau)

    @property
    def target_policies(self):
        return [agent.actor_target for agent in self.agents]

    def update(self, samples, i_agent):
        obs, act, rew, next_obs, done = samples
        cur_agent = self.agents[i_agent]

        cur_agent.critic.optimizer.zero_grad()
        all_target_act = [onehot_from_logits(pi(_next_obs)) for pi, _next_obs in zip(self.target_policies, next_obs)]


def stack_array(x, device):
    rearanged = [[sub_x[i] for sub_x in x] for i in range(len(x[0]))]
    return [torch.FloatTensor(np.vstack(aa)).to(device) for aa in rearanged]


if __name__ == '__main__':
    """
    Hyperparameters:
    """
    num_episodes = 5000
    episode_length = 25
    hidden_dim = 64
    actor_lr = 1e-2
    critic_lr = 1e-2
    gamma = 0.95
    tau = 0.01
    batch_size = 1024
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    buffer_size = 100000
    minimal_size = 4000
    update_interval = 100

    seed = 42
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
    env_id = "simple_adversary"
    env = make_env(env_id)
    replay_buffer = rl_utils.ReplayBuffer(buffer_size)

    state_dims = []
    action_dims = []
    for action_space in env.action_space:
        action_dims.append(action_space.n)
    for state_space in env.observation_space:
        state_dims.append(state_space.shape[0])
    critic_input_dim = sum(state_dims) + sum(action_dims)
    maddpg = MADDPG(env, state_dims, action_dims, hidden_dim, critic_input_dim, device, actor_lr, critic_lr, tau)

    return_list = []
    total_step = 0
    for i_episode in range(num_episodes):
        state = env.reset()
        for t_i in range(episode_length):
            actions = maddpg.take_action(state, explore=False)
            next_state, reward, done, _ = env.step(actions)
            replay_buffer.add(state, actions, reward, next_state, done)
            state = next_state
            total_step += 1
            if replay_buffer.size() > minimal_size and total_step % update_interval == 0:
                samples = replay_buffer.sample(batch_size)
                samples = [stack_array(x, device) for x in samples]
                for a_i in range(len(env.agents)):
                    maddpg.update(samples, a_i)
                maddpg.update_all_targets()
        if (i_episode + 1) % 100 == 0:
            ep_returns = evaluate(env_id, maddpg, n_episode=100)
            return_list.append(ep_returns)
            print("Episode {}\tReturn: {}".format(i_episode + 1, ep_returns))

    pass
