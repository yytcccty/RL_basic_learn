import gym
from MDP_DP_Solu import PolicyIter, ValueIter, printPolicy

if __name__ == '__main__':
    env = gym.make("FrozenLake-v1")
    env = env.unwrapped
    env.render()

    holes = set()
    ends = set()
    for s in env.P:
        for a in env.P[s]:
            for s_ in env.P[s][a]:
                if s_[2] == 1.0:
                    ends.add(s_[1])
                if s_[3] is True:
                    holes.add(s_[1])
    holes = holes - ends
    print('Index of ice caves: ', holes)
    print('Index of target: ', ends)

    for a in env.P[14]:
        print(env.P[14][a])

    gamma = 0.5
    theta = 1e-3

    agent = PolicyIter(env, gamma, theta)
    agent.policy_iter()
    # agent = ValueIter(env, gamma, theta)
    # agent.value_iter()

    printPolicy(agent, [5, 7, 11, 12], [15])
