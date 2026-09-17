import numpy as np


def ReturnCal(r_list, chains, gamma):  # Calculate the return/G of a given specific state chain
    G = 0
    for ele in reversed(chains):
        G = G * gamma + r_list[ele - 1]
    return G


def V_cal(gamma, P, r_list):  # Calculate the value functions/V of a given Markov reward process
    R = np.array(r_list).reshape((-1, 1))
    V = np.linalg.inv(np.eye(P.shape[0]) - gamma*P)@R
    return V


if __name__ == '__main__':
    """
    Hyperparameter Settings
    """
    P = np.array([
        [0.9, 0.1, 0.0, 0.0, 0.0, 0.0],
        [0.5, 0.0, 0.5, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.6, 0.0, 0.4],
        [0.0, 0.0, 0.0, 0.0, 0.3, 0.7],
        [0.0, 0.2, 0.3, 0.5, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    ])
    reward_list = [-1, -2, -2, 10, 1, 0]
    gamma = 0.5

    """
    Codes
    """
    # chains = [1, 2, 3, 6]
    # G = ReturnCal(reward_list, chains, gamma)
    # print(f'Calculated Gain is {G}')
    V = V_cal(gamma, P, reward_list)
    print(V)

