import numpy as np
import copy
import numpy.random as random


def MyJoin(str1, str2):
    return str1 + '-' + str2


def Sample(MDP, pi, sample_num, seq_len):  # Sample function for Monte-Carlo method
    """
    :param MDP: MDP=(S, A, P, R, gamma)
    :param pi:  Given policy
    :param sample_num:  Sample number
    :param seq_len: Sample sequence length
    :return: A list include all the sampled sequences
    """
    sampled_paths = []
    for _ in range(sample_num):
        cur_s = MDP[0][random.randint(4)]
        cur_len = 0
        path = []
        while cur_s != 's5' and cur_len < seq_len:
            tmp = 0.
            thre = random.rand()
            for a in MDP[1]:
                tmp += pi.get(MyJoin(cur_s, a), 0)
                if tmp >= thre:
                    ttmp = 0.
                    tthre = random.rand()
                    for tmp_s in MDP[0]:
                        ttmp += MDP[2].get(MyJoin(MyJoin(cur_s, a), tmp_s), 0)
                        if ttmp >= tthre:
                            path.append((cur_s, a, tmp_s, MDP[3].get(MyJoin(cur_s, a))))
                            cur_s = tmp_s
                            break
                    break
            cur_len += 1
        sampled_paths.append(path)

    return sampled_paths


def CalV_EV(samp_paths, gamma, N, V):  # Calculate V in an 'Every-visit' manner
    for one_path in samp_paths:
        tmp_len = len(one_path)
        G = 0.
        for i in range(tmp_len - 1, -1, -1):
            N[one_path[i][0]] += 1
            G = one_path[i][3] + gamma * G
            V[one_path[i][0]] += (G - V[one_path[i][0]]) / N[one_path[i][0]]
    """In a 'First-visit' manner"""


def CalV_FV(samp_paths, gamma, N, V, S):  # Calculate V in an 'First-visit' manner
    for one_path in samp_paths:
        tmp_len = len(one_path)
        G = 0.
        FV_t = {}
        for c in range(tmp_len):
            (s, a, s_next, r) = one_path[c]
            if s not in FV_t:
                FV_t[s] = c

        for i in range(tmp_len - 1, -1, -1):
            (s, a, s_next, r) = one_path[i]
            G = r + gamma * G
            if i == FV_t[s]:
                N[s] += 1
                V[s] += (G - V[s]) / N[s]


def CalOccupancy(s, a, samp_paths, gamma, max_seq_len):  # Calculate 'Occupancy Measure' of a state-action pair (s, a)
    rho = 0.
    t_times = np.zeros(max_seq_len)
    happen_times = np.zeros(max_seq_len)
    for one_path in samp_paths:
        tmp_len = len(one_path)
        for i in range(tmp_len):
            t_times[i] += 1
            (s_tmp, a_tmp, s_next, r) = one_path[i]
            if (s, a) == (s_tmp, a_tmp):
                happen_times[i] += 1

    for i in range(max_seq_len):
        if t_times[i]:
            rho += gamma ** i * happen_times[i] / t_times[i]
    rho *= (1 - gamma)
    return rho


if __name__ == '__main__':
    """
    Hyperparameter settings
    """
    S = [f's{ele + 1}' for ele in range(5)]
    A = ['Keep s1', 'Goto s2', 'Goto s1', 'Goto s3', 'Goto s4', 'Goto s5', 'Prob go']
    P = {
        's1-Keep s1-s1': 1.0,
        's1-Goto s2-s2': 1.0,
        's2-Goto s1-s1': 1.0,
        's2-Goto s3-s3': 1.0,
        's3-Goto s4-s4': 1.0,
        's3-Goto s5-s5': 1.0,
        's4-Goto s5-s5': 1.0,
        's4-Prob go-s2': 0.2,
        's4-Prob go-s3': 0.4,
        's4-Prob go-s4': 0.4,
    }
    R = {
        's1-Keep s1': -1,
        's1-Goto s2': 0,
        's2-Goto s1': -1,
        's2-Goto s3': -2,
        's3-Goto s4': -2,
        's3-Goto s5': 0,
        's4-Goto s5': 10,
        's4-Prob go': 1,
    }
    gamma = 0.5
    Pi_1 = {
        's1-Keep s1': 0.5,
        's1-Goto s2': 0.5,
        's2-Goto s1': 0.5,
        's2-Goto s3': 0.5,
        's3-Goto s4': 0.5,
        's3-Goto s5': 0.5,
        's4-Goto s5': 0.5,
        's4-Prob go': 0.5,
    }
    Pi_2 = {
        's1-Keep s1': 0.6,
        's1-Goto s2': 0.4,
        's2-Goto s1': 0.3,
        's2-Goto s3': 0.7,
        's3-Goto s4': 0.5,
        's3-Goto s5': 0.5,
        's4-Goto s5': 0.1,
        's4-Prob go': 0.9,
    }

    """
    Codes
    """
    MDP = (S, A, P, R, gamma)

    """Calculate V^Pi(s) based on marginalization method, turning MDP into MRP"""

    R_prime_Pi_1 = {ele: 0. for ele in S}
    P_prime_Pi_1 = np.zeros((5, 5))
    for pair in R:
        [s, a] = pair.split('-')
        R_prime_Pi_1[s] += Pi_1[pair] * R[pair]
    print(f'\nR\'={R_prime_Pi_1}')

    for pair in P:
        [s, a, s_prime] = pair.split('-')
        P_prime_Pi_1[int(s[1]) - 1, int(s_prime[1]) - 1] += Pi_1[MyJoin(s, a)] * P[pair]
    tmp = np.where(np.isclose(P_prime_Pi_1.sum(axis=1), 0.))[0]
    P_prime_Pi_1[tmp, tmp] = 1.0
    if not np.all(np.isclose(P_prime_Pi_1.sum(axis=1), 1.0)):
        print('\n\033[1;31mInvalid P matrix !\033[0m')
    else:
        print(f'\nP\'={P_prime_Pi_1}')

    R_vec = np.zeros((5, 1))
    for ele in R_prime_Pi_1:
        R_vec[int(ele[1]) - 1] = R_prime_Pi_1[ele]
    V_pi = np.linalg.inv(np.eye(P_prime_Pi_1.shape[0]) - gamma * P_prime_Pi_1) @ R_vec
    print(f'\nV={V_pi}')

    Q_pi = copy.deepcopy(R)
    for pair in Q_pi:
        for pairs in P:
            [s, a, s_prime] = pairs.split('-')
            if MyJoin(s, a) == pair:
                Q_pi[pair] += gamma * P[pairs] * V_pi[int(s_prime[1]) - 1]
    print('\nQ^pi=')
    for pair in Q_pi:
        print(f'{pair} -> {Q_pi[pair]}')

    """Calculate V^Pi(s) based on Monte-Carlo method"""

    MC_samples = Sample(MDP, Pi_1, sample_num=1000, seq_len=50)
    print(f'\nFirst sequence: {MC_samples[0]}')
    print(f'Second sequence: {MC_samples[1]}')
    print(f'Fifth sequence: {MC_samples[4]}')

    N = {ele: 0 for ele in S}
    V = {ele: 0 for ele in S}
    CalV_FV(MC_samples, gamma, N, V, S)
    print(f'\nUsing \'First-visit\' strategy, V=\n{V}')

    N = {ele: 0 for ele in S}
    V = {ele: 0 for ele in S}
    CalV_EV(MC_samples, gamma, N, V)
    print(f'\nUsing \'Every-visit\' strategy, V=\n{V}')

    """Calculate 'Occupancy Measure' given a specific state-action pair"""
    MC_samples_1 = Sample(MDP, Pi_1, sample_num=1000, seq_len=1000)
    MC_samples_2 = Sample(MDP, Pi_2, sample_num=1000, seq_len=1000)

    rho_1 = CalOccupancy('s4', 'Prob go', MC_samples_1, gamma, 1000)
    rho_2 = CalOccupancy('s4', 'Prob go', MC_samples_2, gamma, 1000)
    print(f'\nPolicy 1 occupancy measure: {rho_1}')
    print(f'\nPolicy 2 occupancy measure: {rho_2}')
