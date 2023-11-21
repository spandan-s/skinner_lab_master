from copy import deepcopy
from icecream import ic

import numpy as np
from matplotlib import pyplot as plt

from prm_htest import hypothesis_test


def crossover(p1, p2):
    '''
    produces crossover offspring of two parents by selecting a split point
    in their sequence.
    e.g. if the parents have the sequences:
    p1 = 00000
    p2 = 11111,
    a possible offspring sequence would be:
    c1 = 00111, or
    c2 = 11000
    '''

    len_seq = len(p1)
    if len(p2) != len_seq:
        raise Exception("Sequences of Different Lengths")

    split_point = np.random.randint(0, len_seq)
    c1 = np.zeros(len_seq)
    try:
        c1[:split_point] = p1[:split_point]
        c1[split_point:] = p2[split_point:]
    except TypeError:
        print(split_point, c1, p1, p2)
    return np.array(c1)


def mutate(seq):
    '''
    mutate a given sequence at loci
    the number of loci where mutations occur is a poisson variable (exp=2)
    the mutated value at the loci is the original value * a random number from a normal distribution about 1 (abs)
    '''
    seq = deepcopy(seq)
    len_seq = len(seq)
    mutation_rate = 2
    num_mutations = np.random.poisson(mutation_rate)
    # print(num_mutations)

    mutation_locs = np.random.randint(0, len_seq, size=num_mutations)

    for loc in mutation_locs:
        seq[loc] *= np.abs(np.random.normal(1, 1))
        # seq[loc] = np.random.randint(0, 2)
    return np.array(seq)


def conns_to_list(conn):
    v = []
    for c in [*conn]:
        v.append([*conn[c].values()])
    v = np.array(v).reshape(16)
    return v


def list_to_conns(v):
    v = np.reshape(v, (4, 4))

    conns_list = ["pyr", "bic", "pv", "cck"]
    conns = dict.fromkeys(conns_list)
    for i, conn1 in enumerate(conns_list):
        conns[conn1] = dict.fromkeys(conns_list)
        for j, conn2 in enumerate(conns_list):
            conns[conn1][conn2] = v[i][j]

    return conns


def run_gen_alg(f_v, n_0=2, max_iter=20):
    valid_list = []

    for it in range(max_iter):
        temp_list = []
        if it % 5 == 0:
            print(f"{it} Iterations Completed")
        # if there are fewer valid sets than the number of starting sets, fill the temp_list
        if len(valid_list) < n_0:
            for i in range(n_0):
                temp_list.append(gen_possible_conns())

        for item in temp_list:
            if f_v(item):
                if not in_Array(item, valid_list):
                    valid_list.append(item)
                    print(f"{len(valid_list)} Valid Items Found")

            child = crossover(mutate(item), item)
            if f_v(child):
                if not in_Array(child, valid_list):
                    valid_list.append(child)
                    print(f"{len(valid_list)} Valid Items Found")

        for item in valid_list:
            mate_n = np.random.randint(0, len(valid_list))
            mate = valid_list[mate_n]

            child = mutate(crossover(item, mate))
            if f_v(child):
                if not in_Array(child, valid_list):
                    valid_list.append(child)
                    print(f"{len(valid_list)} Valid Items Found")

        # if (len(valid_list) % 10 == 0) and (len(valid_list) > 0):
        #     print(f"{len(valid_list)} Valid Items Found")

        if len(valid_list) >= 100:
            return valid_list

    return valid_list


def validity_criteria(x):
    x_h = list_to_conns(x)
    return hypothesis_test(x_h)


def in_Array(array_to_check, array_list):
    return np.sum([np.array_equal(array_to_check, l) for l in array_list]) > 0


def gen_possible_conns() -> object:
    # A = {
    #     "pyr": {
    #         "pyr": 0.05, "bic": 0.04, "pv": 0.02, "cck": 0.0
    #     },
    #     "bic": {
    #         "pyr": -0.02, "bic": 0.0, "pv": 0.0, "cck": 0.0
    #     },
    #     "pv": {
    #         "pyr": -0.03, "bic": 0.0, "pv": -0.055, "cck": -0.075
    #     },
    #     "cck": {
    #         "pyr": -0.05, "bic": 0.0, "pv": -0.15, "cck": -0.15
    #     },
    # }
    CR = {
        "pyr": {
            "pyr": [0.1, 0.2], "bic": [0.1, 0.2], "pv": [0.01, 0.06], "cck": [0.0, 0.0]
        },
        "bic": {
            "pyr": [-0.1, -0.01], "bic": [0.0, 0.0], "pv": [0.0, 0.0], "cck": [0.0, 0.0]
        },
        "pv": {
            "pyr": [-0.1, -0.01], "bic": [0.0, 0.0], "pv": [-0.15, -0.03], "cck": [-0.1, -0.03]
        },
        "cck": {
            "pyr": [-0.5, -0.05], "bic": [0.0, 0.0], "pv": [-0.5, -0.05], "cck": [-0.5, -0.05]
        },
    }

    IR = {
        "pyr": [0.01, 0.1],
        "bic": [-1.85, -0.5],
        "pv": [-1.0, -0.4],
        "cck": [-1.0, -0.4]
    }

    conns_list = ["pyr", "bic", "pv", "cck"]
    new_conns = dict.fromkeys(conns_list)
    new_i = dict.fromkeys(conns_list)
    for i, conn1 in enumerate(conns_list):
        new_conns[conn1] = dict.fromkeys(conns_list)
        for j, conn2 in enumerate(conns_list):
            new_conns[conn1][conn2] = np.random.uniform(*(CR[conn1][conn2]))

    for cell in conns_list:
        new_i[cell] = np.random.uniform(*(IR[cell]))

    return conns_to_list(new_conns), new_i


# V = run_gen_alg(validity_criteria, max_iter=200, n_0=20)
# for item in V:
#     print(item)

# a1 = np.random.randint(0, 2, size=16)
# a2 = np.random.randint(0, 2, size=16)
#
# c = mutate(crossover(a1, a2))
# print(c)

I = {
    "pyr": 0.9,
    "bic": -1.25,
    "pv": 0.7,
    "cck": 0.8,
}


conns, i = gen_possible_conns()
ic(list_to_conns(conns), i)

ic(hypothesis_test(list_to_conns(conns), i, plot=True))

plt.show()
