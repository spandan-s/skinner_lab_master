import json

import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from prm_v2 import *
from prm_htest import hypothesis_test

rng = np.random.default_rng()

def gen_search(ref_conns, max_iter=100):
    valid_conns = []
    # start at a given point in the parameter space (let: reference parameter)
    #   iniitialise PRM with conns
    #   run a simulation and do spectral analysis
    #   perform hypothesis test (should be true if ref set)
    hypothesis_test(ref_conns)
    #   append conns to set of valid conns
    valid_conns.append(ref_conns)

    for i in range(max_iter):
        if i%(max_iter//10) == 0:
            print(f"iteration {i}")
        # loop through set of valid conns
        for conn in valid_conns:
            #   for each VC, change one of the conns by a random value (within constraints)
            new_conn = change_conns(conn)
            #   initialise PRM with new conns
            #   run simulation and do spectral analysis
            #   perform hypothesis test
            conn_validity = hypothesis_test(new_conn)
            #       if true: add to set of valid conns
            if conn_validity:
                if new_conn not in valid_conns:
                    valid_conns.append(new_conn)
                    with open("search_results.json", "w") as w:
                        json.dump(valid_conns, w)
                    print(f"{len(valid_conns)} valid configurations found")

    # run for either a set number of configurations OR till a certain number of valid configurations are found
    return valid_conns

def change_conns(conns, I):
    nrng = np.random.default_rng()

    c_list = ["pyr", "bic", "pv", "cck"]
    new_conns = deepcopy(conns)
    new_I = deepcopy(I)

    while new_conns == conns:
        n_mutations = rng.poisson(2)
        for i in range(n_mutations):
            if bool(rng.integers(0, 2)):
                c1, c2 = c_list[nrng.integers(4)], c_list[nrng.integers(4)]
                new_conns[c1][c2] *= np.linspace(0.5, 1.5, 10)[nrng.integers(10)]
        else:
            c1 = c_list[nrng.integers(4)]
            i_constraints = {
                "pyr": [0, 0.1],
                "bic": [-2, -1],
                "pv": [0.5, 1],
                "cck": [0.25, 1]
            }
            new_I[c1] = nrng.uniform(*i_constraints[c1])

        # print(new_conns == conns)

    return new_conns, new_I

# Parameters
T = 8.0  # total time (units in sec)
dt = 0.001  # plotting and Euler timestep (parameters adjusted accordingly)
fs = 1 / dt

# FI curve
beta = 10
tau = 5
h = 0
# r_o = 30

c_list = ["pyr", "bic", "pv", "cck"]

ref_prm = PRM_v2()

#
# for i in range(10):
#     c1, c2 = c_list[rng.integers(4)], c_list[rng.integers(4)]
#     # print(c1, c2)
#     new_conns["pyr"]["bic"] *= [0.5, 2][rng.integers(2)]
#     print(new_conns == conns)


# new_conn = change_one_conn(ref_prm.conns)
# print(new_conn)

search_results = gen_search(ref_prm.conns)

# with open("search_results.json", "w") as final:
#     json.dump(search_results, final)


# print(change_conns(ref_prm.conns))
