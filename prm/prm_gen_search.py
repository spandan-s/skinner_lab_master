import numpy as np
import matplotlib.pyplot as plt

from prm_v2 import *
from prm_htest import hypothesis_test

def gen_search(ref_conns, max_iter=1000):
    valid_conns = []
    # start at a given point in the parameter space (let: reference parameter)
    #   iniitialise PRM with conns
    #   run a simulation and do spectral analysis
    #   perform hypothesis test (should be true if ref set)
    hypothesis_test(ref_conns)
    #   append conns to set of valid conns
    valid_conns.append(ref_conns)

    for i in range(max_iter):
        # loop through set of valid conns
        for conn in valid_conns:
            #   for each VC, change one of the conns by a random value (within constraints)
            new_conn = change_one_conn(conn)
            #   initialise PRM with new conns
            #   run simulation and do spectral analysis
            #   perform hypothesis test
            conn_validity = hypothesis_test(new_conn)
            #       if true: add to set of valid conns
            if conn_validity:
                valid_conns.append(new_conn)

    # run for either a set number of configurations OR till a certain number of valid configurations are found

def change_one_conn(conns):
