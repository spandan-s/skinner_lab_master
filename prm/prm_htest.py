from copy import deepcopy

import numpy.random

from prm_v2 import *

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def hypothesis_test(in_conns="default", in_I="default", plot=False):
    # # ===============================================================
    # new_prm = PRM_v2()
    #
    # time = np.arange(0, T, dt)
    #
    # new_prm.set_init_state(len(time))
    # new_prm = simulate(time, new_prm, dt, tau)
    # # =================================================================
    # dps_tpp = calc_spectral(new_prm.R, fs, time, 'theta', 'power')["pyr"]
    # dps_gpp = calc_spectral(new_prm.R, fs, time, 'gamma', 'power')["pyr"]
    # # =================================================================

    dps_tpp, dps_gpp = ref_power()

    conns = deepcopy(in_conns)
    I = deepcopy(in_I)

    # Checking validity of parameter set
    test_prm = PRM_v2(conns, I)
    h_test = [False] * 5
    new_conns = test_prm.conns.copy()

    test_prm.set_init_state(len(time))
    test_prm = simulate(time, test_prm, dt, tau)
    if plot:
        plot_layout = [
            ["top"]*2,
            ["midL", "midR"],
            ["botL", "botR"]
        ]

        fig, ax = plt.subplot_mosaic(plot_layout)

        ax["top"].plot(time[3 * len(time)//4:], test_prm.R["pyr"][3 * len(time)//4:])
        ax["top"].set_title("Primary Hypothesis")

    # Check if it satisfies primary hypothesis
    #   Has theta power >= 50% of default parameter set (DPS)
    #   Has gamma power >= 25% of DPS
    # print("Checking primary hypothesis")
    pH0 = np.zeros(2)
    pH0[0] = find_pyr_power(test_prm.R, fs, "theta")[1]
    pH0[1] = find_pyr_power(test_prm.R, fs, 'gamma')[1]
    pbr = pv_bic_ratio(test_prm.R)
    max_cck = np.max(test_prm.R["cck"][int(fs):])
    if (pH0[0] >= (0.6 * dps_tpp)) and (pH0[1] >= (0.6 * dps_gpp)):
        if (pbr >= 0.67) and (max_cck >= 4.2):
            h_test[0] = True
        else:
            # print("Failed 67% PV-BiC ratio threshold")
            return False

        # Check if it satisfies secondary hypothesis #1 (removal of PYR->PYR connections)
        #   Has theta power <= 10% of DPS
        # print("Checking secondary hypothesis #1")
        test_prm.conns["pyr"]["pyr"] = 0
        test_prm.set_init_state(len(time))
        test_prm = simulate(time, test_prm, dt, tau)
        if plot:
            ax["midL"].plot(time[3 * len(time) // 4:], test_prm.R["pyr"][3 * len(time) // 4:])
            ax["midL"].set_title("Removed $PYR \\rightarrow PYR$ connections")
        pH1 = find_pyr_power(test_prm.R, fs, "theta")[1]
        if pH1 <= 0.15 * dps_tpp:
            h_test[1] = True

            # Check if it satisfies secondary hypothesis #2 (removal of CCK->PV connections)
            #   Has theta power <= 10% of DPS
            # print("Checking secondary hypothesis #2")
            test_prm.set_connections(conns)
            test_prm.conns["cck"]["pv"] = 0
            test_prm.set_init_state(len(time))
            test_prm = simulate(time, test_prm, dt, tau)
            if plot:
                ax["midR"].plot(time[3 * len(time) // 4:], test_prm.R["pyr"][3 * len(time) // 4:])
                ax["midR"].set_title("Removed $CCK \\rightarrow PV$ connections")

            pH2 = find_pyr_power(test_prm.R, fs, "theta")[1]
            if pH2 <= 0.15 * dps_tpp:
                h_test[2] = True

                # Check if it satisfies secondary hypothesis #3 (removal of PV->PYR connections)
                #   Has theta power <= 10% of DPS
                # print("Checking secondary hypothesis #3")
                test_prm.set_connections(conns)
                test_prm.conns["pv"]["pyr"] = 0
                test_prm.set_init_state(len(time))
                test_prm = simulate(time, test_prm, dt, tau)
                if plot:
                    ax["botL"].plot(time[3 * len(time) // 4:], test_prm.R["pyr"][3 * len(time) // 4:])
                    ax["botL"].set_title("Removed $PV \\rightarrow PYR$ connections")

                pH3 = find_pyr_power(test_prm.R, fs, "theta")[1]
                if pH3 <= 0.15 * dps_tpp:
                    h_test[3] = True

                    # Check if it satisfies secondary hypothesis #4 (removal of BiC->PYR connections)
                    #   Has theta power <= 10% of DPS
                    # print("Checking secondary hypothesis #4")
                    test_prm.set_connections(conns)
                    test_prm.conns["bic"]["pyr"] = 0
                    test_prm.set_init_state(len(time))
                    test_prm = simulate(time, test_prm, dt, tau)
                    if plot:
                        ax["botR"].plot(time[3 * len(time) // 4:], test_prm.R["pyr"][3 * len(time) // 4:])
                        ax["botR"].set_title("Removed $BiC \\rightarrow PYR$ connections")

                    pH4 = find_pyr_power(test_prm.R, fs, "theta")[1]
                    if pH4 <= 0.15 * dps_tpp:
                        h_test[4] = True
                    else:
                        # print("Failed secondary hypothesis #4 (removal of BiC->PYR connections)")
                        return False
                else:
                    # print("Failed secondary hypothesis #3 (removal of PV->PYR connections)")
                    return False
            else:
                # print("Failed secondary hypothesis #2 (removal of CCK->PV connections)")
                return False
        else:
            # print("Failed secondary hypothesis #1 (removal of PYR->PYR connections)")
            return False
    else:
        # print("Failed primary hypothesis")
        return False

    # print("Valid reference parameter set")
    return True
# =============================================================
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
# # ===============================================================
# new_prm = PRM_v2()
#
# time = np.arange(0, T, dt)
#
# new_prm.set_init_state(len(time))
# new_prm = simulate(time, new_prm, dt, tau, r_o)
# # =================================================================
# dps_tpp = calc_spectral(new_prm.R, fs, time, new_prm.labels, 'theta', 'power')["pyr"]
# dps_gpp = calc_spectral(new_prm.R, fs, time, new_prm.labels, 'gamma', 'power')["pyr"]
# # =================================================================

# new_conns = new_prm.conns.copy()
# new_I = new_prm.I.copy()
#
# chn = [0.5, 1, 2]
# for c1 in c_list:
#     new_I[c1] /= chn[np.random.randint(3)]
#     for c2 in c_list:
#         if new_conns[c1][c2] != 0:
#
#             new_conns[c1][c2] /= chn[np.random.randint(3)]
#
# print(new_conns)
# print(new_I)
# hypothesis_test(new_conns, new_I)

# loop over each channel?
# how many values per weight/input
# steps = 5
# mf = np.linspace(0.5, 1.5, steps)
#
# h_prm = PRM_v2()
# count = 0
# for m in mf:
#     for c1 in c_list:
#         for c2 in c_list:
#             if new_conns[c1][c2] != 0:
#                 new_conns[c1][c2] *= m
#                 count +=1
#
# print(count)



# new_conns = {
#     'pyr': {'pyr': 0.03, 'bic': 0.04, 'pv': 0.02, 'cck': 0.0},
#     'bic': {'pyr': -0.06, 'bic': 0.0, 'pv': 0.0, 'cck': 0.0},
#     'pv': {'pyr': -0.04, 'bic': 0.0, 'pv': -0.055, 'cck': -0.15},
#     'cck': {'pyr': 0.0, 'bic': 0.0, 'pv': -0.3, 'cck': -0.075}
# }
# new_I = {
#     'pyr': 0.07, 'bic': -0.525, 'pv': 0.225, 'cck': 1.4
# }

#

# print(hypothesis_test(plot=True))
#
# p2 = PRM_v2(new_conns, new_I)
# p2.set_init_state(len(time))
# p2 = simulate(time, p2, dt, tau, r_o)
#
# p2_gpp = calc_spectral(p2.R, fs, time, p2.labels, 'gamma', 'power', plot_Fig=True, plot_Filter=True)["pyr"]
# print(dps_gpp, p2_gpp, (p2_gpp/dps_gpp)*100)
#
# plot_trace(time, p2.R, p2.labels)
# # #
# p2.set_connections(new_conns)
# # p2.conns["cck"]["pv"] = 0
# p2.set_init_state(len(time))
# p2 = simulate(time, p2, dt, tau, r_o)
#
# plot_trace(time, p2.R, p2.labels)
# new_tpp = calc_spectral(p2.R, fs, time, p2.labels, 'theta', 'power')["pyr"]
#
# print(f"dps ttp: {dps_tpp}")
# print(f"no_cck_pv_tpp: {new_tpp}")

# plt.tight_layout()
plt.show()
