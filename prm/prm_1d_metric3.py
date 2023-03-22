from copy import deepcopy

from prm_v2 import *

import numpy as np
import matplotlib.pyplot as plt

def plot_stim(cell, n_pts=40, ref=None, conns="default", I="default", exclude_invalid=True):
    new_conns = deepcopy(conns)
    new_I = deepcopy(I)
    if ref is None:
        ylabel = "Log Power Ratio"
        save_name = f"stim_to_{cell}_cell_exc.png"
        ylim = (-1, 3)

    else:
        ylabel = "%$\Delta$ Log Power Ratio"
        save_name = f"stim_to_{cell}_cell.png"
        ylim = (-60, 150)

    plt.figure(figsize=[9, 3])

    P = PRM_v2(new_conns, new_I)
    x_vec = np.linspace(-2, 2, num_pts)

    if exclude_invalid:
        ref_prm = PRM_v2()
        ref_prm.set_init_state(len(time))
        ref_prm = simulate(time, ref_prm)

        ref_tpp = calc_spectral(ref_prm.R, fs, time, 'theta', 'power')["pyr"]
        ref_gpp = calc_spectral(ref_prm.R, fs, time, 'gamma', 'power')["pyr"]

    Y = np.zeros(n_pts)
    Y_pct = np.zeros_like(Y)

    stim = {
        "pyr": 0,
        "bic": 0,
        "pv": 0,
        "cck": 0
    }

    for idx, val in enumerate(x_vec):
        stim[cell] = val
        P.set_init_state(len(time))
        P = simulate(time, P, dt, tau, stim=stim)

        if exclude_invalid:
            tpp, gpp = valid_oscillation(P.R, fs, time, ref=[ref_tpp, ref_gpp])

            Y[idx] = np.log10(tpp/gpp)

        else:
            Y[idx] = np.log10(
                calc_spectral(P.R, fs, time, 'theta', 'power')["pyr"] /
                calc_spectral(P.R, fs, time, 'gamma', 'power')["pyr"]
            )

    if ref is None:
        plt.plot(x_vec, Y,
                 ls='--', marker='.')

    else:
        Y_pct = Y / ref * 100
        plt.plot(x_vec, Y_pct,
                 ls='--', marker='.')

    plt.ylabel(ylabel)
    plt.xlabel("Stimulation")
    plt.title(f"Stimulation to {ctype} cell".upper())
    plt.xlim(x_vec[0], x_vec[-1])
    plt.ylim(ylim)
    plt.grid()
    # plt.legend()
    plt.tight_layout()
    plt.savefig(f"./figures/1d_metric3/new_ref_set/{save_name}")


def plot_1d(cell, max_in, n_pts=40, ref=None, conns="default", I="default"):
    new_conns = deepcopy(conns)
    new_I = deepcopy(I)
    if ref is None:
        ylabel = "Log Power Ratio"
        save_name = f"inputs_to_{cell}_cell_raw.png"
        ylim = (-1, 3)

    else:
        ylabel = "%$\Delta$ Log Power Ratio"
        save_name = f"inputs_to_{cell}_cell.png"
        ylim = (-60, 150)

    plt.figure(figsize=[9, 3])

    for i in range(max_in):
        P = PRM_v2(new_conns, new_I)
        x_vec = np.linspace(0.5, 1.5, num_pts)

        log_power_ratio = np.zeros((max_in, n_pts))
        y_pct = np.zeros_like(log_power_ratio)

        if i == len(c_list):
            p_vec = x_vec * P.I[cell]
            y_label = f"$I_{{{cell.upper()}}}$"

            for idx, val in enumerate(p_vec):
                P.I[cell] = val
                P.set_init_state(len(time))
                P = simulate(time, P, dt, tau, r_o)

                log_power_ratio[i, idx] = np.log10(
                    calc_spectral(P.R, fs, time, P.labels, 'theta', 'power')["pyr"] /
                    calc_spectral(P.R, fs, time, P.labels, 'gamma', 'power')["pyr"]
                )

            if ref is None:
                plt.plot(x_vec * 100, log_power_ratio[i], label=y_label,
                         ls='--', marker='.')

            else:
                y_pct[i] = log_power_ratio[i] / ref * 100
                plt.plot(x_vec * 100, y_pct[i], label=y_label,
                         ls='--', marker='.')

        else:
            if P.conns[c_list[i]][cell] == 0:
                pass
            else:
                p_vec = x_vec * P.conns[c_list[i]][cell]
                y_label = f"$w_{{{c_list[i].upper()} \\rightarrow {cell.upper()}}}$"

                for idx, val in enumerate(p_vec):
                    P.conns[c_list[i]][cell] = val
                    P.set_init_state(len(time))
                    P = simulate(time, P, dt, tau, r_o)

                    log_power_ratio[i, idx] = np.log10(
                        calc_spectral(P.R, fs, time, P.labels, 'theta', 'power')["pyr"] /
                        calc_spectral(P.R, fs, time, P.labels, 'gamma', 'power')["pyr"]
                    )

                if ref is None:
                    plt.plot(x_vec * 100, log_power_ratio[i], label=y_label,
                             ls='--', marker='.')

                else:
                    y_pct[i] = log_power_ratio[i] / ref * 100
                    plt.plot(x_vec * 100, y_pct[i], label=y_label,
                             ls='--', marker='.')

    plt.ylabel(ylabel)
    plt.xlabel("%$\Delta$ Parameter")
    plt.title(f"inputs to {ctype} cell".upper())
    plt.ylim(ylim)
    plt.grid()
    # plt.legend()
    plt.tight_layout()
    plt.savefig(f"./figures/1d_metric3/ref_set_2/{save_name}")


# ===============================================================
# Parameters
T = 8.0  # total time (units in sec)
dt = 0.001  # plotting and Euler timestep (parameters adjusted accordingly)
fs = 1 / dt
time = np.arange(0, T, dt)

# FI curve
beta = 10
tau = 5
h = 0
r_o = 30

c_list = ["pyr", "bic", "pv", "cck"]
# ===============================================================
# new_conns = {
#     'pyr': {'pyr': 0.06, 'bic': 0.08, 'pv': 0.02, 'cck': 0.0},
#     'bic': {'pyr': -0.03, 'bic': 0.0, 'pv': 0.0, 'cck': 0.0},
#     'pv': {'pyr': -0.08, 'bic': 0.0, 'pv': -0.11, 'cck': -0.075},
#     'cck': {'pyr': 0.0, 'bic': 0.0, 'pv': -0.15, 'cck': -0.075}
# }
# new_I = {
#     'pyr': 0.07, 'bic': -0.525, 'pv': 0.9, 'cck': 0.7
# }
#
# P_ref = PRM_v2(new_conns, new_I)
# P_ref.set_init_state(len(time))
# P_ref = simulate(time, P_ref, dt, tau, r_o)
#
# ref_metric3 = np.log10(
#     calc_spectral(P_ref.R, fs, time, P_ref.labels, 'theta', 'power')["pyr"] /
#     calc_spectral(P_ref.R, fs, time, P_ref.labels, 'gamma', 'power')["pyr"]
# )
# ===============================================================

# cell type to look at
ctype = "cck"

num_pts = 100  # number of points to plot

max_inputs = len(c_list) + 1

plot_stim(ctype, num_pts, exclude_invalid=True)
plt.show()
