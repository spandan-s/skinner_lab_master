import json
from copy import deepcopy

from tqdm import tqdm

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

            Y[idx] = np.log10(tpp / gpp)

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
    # plt.savefig(f"./figures/1d_metric3/new_ref_set/{save_name}")


def plot_stim_theta_gamma(cell, n_pts=40, conns="default", I="default",
                          plot_lpr=True, exclude_invalid=True, sdir=""):
    new_conns = deepcopy(conns)
    new_I = deepcopy(I)

    if exclude_invalid:
        ref_prm = PRM_v2()

        temp_theta, temp_gamma = np.zeros(5), np.zeros(5)

        for i in range(5):
            ref_prm.set_init_state(len(time))
            ref_prm = simulate(time, ref_prm)

            temp_theta[i] = calc_spectral(ref_prm.R, fs, time, 'theta', 'power')["pyr"]
            temp_gamma[i] = calc_spectral(ref_prm.R, fs, time, 'gamma', 'power')["pyr"]

        ref_tpp = np.mean(temp_theta)
        ref_gpp = np.mean(temp_gamma)

    save_name = f"stim_to_{cell}_cell_theta_gamma.png"

    if plot_lpr:
        fig, ax = plt.subplots(nrows=2, sharex=True,
                               figsize=[9, 6.4], dpi=250)
        ax1 = ax[0].twinx()

    else:
        fig, ax = plt.subplots(figsize=[9, 3], dpi=250)
        ax1 = ax.twinx()

    P = PRM_v2(new_conns, new_I)
    x_vec = np.linspace(-2, 2, num_pts)

    theta = np.zeros(n_pts)
    gamma = np.zeros(n_pts)
    theta_std = np.zeros(n_pts)
    gamma_std = np.zeros(n_pts)

    if plot_lpr:
        LPR = np.zeros(n_pts)
        LPR_valid = np.zeros(n_pts)

    stim = {
        "pyr": 0,
        "bic": 0,
        "pv": 0,
        "cck": 0
    }

    for idx, val in tqdm(enumerate(x_vec)):
        stim[cell] = val
        temp_theta, temp_gamma = np.zeros(5), np.zeros(5)

        for j in range(5):
            P.set_init_state(len(time))
            P = simulate(time, P, dt, tau, stim=stim)
            temp_theta[j] = calc_spectral(P.R, fs, time, 'theta', 'power')["pyr"]
            temp_gamma[j] = calc_spectral(P.R, fs, time, 'gamma', 'power')["pyr"]

        theta[idx] = np.mean(temp_theta)
        theta_std[idx] = np.std(temp_theta)
        gamma[idx] = np.mean(temp_gamma)
        gamma_std[idx] = np.std(temp_gamma)

        if plot_lpr:
            LPR[idx] = np.log10(theta[idx] / gamma[idx])
            if exclude_invalid:
                if theta[idx] < (0.25 * ref_tpp) or gamma[idx] < (0.25 * ref_gpp):
                    LPR_valid[idx] = np.nan
                else:
                    LPR_valid[idx] = LPR[idx]

    # ==============================================================================
    # First subplot --> theta and gamma power vs stim
    p1, = ax[0].plot(x_vec, theta,
                     ls='--', marker='.', color='C0')  # plot theta power
    ax[0].fill_between(x_vec, theta - theta_std, theta + theta_std,
                       color="C0", alpha=0.2)  # plot theta error
    p2, = ax1.plot(x_vec, gamma,
                   ls='--', marker='.', color='C1')  # plot gamma power
    ax1.fill_between(x_vec, gamma - gamma_std, gamma + gamma_std,
                     color="C1", alpha=0.2)  # plot gamma error

    # subplot 1 axes and labels
    ax[0].set_title(f"Stimulation to {ctype} cell".upper())
    ax[0].set_xlim(x_vec[0], x_vec[-1])
    ax[0].set_ylabel("Theta Power")
    ax1.set_ylabel("Gamma Power")

    ax[0].yaxis.label.set_color(p1.get_color())
    ax1.yaxis.label.set_color(p2.get_color())

    ax[0].tick_params(axis='y', colors=p1.get_color())
    ax1.tick_params(axis='y', colors=p2.get_color())
    ax[0].grid(axis='x')
    # end of first subplot
    # ==============================================================================

    # ==============================================================================
    # second subplot --> log power ratio vs stim
    ax[1].plot(x_vec, LPR,
               ls='--', marker='.', color='C0')
    if exclude_invalid:
        ax[1].plot(x_vec, LPR_valid,
                   ls='', marker='.', color='C3')

    # subplot 2 axes and labels
    ax[1].set_xlabel("Stimulation")
    ax[1].set_ylabel("Log Power Ratio")
    ax[1].grid(axis='x')

    # ax.grid()
    plt.tight_layout()
    plt.savefig(f"./figures/1d_metric3/{sdir}/{save_name}")


def plot_1d(cell, max_in, n_pts=40, ref=None, conns="default", I="default",
            exclude_invalid=True, sdir=""):
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

    if exclude_invalid:
        ref_prm = PRM_v2()

        temp_theta, temp_gamma = np.zeros(5), np.zeros(5)

        for i in range(5):
            ref_prm.set_init_state(len(time))
            ref_prm = simulate(time, ref_prm)

            temp_theta[i] = calc_spectral(ref_prm.R, fs, time, 'theta', 'power')["pyr"]
            temp_gamma[i] = calc_spectral(ref_prm.R, fs, time, 'gamma', 'power')["pyr"]

        ref_tpp = np.mean(temp_theta)
        ref_gpp = np.mean(temp_gamma)

    plt.figure(figsize=[9, 3])

    for i in range(max_in):
        P = PRM_v2(new_conns, new_I)
        x_vec = np.linspace(0.5, 1.5, num_pts)

        # log_power_ratio = np.zeros((max_in, n_pts))
        # y_pct = np.zeros_like(log_power_ratio)

        # if i == len(c_list):
        #     p_vec = x_vec * P.I[cell]
        #     y_label = f"$I_{{{cell.upper()}}}$"
        #
        #     for idx, val in enumerate(p_vec):
        #         P.I[cell] = val
        #         P.set_init_state(len(time))
        #         P = simulate(time, P, dt, tau, r_o)
        #
        #         log_power_ratio[i, idx] = np.log10(
        #             calc_spectral(P.R, fs, time, P.labels, 'theta', 'power')["pyr"] /
        #             calc_spectral(P.R, fs, time, P.labels, 'gamma', 'power')["pyr"]
        #         )
        #
        #     if ref is None:
        #         plt.plot(x_vec * 100, log_power_ratio[i], label=y_label,
        #                  ls='--', marker='.')
        #
        #     else:
        #         y_pct[i] = log_power_ratio[i] / ref * 100
        #         plt.plot(x_vec * 100, y_pct[i], label=y_label,
        #                  ls='--', marker='.')
        #
        # else:
        if P.conns[c_list[i]][cell] == 0:
            pass
        else:
            p_vec = x_vec * P.conns[c_list[i]][cell]
            y_label = f"$w_{{{c_list[i].upper()} \\rightarrow {cell.upper()}}}$"

            theta = np.zeros(n_pts)
            gamma = np.zeros(n_pts)
            theta_std = np.zeros(n_pts)
            gamma_std = np.zeros(n_pts)

            LPR = np.zeros(n_pts)
            LPR_valid = np.zeros(n_pts)

            for idx, val in tqdm(enumerate(p_vec)):
                P.conns[c_list[i]][cell] = val
                temp_theta, temp_gamma = np.zeros(5), np.zeros(5)

                for j in range(5):
                    P.set_init_state(len(time))
                    P = simulate(time, P, dt, tau)
                    temp_theta[j] = calc_spectral(P.R, fs, time, 'theta', 'power')["pyr"]
                    temp_gamma[j] = calc_spectral(P.R, fs, time, 'gamma', 'power')["pyr"]

                theta[idx] = np.mean(temp_theta)
                theta_std[idx] = np.std(temp_theta)
                gamma[idx] = np.mean(temp_gamma)
                gamma_std[idx] = np.std(temp_gamma)

                LPR[idx] = np.log10(theta[idx] / gamma[idx])
                if exclude_invalid:
                    if theta[idx] < (0.25 * ref_tpp) or gamma[idx] < (0.25 * ref_gpp):
                        LPR_valid[idx] = np.nan
                    else:
                        LPR_valid[idx] = LPR[idx]

            if ref is None:
                plt.plot(x_vec, LPR,
                           ls='--', marker='.', label=y_label)
                if exclude_invalid:
                    plt.plot(x_vec, LPR_valid,
                               ls='', marker='.', color='red')

            else:
                LPR_ratio = LPR / ref * 100
                plt.plot(x_vec * 100, LPR, label=y_label,
                         ls='--', marker='.')

    plt.ylabel(ylabel)
    plt.xlabel("%$\Delta$ Parameter")
    plt.title(f"inputs to {ctype} cell".upper())
    # plt.ylim(ylim)
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"./figures/1d_metric3/{sdir}/{save_name}")


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
with open("search_results/search_results_conn_6.json", "r") as f:
    conn_data = json.load(f)

new_conns = conn_data[26]
# ===============================================================

# cell type to look at
ctype = "cck"

num_pts = 21  # number of points to plot

max_inputs = len(c_list)

for ctype in c_list:
    plot_1d(ctype, max_inputs, num_pts, conns=new_conns,
            exclude_invalid=True, sdir="conn_6_26/6_26_w_plots")
    print(f"Completed for {ctype} cell")

# plot_stim_theta_gamma(ctype, num_pts, conns=new_conns, exclude_invalid=True)
# for ctype in c_list:
#     plot_stim_theta_gamma(ctype, num_pts, conns="default",
#                           exclude_invalid=True, sdir="new_ref_set/ref_theta_gamma_plots")
#     print(f"Completed for {ctype} cell")
# plt.show()
