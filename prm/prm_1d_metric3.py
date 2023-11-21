import json
import os
from copy import deepcopy

import pandas as pd
from numpy.linalg import LinAlgError
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
                          n_trials=1, run_type='baseline',
                          plot_lpr=True, exclude_invalid=True, stim_range=(-2, 2), sdir=""):
    new_conns = deepcopy(conns)
    new_I = deepcopy(I)

    if exclude_invalid:
        ref_tpp, ref_gpp = ref_power()

    save_name = f"stim_to_{cell}_cell_theta_gamma_2"
    save_ext_img = ".png"
    save_ext_txt = ".dat"

    if plot_lpr:
        fig, ax = plt.subplots(nrows=2, sharex=True,
                               figsize=[9, 6.4], dpi=250)
        ax1 = ax[0].twinx()

    else:
        fig, ax = plt.subplots(figsize=[9, 3], dpi=250)
        ax1 = ax.twinx()

    P = PRM_v2(new_conns, new_I)
    # x_vec = np.linspace(stim_range[0], stim_range[1], num_pts)
    x_vec = np.arange(stim_range[0], stim_range[1], 0.005)
    # theta = np.zeros(n_pts)
    # gamma = np.zeros(n_pts)
    # theta_std = np.zeros(n_pts)
    # gamma_std = np.zeros(n_pts)

    theta = np.zeros_like(x_vec)
    gamma = np.zeros_like(x_vec)
    theta_std = np.zeros_like(x_vec)
    gamma_std = np.zeros_like(x_vec)
    if plot_lpr:
        # LPR = np.zeros(n_pts)
        # LPR_valid = np.zeros(n_pts)
        LPR = np.zeros_like(x_vec)
        LPR_valid = np.zeros_like(x_vec)

    stim = {
        "pyr": 0,
        "bic": 0,
        "pv": 0,
        "cck": 0
    }
    if run_type != 'baseline':
        cell_type, s = run_type.split('_')
        s_val = float('0.' + s.split('n')[-1].split('0')[-1]) * (1 if s.split('n')[0] else -1)
        stim[cell_type] = s_val

    for idx, val in tqdm(enumerate(x_vec)):
        stim[cell] = val
        temp_theta, temp_gamma = np.zeros((n_trials, 2)), np.zeros((n_trials, 2))

        for j in range(n_trials):
            temp_theta[j], temp_gamma[j] = run_prm(conns=new_conns, I=new_I, stim=stim)

        theta[idx] = np.mean(temp_theta[:, 1])
        theta_std[idx] = np.std(temp_theta[:, 1])
        gamma[idx] = np.mean(temp_gamma[:, 1])
        gamma_std[idx] = np.std(temp_gamma[:, 1])

        if plot_lpr:
            LPR[idx] = np.log10(theta[idx] / gamma[idx])
            if exclude_invalid:
                if theta[idx] < 48 or gamma[idx] < 0.31:
                    LPR_valid[idx] = np.nan
                else:
                    LPR_valid[idx] = LPR[idx]

    save_arr = np.vstack([x_vec, theta, gamma, LPR]).T
    np.savetxt(f"./figures/{sdir}/raw/{save_name}{save_ext_txt}", save_arr, header="stim, theta, gamma, LPR")

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
    ax[0].set_title(f"Stimulation to {cell} cell".upper())
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
    plt.savefig(f"./figures/{sdir}/{save_name}{save_ext_img}")


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


def plot_stim_v_freq(cell, n_pts=40, conns="default", I="default",
                     stim_range=(-2, 2), n_trials=1,
                     run_type='baseline',
                     plot_lpr=True, exclude_invalid=True, draw_fit=True, sdir=""):
    # draw_fit = False  # override
    new_conns = deepcopy(conns)
    new_I = deepcopy(I)

    if exclude_invalid:
        ref_tpp, ref_gpp = ref_power()
        # print(ref_tpp, ref_gpp)

    save_name = f"stim_to_{run_type}_{cell}_cell_theta_freq_{n_pts}"
    save_ext_img = ".png"
    save_ext_txt = ".dat"

    if plot_lpr:
        fig, ax = plt.subplots(nrows=2, sharex=True,
                               figsize=[9, 6.4], dpi=250)
    else:
        fig, ax = plt.subplots(figsize=[9, 3], dpi=250)

    x_vec = np.arange(stim_range[0], stim_range[1], 0.005)
    x_valid = x_vec.copy()

    theta = np.zeros_like(x_vec)
    gamma = np.zeros_like(x_vec)
    theta_std = np.zeros_like(x_vec)
    gamma_std = np.zeros_like(x_vec)

    theta_freq = np.zeros_like(x_vec)
    theta_freq_valid = np.zeros_like(theta_freq)
    theta_freq_std = np.zeros_like(x_vec)

    if plot_lpr:
        LPR = np.zeros_like(x_vec)
        LPR_valid = np.zeros_like(x_vec)

    stim = {
        "pyr": 0,
        "bic": 0,
        "pv": 0,
        "cck": 0
    }
    if run_type != 'baseline':
        cell_type, s = run_type.split('_')
        s_val = float('0.' + s.split('n')[-1].split('0')[-1]) * (1 if s.split('n')[0] else -1)
        stim[cell_type] = s_val

    for idx, val in tqdm(enumerate(x_vec)):
        stim[cell] = val
        temp_theta, temp_gamma = np.zeros((n_trials, 2)), np.zeros((n_trials, 2))

        for j in range(n_trials):
            temp_theta[j], temp_gamma[j] = run_prm(conns=new_conns, I=new_I, stim=stim)

        theta[idx] = np.mean(temp_theta[:, 1])
        theta_std[idx] = np.std(temp_theta[:, 1])
        gamma[idx] = np.mean(temp_gamma[:, 1])
        gamma_std[idx] = np.std(temp_gamma[:, 1])

        theta_freq[idx] = np.mean(temp_theta[:, 0])
        theta_freq_std[idx] = np.std(temp_theta[:, 0])
        # print(theta_freq[idx], theta_freq_std[idx])
        # print(theta[idx], gamma[idx])

        if plot_lpr:
            LPR[idx] = np.log10(theta[idx] / gamma[idx])
            if exclude_invalid:
                if theta[idx] < 48 or gamma[idx] < 0.31:
                    LPR_valid[idx] = np.nan
                    theta_freq_valid[idx] = np.nan
                    x_valid[idx] = np.nan
                else:
                    LPR_valid[idx] = LPR[idx]
                    theta_freq_valid[idx] = theta_freq[idx]

    if draw_fit:
        x_valid = x_valid[~np.isnan(x_valid)]
        theta_freq_valid = theta_freq_valid[~np.isnan(theta_freq_valid)]
        try:
            a1, a0 = np.polyfit(x_valid, theta_freq_valid, 1)
            print(a1, a0)
        except LinAlgError:
            a1, a0 = np.nan, np.nan
        text_coords = (stim_range[0] + 0.1 * (stim_range[1] - stim_range[0]), max(theta_freq))
        print(f"Valid Stim Range: {np.min(x_valid).round(2), np.max(x_valid).round(2)}")
        print(f"Theta Frequency Range: {np.min(theta_freq_valid), np.max(theta_freq_valid)}")

    save_arr = np.vstack([x_vec, theta_freq, theta, gamma, LPR]).T
    np.savetxt(f"./figures/{sdir}/raw/{save_name}{save_ext_txt}", save_arr,
               header="stim, theta_freq, theta, gamma, LPR")
    # ==============================================================================
    # First subplot --> theta frequency vs stim
    p1, = ax[0].plot(x_vec, theta_freq,
                     ls='--', marker='.', color='C0')  # plot theta freq
    if exclude_invalid:
        ax[0].plot(x_valid, theta_freq_valid,
                   ls="", marker=".", color="C3")
        if draw_fit:
            ax[0].plot(x_valid, a0 + a1 * x_valid,
                       ls="dotted", color="purple")
            ax[0].text(*text_coords, f"r = {np.round(a1, 3)}")
    ax[0].fill_between(x_vec, theta_freq - theta_freq_std, theta_freq + theta_freq_std,
                       color="C0", alpha=0.2)  # plot theta error

    # subplot 1 axes and labels
    ax[0].set_title(f"Stimulation to {cell} cell".upper())
    ax[0].set_xlim(x_vec[0], x_vec[-1])
    ax[0].set_ylabel("Theta Frequency")

    ax[0].yaxis.label.set_color(p1.get_color())

    ax[0].tick_params(axis='y', colors=p1.get_color())
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
    plt.savefig(f"./figures/{sdir}/{save_name}{save_ext_img}")


# def conn_freq_power(conn1, conn2, n_pts=121,
#                     conns="default", I="default",
#                     conn_range=[0.5, 1.5], n_trials=1, sdir=""):
#     if conns == "default":
#         P = PRM_v2()
#         new_conns = deepcopy(P.conns)
#     else:
#         new_conns = deepcopy(conns)
#
#     if I == "default":
#         P = PRM_v2()
#         new_I = deepcopy(P.I)
#     else:
#         new_I = deepcopy(I)
#
#     save_name = f"w_{conn1}{conn2}_theta_freq_power_{n_pts}"
#     save_ext_img = ".png"
#     save_ext_txt = ".dat"
#
#     x_vec = np.linspace(conn_range[0], conn_range[1], num_pts)
#     p_vec = x_vec * new_conns[conn1][conn2]
#
#     theta = np.zeros(n_pts)
#     gamma = np.zeros(n_pts)
#     theta_std = np.zeros(n_pts)
#     gamma_std = np.zeros(n_pts)
#
#     theta_freq = np.zeros(n_pts)
#     theta_freq_std = np.zeros(n_pts)
#
#     for idx, val in tqdm(enumerate(p_vec)):
#         new_conns[conn1][conn2] = val
#         temp_theta, temp_gamma = np.zeros((5, 2)), np.zeros((5, 2))
#
#         for j in range(n_trials):
#             temp_theta[j], temp_gamma[j] = run_prm(conns=new_conns, I=new_I)
#
#         theta[idx] = np.mean(temp_theta[:, 1])
#         theta_std[idx] = np.std(temp_theta[:, 1])
#         gamma[idx] = np.mean(temp_gamma[:, 1])
#         gamma_std[idx] = np.std(temp_gamma[:, 1])
#
#         theta_freq[idx] = np.mean(temp_theta[:, 0])
#         theta_freq_std[idx] = np.std(temp_theta[:, 0])
#
#         save_arr = np.vstack([p_vec, theta_freq, theta, gamma]).T
#         np.savetxt(f"./figures/{sdir}/raw/{save_name}{save_ext_txt}", save_arr,
#                    header="weight, theta_freq, theta, gamma")

def plot_conn_v_theta_freq(conn1, conn2, n_pts=41,
                           conns=None, I=None,
                           conn_range=[0.5, 1.5], n_trials=1,
                           plot_lpr=True, exclude_invalid=True, sdir=""):
    if conns == None:
        P = PRM_v2()
        new_conns = deepcopy(P.conns)
    else:
        new_conns = deepcopy(conns)

    if I == None:
        P = PRM_v2()
        new_I = deepcopy(P.I)
    else:
        new_I = deepcopy(I)

    if exclude_invalid:
        ref_tpp, ref_gpp = ref_power()
        # print(ref_tpp, ref_gpp)

    save_name = f"w{conn1}_{conn2}_theta_freq"
    save_ext_img = '.png'
    save_ext_txt = '.dat'

    if plot_lpr:
        fig, ax = plt.subplots(nrows=2, sharex=True,
                               figsize=[9, 6.4], dpi=250)
    else:
        fig, ax = plt.subplots(figsize=[9, 3], dpi=250)

    x_vec = np.linspace(conn_range[0], conn_range[1], num_pts)
    p_vec = x_vec * new_conns[conn1][conn2]

    theta = np.zeros(n_pts)
    gamma = np.zeros(n_pts)
    theta_std = np.zeros(n_pts)
    gamma_std = np.zeros(n_pts)

    theta_freq = np.zeros(n_pts)
    theta_freq_std = np.zeros(n_pts)

    if plot_lpr:
        LPR = np.zeros(n_pts)
        LPR_valid = np.zeros(n_pts)

    for idx, val in tqdm(enumerate(p_vec)):
        new_conns[conn1][conn2] = val
        temp_theta, temp_gamma = np.zeros((n_trials, 2)), np.zeros((n_trials, 2))

        for j in range(n_trials):
            temp_theta[j], temp_gamma[j] = run_prm(conns=new_conns, I=new_I)

        theta[idx] = np.mean(temp_theta[:, 1])
        theta_std[idx] = np.std(temp_theta[:, 1])
        gamma[idx] = np.mean(temp_gamma[:, 1])
        gamma_std[idx] = np.std(temp_gamma[:, 1])

        theta_freq[idx] = np.mean(temp_theta[:, 0])
        theta_freq_std[idx] = np.std(temp_theta[:, 0])

        if plot_lpr:
            LPR[idx] = np.log10(theta[idx] / gamma[idx])
            if exclude_invalid:
                if theta[idx] < (0.25 * ref_tpp) or gamma[idx] < (0.25 * ref_gpp):
                    LPR_valid[idx] = np.nan
                else:
                    LPR_valid[idx] = LPR[idx]
    save_arr = np.vstack([x_vec, theta_freq, theta, gamma, LPR]).T
    # ==============================================================================
    # First subplot --> theta frequency vs stim
    p1, = ax[0].plot(x_vec, theta_freq,
                     ls='--', marker='.', color='C0')  # plot theta freq
    ax[0].fill_between(x_vec, theta_freq - theta_freq_std, theta_freq + theta_freq_std,
                       color="C0", alpha=0.2)  # plot theta error

    # subplot 1 axes and labels
    ax[0].set_title(f"$w_{{{conn1.upper()} \\rightarrow {conn2.upper()}}}$")
    ax[0].set_xlim(x_vec[0], x_vec[-1])
    ax[0].set_ylabel("Theta Frequency")

    ax[0].yaxis.label.set_color(p1.get_color())

    ax[0].tick_params(axis='y', colors=p1.get_color())
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
    ax[1].set_xlabel("%$\Delta$ Parameter")
    ax[1].set_ylabel("Log Power Ratio")
    ax[1].grid(axis='x')

    # ax.grid()
    plt.tight_layout()
    try:
        plt.savefig(f"./figures/{sdir}/{save_name}{save_ext_img}")
        np.savetxt(f"./figures/{sdir}/raw/{save_name}{save_ext_txt}", save_arr,
                   header="stim, theta_freq, theta, gamma, LPR")
    except FileNotFoundError:
        os.makedirs(f"./figures/{sdir}/raw")
        plt.savefig(f"./figures/{sdir}/{save_name}{save_ext_img}")
        np.savetxt(f"./figures/{sdir}/raw/{save_name}{save_ext_txt}", save_arr,
                   header="stim, theta_freq, theta, gamma, LPR")

    plt.close()


# def plot_conn_v_theta_freq(conn1, conn2, n_pts=41,
#                            conns=None, I=None,
#                            conn_range=[0.5, 1.5], plot_lpr=True, exclude_invalid=True, sdir=""):
#     if conns == None:
#         P = PRM_v2()
#         new_conns = deepcopy(P.conns)
#     else:
#         new_conns = deepcopy(conns)
#
#     if I == None:
#         P = PRM_v2()
#         new_I = deepcopy(P.I)
#     else:
#         new_I = deepcopy(I)
#
#     if exclude_invalid:
#         ref_tpp, ref_gpp = ref_power()
#         # print(ref_tpp, ref_gpp)
#
#     save_name = f"w{conn1}_{conn2}_theta_freq"
#     save_ext_img = '.png'
#     save_ext_txt = '.dat'
#
#     if plot_lpr:
#         fig, ax = plt.subplots(nrows=2, sharex=True,
#                                figsize=[9, 6.4], dpi=250)
#     else:
#         fig, ax = plt.subplots(figsize=[9, 3], dpi=250)
#
#     x_vec = np.linspace(conn_range[0], conn_range[1], num_pts)
#     p_vec = x_vec * new_conns[conn1][conn2]
#
#     theta = np.zeros(n_pts)
#     gamma = np.zeros(n_pts)
#     theta_std = np.zeros(n_pts)
#     gamma_std = np.zeros(n_pts)
#
#     theta_freq = np.zeros(n_pts)
#     theta_freq_std = np.zeros(n_pts)
#
#     if plot_lpr:
#         LPR = np.zeros(n_pts)
#         LPR_valid = np.zeros(n_pts)
#
#     for idx, val in tqdm(enumerate(p_vec)):
#         new_conns[conn1][conn2] = val
#         temp_theta, temp_gamma = np.zeros((5, 2)), np.zeros((5, 2))
#
#         for j in range(n_trials):
#             temp_theta[j], temp_gamma[j] = run_prm(conns=new_conns, I=new_I)
#
#         theta[idx] = np.mean(temp_theta[:, 1])
#         theta_std[idx] = np.std(temp_theta[:, 1])
#         gamma[idx] = np.mean(temp_gamma[:, 1])
#         gamma_std[idx] = np.std(temp_gamma[:, 1])
#
#         theta_freq[idx] = np.mean(temp_theta[:, 0])
#         theta_freq_std[idx] = np.std(temp_theta[:, 0])
#
#         if plot_lpr:
#             LPR[idx] = np.log10(theta[idx] / gamma[idx])
#             if exclude_invalid:
#                 if theta[idx] < 48 or gamma[idx] < 0.31:
#                     LPR_valid[idx] = np.nan
#                     theta_freq_valid[idx] = np.nan
#                     p_valid[idx] = np.nan
#                 else:
#                     LPR_valid[idx] = LPR[idx]
#                     theta_freq_valid[idx] = theta_freq[idx]
#
#     if draw_fit:
#         x_valid = x_valid[~np.isnan(x_valid)]
#         theta_freq_valid = theta_freq_valid[~np.isnan(theta_freq_valid)]
#         try:
#             a1, a0 = np.polyfit(x_valid, theta_freq_valid, 1)
#             print(a1, a0)
#         except LinAlgError:
#             a1, a0 = np.nan, np.nan
#         text_coords = (stim_range[0] + 0.1*(stim_range[1] - stim_range[0]), max(theta_freq))
#         print(f"Valid Stim Range: {np.min(x_valid).round(2), np.max(x_valid).round(2)}")
#         print(f"Theta Frequency Range: {np.min(theta_freq_valid), np.max(theta_freq_valid)}")
#
#
#     save_arr = np.vstack([x_vec, theta_freq, theta, gamma, LPR]).T
#     np.savetxt(f"./figures/{sdir}/raw/{save_name}{save_ext_txt}", save_arr,
#                header="stim, theta_freq, theta, gamma, LPR")
#     # ==============================================================================
#     # First subplot --> theta frequency vs stim
#     p1, = ax[0].plot(x_vec, theta_freq,
#                      ls='--', marker='.', color='C0')  # plot theta freq
#     ax[0].fill_between(x_vec, theta_freq - theta_freq_std, theta_freq + theta_freq_std,
#                        color="C0", alpha=0.2)  # plot theta error
#
#     # subplot 1 axes and labels
#     ax[0].set_title(f"$w_{{{conn1.upper()} \\rightarrow {conn2.upper()}}}$")
#     ax[0].set_xlim(x_vec[0], x_vec[-1])
#     ax[0].set_ylabel("Theta Frequency")
#
#     ax[0].yaxis.label.set_color(p1.get_color())
#
#     ax[0].tick_params(axis='y', colors=p1.get_color())
#     ax[0].grid(axis='x')
#     # end of first subplot
#     # ==============================================================================
#     # ==============================================================================
#     # second subplot --> log power ratio vs stim
#     ax[1].plot(x_vec, LPR,
#                ls='--', marker='.', color='C0')
#     if exclude_invalid:
#         ax[1].plot(x_vec, LPR_valid,
#                    ls='', marker='.', color='C3')
#
#     # subplot 2 axes and labels
#     ax[1].set_xlabel("%$\Delta$ Parameter")
#     ax[1].set_ylabel("Log Power Ratio")
#     ax[1].grid(axis='x')
#
#     # ax.grid()
#     plt.tight_layout()
#     plt.tight_layout()
#     plt.savefig(f"./figures/{sdir}/{save_name}{save_ext_img}")
#

def plot_conn_v_theta_gamma(conn1, conn2, n_pts=41,
                            conns=None, I=None,
                            conn_range=[0.5, 1.5], plot_lpr=True, exclude_invalid=True, sdir=""):
    if conns == None:
        P = PRM_v2()
        new_conns = deepcopy(P.conns)
    else:
        new_conns = deepcopy(conns)

    if I == None:
        P = PRM_v2()
        new_I = deepcopy(P.I)
    else:
        new_I = deepcopy(I)

    if exclude_invalid:
        ref_tpp, ref_gpp = ref_power()
        # print(ref_tpp, ref_gpp)

    save_name = f"w{conn1}_{conn2}_theta_gamma.png"

    if plot_lpr:
        fig, ax = plt.subplots(nrows=2, sharex=True,
                               figsize=[9, 6.4], dpi=250)
        ax1 = ax[0].twinx()
    else:
        fig, ax = plt.subplots(figsize=[9, 3], dpi=250)
        ax1 = ax.twinx()

    x_vec = np.linspace(conn_range[0], conn_range[1], num_pts)
    p_vec = x_vec * new_conns[conn1][conn2]

    theta = np.zeros(n_pts)
    gamma = np.zeros(n_pts)
    theta_std = np.zeros(n_pts)
    gamma_std = np.zeros(n_pts)

    theta_freq = np.zeros(n_pts)
    theta_freq_std = np.zeros(n_pts)

    if plot_lpr:
        LPR = np.zeros(n_pts)
        LPR_valid = np.zeros(n_pts)

    for idx, val in tqdm(enumerate(p_vec)):
        new_conns[conn1][conn2] = val
        temp_theta, temp_gamma = np.zeros((5, 2)), np.zeros((5, 2))

        for j in range(5):
            temp_theta[j], temp_gamma[j] = run_prm(conns=new_conns, I=new_I)

        theta[idx] = np.mean(temp_theta[:, 1])
        theta_std[idx] = np.std(temp_theta[:, 1])
        gamma[idx] = np.mean(temp_gamma[:, 1])
        gamma_std[idx] = np.std(temp_gamma[:, 1])

        theta_freq[idx] = np.mean(temp_theta[:, 0])
        theta_freq_std[idx] = np.std(temp_theta[:, 0])

        if plot_lpr:
            LPR[idx] = np.log10(theta[idx] / gamma[idx])
            if exclude_invalid:
                if theta[idx] < (0.25 * ref_tpp) or gamma[idx] < (0.25 * ref_gpp):
                    LPR_valid[idx] = np.nan
                else:
                    LPR_valid[idx] = LPR[idx]

    # ==============================================================================
    # First subplot --> theta frequency vs stim
    p1, = ax[0].plot(x_vec, theta,
                     ls='--', marker='.', color='C0')  # plot theta power
    ax[0].fill_between(x_vec, theta - theta_std, theta + theta_std,
                       color="C0", alpha=0.2)  # plot theta error
    p2, = ax1.plot(x_vec, gamma,
                   ls='--', marker='.', color='C1')  # plot gamma power
    ax1.fill_between(x_vec, gamma - gamma_std, gamma + gamma_std,
                     color="C1", alpha=0.2)  # plot gamma error

    # subplot 1 axes and labels
    ax[0].set_title(f"$w_{{{conn1.upper()} \\rightarrow {conn2.upper()}}}$")
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
    ax[1].set_xlabel("%$\Delta$ Parameter")
    ax[1].set_ylabel("Log Power Ratio")
    ax[1].grid(axis='x')

    # ax.grid()
    plt.tight_layout()
    try:
        plt.savefig(f"./figures/1d_metric3/{sdir}/{save_name}")
    except FileNotFoundError:
        os.mkdir(f"./figures/1d_metric3/{sdir}")
        plt.savefig(f"./figures/1d_metric3/{sdir}/{save_name}")


# ===============================================================
# Parameters
T = 8.0  # total time (units in sec)
dt = 0.001  # plotting and Euler timestep (parameters adjusted accordingly)
fs = 1 / dt
time = np.arange(0, T, dt)

# FI curve
# beta = 10
# tau = 5
# h = 0
# r_o = 30

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
conn_file_num = 10

with open(f"search_results/search_results_conn_{conn_file_num}.json", "r") as f:
    conn_data = json.load(f)

# ===============================================================

# cell type to look at
ctype = "pyr"

num_pts = 121  # number of points to plot

max_inputs = len(c_list)

# for ctype in c_list:
#     plot_1d(ctype, max_inputs, num_pts, conns=new_conns,
#             exclude_invalid=True, sdir="conn_6_26/6_26_w_plots")
#     print(f"Completed for {ctype} cell")

# STIM VS THETA GAMMA POWER
# plot_stim_theta_gamma(ctype, num_pts, conns=new_conns, stim_range=(-0.5, 0.5),
#                       exclude_invalid=True, sdir=f"conn_{conn_file_num}_{n}/{conn_file_num}_{n}_theta_gamma_plots")

# plot_stim_theta_gamma("cck", num_pts, conns="default", stim_range=(-2, 2),
#                       exclude_invalid=True, sdir=f"new_ref_set/ref_theta_gamma_plots")

# for ctype in c_list:
#     plot_stim_theta_gamma(ctype, num_pts, conns="default",
#                           exclude_invalid=True, sdir="set_0/set_0_theta_gamma_plots")
#     print(f"Completed for {ctype} cell")

# for ctype in c_list:
#     plot_stim_theta_gamma(ctype, num_pts, conns=new_conns,
#                           exclude_invalid=True, sdir=f"conn_{conn_file_num}_{n}/{conn_file_num}_{n}_theta_gamma_plots")
#     print(f"Completed for {ctype} cell")

# STIM VS THETA FREQ
# for ctype in ["pyr", "cck", "pv", "bic"]:
#     plot_stim_v_freq(ctype, num_pts, conns=new_conns,
#                      sdir=f"conn_{conn_file_num}_{n}/{conn_file_num}_{n}_stim_v_freq", stim_range=(-2, 2))
#     print(f"Completed for {ctype} cell")

# plot_stim_v_freq("cck", num_pts, conns="default",
#                      sdir=f"conn_{conn_file_num}_{n}/{conn_file_num}_{n}_stim_v_freq", stim_range=(-2, 2))
# plot_stim_v_freq("bic", num_pts, conns=new_conns,
#                      sdir=f"conn_{conn_file_num}_{n}/{conn_file_num}_{n}_stim_v_freq", stim_range=(-2, 2))
# for ctype in c_list:
#     plot_stim_v_freq(ctype, num_pts, conns="default",
#                      sdir=f"new_ref_set/ref_stim_v_freq", stim_range=(-2, 2))

n = 8
new_conns = conn_data[n]
plot_conn_v_theta_freq("pv", "cck", n_pts=121,
                       sdir=f"conn_10/conn_10_{n}/10_{n}_conn_v_power")
#
# plot_conn_v_theta_freq("pv", "pyr", num_pts, conns=new_conns,
#                        sdir=f"conn_{conn_file_num}_{n}/{conn_file_num}_{n}_conn_v_freq")
#
# plot_conn_v_theta_gamma("pv", "pyr", num_pts, conns=new_conns,
#                        sdir=f"conn_{conn_file_num}_{n}/{conn_file_num}_{n}_conn_v_power")

# plot_conn_v_theta_freq("pv", "pyr", num_pts, conns=None,
#                        sdir="new_ref_set/ref_conn_v_freq")

# plot_conn_v_theta_gamma("pv", "pyr", num_pts, conns=None,
#                        sdir="new_ref_set/ref_conn_v_power")

# ==============================================================================================
# DATA = pd.read_csv("/prm/sheets/freq_ranges_pyr_stim.csv", index_col=0)
# DATA = DATA.applymap(lambda x: np.array(str(x).split(','), dtype='float32'))
#
# # do all the things for a given connection set
# # for n in [135]: #[0, 2, 3, 7, 8, 22, 38, 50, 103, 135]:
# for conn in DATA.index:
#     n = int(conn.split('_')[-1])
#     new_conns = conn_data[n]
#     try:
#         os.makedirs(f"./figures/conn_{conn_file_num}/conn_{conn_file_num}_{n}/{conn_file_num}_{n}_stim_v_freq/"
#                     f"higher_res/raw/")
#         os.makedirs(f"./figures/conn_{conn_file_num}/conn_{conn_file_num}_{n}/{conn_file_num}_{n}_stim_v_power/"
#                     f"higher_res/raw/")
#     except FileExistsError:
#         pass
# #
# #     ax1 = create_radar()
# #     plot_radar(new_conns, ax1, mode="relative")
# #     plt.savefig(f"./figures/conn_{conn_file_num}/conn_{conn_file_num}_{n}/{conn_file_num}_{n}_conns_radar.png")
#     run = 'pv_025'
#     for ctype in ["pyr"]:
#         plot_stim_theta_gamma(ctype, num_pts, conns=new_conns, run_type=run,
#                               exclude_invalid=True, stim_range=DATA.loc[conn][run],
#                               sdir=f"conn_{conn_file_num}/conn_{conn_file_num}_{n}/{conn_file_num}_{n}_stim_v_power/"
#                                    f"higher_res/")
#         plot_stim_v_freq(ctype, num_pts, conns=new_conns, run_type=run,
#                               exclude_invalid=True, stim_range=DATA.loc[conn][run],
#                               sdir=f"conn_{conn_file_num}/conn_{conn_file_num}_{n}/{conn_file_num}_{n}_stim_v_freq/"
#                                    f"higher_res/")
#         print(f"Completed for {ctype} cell")
#
#     print(f"Completed connection set 10-{n}")

# =====================================================================================================

# plt.show()
