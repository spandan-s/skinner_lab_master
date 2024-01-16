import os

import matplotlib.pyplot as plt
import numpy as np

from prm_v2 import *
from prm_htest import hypothesis_test

import pandas as pd

# print(plt.style.available)
plt.style.use("seaborn-poster")
plt.rcParams.update({"font.size": 20})


def plot_fig_2(conn_set="10_8"):
    conn_dataset, p_set = conn_set.split('_')
    with open(f"/home/spandans/skinner_lab_master/prm/search_results/search_results_conn_{conn_dataset}.json",
              "r") as f:
        conn_data = json.load(f)

    conns = conn_data[int(p_set)]

    hypothesis_test(conns, plot=True)

    plt.tight_layout()
    plt.savefig(f"/home/spandans/skinner_lab_master/prm/figures/new_figs/fig2.pdf")
    plt.close()

def plot_fig_3(conn_set="10"):
    with open(f"/home/spandans/skinner_lab_master/prm/search_results/search_results_conn_{conn_set}.json", "r") as f:
        conn_data = json.load(f)

    make_boxplot(conn_data)
    plt.ylabel("Synaptic Weight [au]")
    plt.tight_layout()
    plt.savefig("/home/spandans/skinner_lab_master/prm/figures/new_figs/fig3.pdf")
    plt.close()

    # ax1 = create_radar()
    #
    # for conn in conn_data:
    #     plot_radar(conn, ax1, mode="absolute", color='black')
    #
    # plt.savefig("/home/spandans/skinner_lab_master/prm/figures/new_figs/fig3_inset.pdf")
    # plt.close()


def plot_fig_4(fname="quantifying.xlsx"):
    DATA = pd.read_excel(fname)
    c_labels = ['PYR', 'BiC', 'CCK', 'PV']
    labels = ['ctype', 'min_stim', 'max_stim', 'range']

    RANGE = np.zeros((10, 4))
    for idx, cell in enumerate(c_labels):
        temp = DATA.loc[DATA.ctype == cell.lower()]['range']
        RANGE[:, idx] = temp

    myparams = dict(
        xlabel="Cell Type",
        ylabel="Range of STIM [au]",
        c_labels=c_labels
    )
    fig, ax = plt.subplots(figsize=(4.8, 6.4))
    make_violin(ax, RANGE, **myparams)

    plt.tight_layout()
    plt.savefig("/home/spandans/skinner_lab_master/prm/figures/new_figs/fig4.pdf")


def adjacent_values(vals, q1, q3):
    upper_adjacent_value = q3 + (q3 - q1) * 1.5
    upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

    lower_adjacent_value = q1 - (q3 - q1) * 1.5
    lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
    return lower_adjacent_value, upper_adjacent_value


def make_violin(ax, data,
                ylabel=None, xlabel=None, c_labels=[]):
    parts = ax.violinplot(data, showmedians=False, showextrema=True, widths=0.67)
    parts['cbars'].set_edgecolor('black')
    parts['cmins'].set_edgecolor('black')
    parts['cmaxes'].set_edgecolor('black')

    for pc in parts['bodies']:
        pc.set_facecolor('black')
        # pc.set_edgecolor('black')
        pc.set_linewidth(0.67)
        # pc.set_edgecolor('black')
        # pc.set_alpha(0.6)
    quartile1, medians, quartile3 = np.percentile(data, [25, 50, 75], axis=0)

    inds = np.arange(1, len(medians) + 1)
    ax.scatter(inds, medians, marker='o', color='w', s=40, zorder=3)
    ax.vlines(inds, quartile1, quartile3, color='black', linestyle='-', lw=8)

    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)

    ax.set_xticks(np.arange(1, len(c_labels) + 1), labels=c_labels)
    ax.set_xlim(0.25, len(c_labels) + 0.75)


def plot_fig_5(n, ctype):
    DATA = np.loadtxt(f'./figures/conn_10/conn_10_{n}/10_{n}_stim_v_freq/raw/'
                      f'stim_to_baseline_{ctype}_cell_theta_freq_121.dat',
                      skiprows=1)

    # DATA = np.loadtxt(f'./figures/new_ranges/conn_10_{n}/10_{n}_conn_v_freq/raw/'
    #                   f'w{ctype}_theta_freq.dat',
    #                   skiprows=1)

    sdir = f'figures/plots_v2/conn_10_{n}/stim_freq_power'
    save_name = f'stim_to_{ctype}_freq_power.png'

    xlabel = {
        "pyr": "$STIM_{PYR}$ [au]",
        "bic": "$STIM_{BiC}$ [au]",
        "cck": "$STIM_{CCK}$ [au]",
        "pv": "$STIM_{PV} [au]$"
    }

    # c1, c2 = ctype.split('_')

    stim = DATA[:, 0]
    theta_freq = DATA[:, 1]
    theta = DATA[:, 2]
    gamma = DATA[:, 3]
    LPR = DATA[:, 4]


    # find points where theta and gamma are valid
    invalid = (theta < 48) | (gamma < 0.31)

    x_valid = np.where(invalid, np.nan, stim)
    theta_freq_valid = np.where(invalid, np.nan, theta_freq)
    theta_valid = np.where(invalid, np.nan, theta)
    gamma_valid = np.where(invalid, np.nan, gamma)

    # find theta freq and power slope, gamma power slope
    X = x_valid[~np.isnan(x_valid)]
    Y = theta_freq_valid[~np.isnan(theta_freq_valid)]
    Y1 = theta_valid[~np.isnan(theta_valid)]
    Y2 = gamma_valid[~np.isnan(gamma_valid)]
    try:
        a1_tf, a0_tf = np.polyfit(X, Y, 1)
        a1_tp, a0_tp = np.polyfit(X, Y1, 1)
        a1_gp, a0_gp = np.polyfit(X, Y2, 1)
    except LinAlgError:
        a1_tf, a0_tf = np.nan, np.nan

    xlim = [np.min(X - (X[2] - X[0])), np.max(X + (X[2] - X[0]))]
    ylim_theta_freq = [np.min(Y) - 1, np.max(Y) + 1]
    # ==============================================================================
    fig, ax = plt.subplots(nrows=2, sharex=True, figsize=[9, 6.4], dpi=250)
    # First subplot --> theta frequency vs stim
    p1, = ax[0].plot(stim, theta_freq,
                     ls='--', marker='.', color='black')  # plot theta freq
    ax[0].plot(x_valid, theta_freq_valid,
               ls="", marker=".", color="C3")
    ax[0].plot(X, a0_tf + a1_tf * X,
               ls="dotted", color="purple")

    # subplot 1 axes and labels
    # ax[0].set_title(f"Stimulation to {ctype} cell".upper())
    # ax[0].set_title(f"$w_{{{c1.upper()} \\rightarrow {c2.upper()}}}$")
    ax[0].set_xlim(xlim)
    ax[0].set_ylabel("Theta Frequency [Hz]")

    ax[0].yaxis.label.set_color(p1.get_color())

    ax[0].tick_params(axis='y', colors=p1.get_color())
    ax[0].grid(axis='x')
    # end of first subplot
    # ==============================================================================

    # ==============================================================================
    # second subplot --> log power ratio vs stim
    ax1 = ax[1].twinx()
    p1, = ax[1].plot(stim, theta,
                     ls='--', marker='.', color='black')  # plot theta power
    # ax[1].axhline(48, color='C0', ls='--')
    # ax[1].plot(x_valid, theta_valid,
    #            ls="", marker=".", color="C3")
    p2, = ax1.plot(stim, gamma,
                   ls='--', marker='.', color='grey')  # plot gamma power
    # ax1.axhline(0.31, color='C1', ls='--')
    # ax1.plot(x_valid, gamma_valid,
    #            ls="", marker=".", color="C3")

    # subplot 1 axes and labels
    ax[1].set_ylabel("Theta Power")
    ax1.set_ylabel("Gamma Power")
    ax[1].set_xlabel(xlabel[ctype])

    ax[1].yaxis.label.set_color(p1.get_color())
    ax1.yaxis.label.set_color(p2.get_color())

    ax[1].tick_params(axis='y', colors=p1.get_color())
    ax1.tick_params(axis='y', colors=p2.get_color())
    ax[1].grid(axis='x')

    # ax.grid()
    plt.tight_layout()
    plt.savefig(f"/home/spandans/skinner_lab_master/prm/figures/new_figs/fig5_{ctype}.pdf")
    # try:
    #     plt.savefig(f"./{sdir}/{save_name}")
    # except FileNotFoundError:
    #     os.makedirs(f"./{sdir}")
    #     plt.savefig(f"./{sdir}/{save_name}")

    plt.close()

def plot_fig_5a(n, conn):
    DATA = np.loadtxt(f'./figures/conn_10/conn_10_{n}/10_{n}_conn_v_power/raw/'
                      f'w{conn}_theta_freq.dat',
                      skiprows=1)

    # DATA = np.loadtxt(f'./figures/new_ranges/conn_10_{n}/10_{n}_conn_v_freq/raw/'
    #                   f'w{ctype}_theta_freq.dat',
    #                   skiprows=1)

    sdir = f'figures/new_figs/'
    save_name = f'fig_5_supp_w_{conn}.png'

    xlabel = {
        "pv_pyr": "$w_{PV \\rightarrow PYR}$ [au]",
        "pyr_pv": "$w_{PYR \\rightarrow PV}$ [au]",
        "pv_cck": "$w_{PV \\rightarrow CCK}$ [au]",
        "cck_pv": "$w_{CCK \\rightarrow PV}$ [au]",

    }

    # c1, c2 = ctype.split('_')

    weight = DATA[:, 0]
    theta_freq = DATA[:, 1]
    theta = DATA[:, 2]
    gamma = DATA[:, 3]
    # LPR = DATA[:, 4]


    # find points where theta and gamma are valid
    invalid = (theta < 48) | (gamma < 0.31)

    x_valid = np.where(invalid, np.nan, weight)
    theta_freq_valid = np.where(invalid, np.nan, theta_freq)
    theta_valid = np.where(invalid, np.nan, theta)
    gamma_valid = np.where(invalid, np.nan, gamma)

    # find theta freq and power slope, gamma power slope
    X = x_valid[~np.isnan(x_valid)]
    Y = theta_freq_valid[~np.isnan(theta_freq_valid)]
    Y1 = theta_valid[~np.isnan(theta_valid)]
    Y2 = gamma_valid[~np.isnan(gamma_valid)]
    try:
        a1_tf, a0_tf = np.polyfit(X, Y, 1)
        a1_tp, a0_tp = np.polyfit(X, Y1, 1)
        a1_gp, a0_gp = np.polyfit(X, Y2, 1)
    except LinAlgError:
        a1_tf, a0_tf = np.nan, np.nan

    xlim = [np.min(X - (X[2] - X[0])), np.max(X + (X[2] - X[0]))]
    ylim_theta_freq = [np.min(Y) - 1, np.max(Y) + 1]
    # ==============================================================================
    fig, ax = plt.subplots(nrows=1, sharex=True, figsize=[12.8, 4.8], dpi=250)
    # First subplot --> theta frequency vs stim
    p1, = ax.plot(weight, theta_freq,
                     ls='--', marker='.', color='black')  # plot theta freq
    ax.plot(x_valid, theta_freq_valid,
               ls="", marker=".", color="C3")
    ax.plot(X, a0_tf + a1_tf * X,
               ls="dotted", color="purple")

    # subplot 1 axes and labels
    # ax[0].set_title(f"Stimulation to {ctype} cell".upper())
    # ax[0].set_title(f"$w_{{{c1.upper()} \\rightarrow {c2.upper()}}}$")
    ax.set_xlim(xlim)
    ax.set_ylabel("Theta Frequency [Hz]")

    ax.yaxis.label.set_color(p1.get_color())
    ax.set_xlabel(xlabel[conn])

    ax.tick_params(axis='y', colors=p1.get_color())
    ax.grid(axis='x')
    # end of first subplot
    # ==============================================================================

    # ==============================================================================
    # second subplot --> log power ratio vs stim
    # ax1 = ax[1].twinx()
    # p1, = ax[1].plot(weight, theta,
    #                  ls='--', marker='.', color='black')  # plot theta power
    # # ax[1].axhline(48, color='C0', ls='--')
    # # ax[1].plot(x_valid, theta_valid,
    # #            ls="", marker=".", color="C3")
    # p2, = ax1.plot(weight, gamma,
    #                ls='--', marker='.', color='grey')  # plot gamma power
    # # ax1.axhline(0.31, color='C1', ls='--')
    # # ax1.plot(x_valid, gamma_valid,
    # #            ls="", marker=".", color="C3")
    #
    # # subplot 1 axes and labels
    # ax[1].set_ylabel("Theta Power")
    # ax1.set_ylabel("Gamma Power")
    # ax[1].set_xlabel(xlabel[conn])
    #
    # ax[1].yaxis.label.set_color(p1.get_color())
    # ax1.yaxis.label.set_color(p2.get_color())
    #
    # ax[1].tick_params(axis='y', colors=p1.get_color())
    # ax1.tick_params(axis='y', colors=p2.get_color())
    # ax[1].grid(axis='x')

    # ax.grid()
    plt.tight_layout()
    plt.savefig(f"/home/spandans/skinner_lab_master/prm/figures/new_figs/fig5_supp_{conn}.pdf")
    # try:
    #     plt.savefig(f"./{sdir}/{save_name}")
    # except FileNotFoundError:
    #     os.makedirs(f"./{sdir}")
    #     plt.savefig(f"./{sdir}/{save_name}")

    plt.close()


def plot_fig_6(fname="table1.ods"):
    DATA = pd.read_excel(f'~/Downloads/{fname}')

    labels = ['cell_type', 'theta_freq_slope', 'theta_power_slope', 'gamma_power_slope']
    c_labels = ['PYR', 'BiC', 'CCK', 'PV']
    # labels = ['ctype', 'min_stim', 'max_stim', 'range']

    TF = np.zeros((10, 4))
    TP = np.zeros((10, 4))
    GP = np.zeros((10, 4))

    for idx, ctype in enumerate(c_labels):
        TF[:, idx] = DATA.loc[DATA.cell_type == ctype.lower()]['theta_freq_slope']
        TP[:, idx] = DATA.loc[DATA.cell_type == ctype.lower()]['theta_power_slope']
        GP[:, idx] = DATA.loc[DATA.cell_type == ctype.lower()]['gamma_power_slope']

    tf_params = dict(
        xlabel="Cell Type",
        ylabel="Slope of Theta Frequency vs STIM",
        c_labels=c_labels
    )
    tp_params = dict(
        xlabel="Cell Type",
        ylabel="Slope of Theta Power vs STIM",
        c_labels=c_labels
    )
    gp_params = dict(
        xlabel="Cell Type",
        ylabel="Slope of Gamma Power vs STIM",
        c_labels=c_labels
    )
    fig, ax = plt.subplots(figsize=(4.8, 6.4), dpi=400)
    make_violin(ax, TF, **tf_params)
    plt.tight_layout()
    plt.savefig("/home/spandans/skinner_lab_master/prm/figures/new_figs/fig6a.pdf")

    fig, ax = plt.subplots(figsize=(4.8, 6.4), dpi=400)
    make_violin(ax, TP, **tp_params)
    plt.tight_layout()
    plt.savefig("/home/spandans/skinner_lab_master/prm/figures/new_figs/fig6b.pdf")

    fig, ax = plt.subplots(figsize=(4.8, 6.4), dpi=400)
    make_violin(ax, GP, **gp_params)
    plt.tight_layout()
    plt.savefig("/home/spandans/skinner_lab_master/prm/figures/new_figs/fig6c.pdf")


def plot_fig_7(fname="table2.ods"):
    DATA = pd.read_excel(f'~/Downloads/{fname}')

    labels = ['cell_type', 'theta_freq_slope', 'theta_power_slope', 'gamma_power_slope']
    conns = [
        "pyr_pyr", "pv_pyr", "bic_pyr",
        "pyr_bic",
        "pyr_pv", "pv_pv", "cck_pv",
        "pv_cck", "cck_cck",
    ]

    # plot_labels = [
    #     "$PYR \\rightarrow PYR$", "$PV \\rightarrow PYR$", "$BiC \\rightarrow PYR$",
    #     "$PYR \\rightarrow BiC$",
    #     "$PYR \\rightarrow PV$", "$PV \\rightarrow PV$", "$CCK \\rightarrow PV$",
    #     "$PV \\rightarrow CCK$", "$CCK \\rightarrow CCK$",
    # ]

    plot_labels = [
        "$PYR$\n\n", "$PV$", "$BiC$",
        "$PYR$",
        "$PYR$", "$PV$", "$CCK$",
        "$PV$", "$CCK$",
    ]

    TF = np.zeros((10, 9))
    TP = np.zeros((10, 9))
    GP = np.zeros((10, 9))

    for idx, conn in enumerate(conns):
        TF[:, idx] = DATA.loc[DATA.cell_type == conn]['theta_freq_slope']
        TP[:, idx] = DATA.loc[DATA.cell_type == conn]['theta_power_slope']
        GP[:, idx] = DATA.loc[DATA.cell_type == conn]['gamma_power_slope']

    tf_params = dict(
        xlabel="Connection",
        ylabel="Slope of Theta Frequency vs Synaptic Weight",
        c_labels=plot_labels
    )
    tp_params = dict(
        xlabel="Connection",
        ylabel="Slope of Theta Power vs Synaptic Weight",
        c_labels=plot_labels
    )
    gp_params = dict(
        xlabel="Connection",
        ylabel="Slope of Gamma Power vs Synaptic Weight",
        c_labels=plot_labels
    )

    fig, ax = plt.subplots(figsize=(10, 10), dpi=400)
    make_violin(ax, TF, **tf_params)
    plt.axvline(3.5, linestyle='--', color='black')
    plt.axvline(4.5, linestyle='--', color='black')
    plt.axvline(7.5, linestyle='--', color='black')
    plt.tight_layout()
    plt.savefig("/home/spandans/skinner_lab_master/prm/figures/new_figs/fig7a.pdf")

    fig, ax = plt.subplots(figsize=(10, 10), dpi=400)
    make_violin(ax, TP, **tp_params)
    ax.set_ylim(bottom=-400)
    plt.axvline(3.5, linestyle='--', color='black')
    plt.axvline(4.5, linestyle='--', color='black')
    plt.axvline(7.5, linestyle='--', color='black')
    plt.tight_layout()
    plt.savefig("/home/spandans/skinner_lab_master/prm/figures/new_figs/fig7b.pdf")

    fig, ax = plt.subplots(figsize=(10, 10), dpi=400)
    make_violin(ax, GP, **gp_params)
    plt.axvline(3.5, linestyle='--', color='black')
    plt.axvline(4.5, linestyle='--', color='black')
    plt.axvline(7.5, linestyle='--', color='black')
    plt.tight_layout()
    plt.savefig("/home/spandans/skinner_lab_master/prm/figures/new_figs/fig7c.pdf")

    plt.close("all")

def plot_fig_100_a():
    os.chdir("/home/spandans/skinner_lab_master/prm/figures/PRM_figures/figure_100/panel_a")

    filled_marker_style = dict(
        marker='o', linestyle='--',
        markerfacecolor='black',
        markeredgecolor='black'
    )

    fig, ax = plt.subplots(figsize=(8, 8))
    ax1 = ax.twinx()

    cck_pv = np.loadtxt("./CCK_PV.txt", skiprows=1)
    pyr_pv = np.loadtxt("./PYR_PV.txt", skiprows=1)
    pyr_bic = np.loadtxt("./PYR_BiC.txt", skiprows=1)

    t1 = [2.49257761675845, 104.266066878981]
    t2 = [0.9321, 275.770506660552]
    t3 = [6.38, 11.1925443850396]

    l1 = ax.plot(cck_pv[:, 0], cck_pv[:, 1], label='CCK+PV',
            color='black', fillstyle='full', **filled_marker_style)
    l2 = ax.plot(pyr_pv[:, 0], pyr_pv[:, 1], label='PYR+PV',
            color='black', fillstyle='none', **filled_marker_style)
    l3 = ax1.plot(pyr_bic[:, 0], pyr_bic[:, 1], label='PYR+BiC',
             color='darkgrey', marker='o', linestyle='--')
    ax.plot(*t1, color='C2', marker='^', markersize=15)
    ax.plot(*t2, color='C3', marker='^', markersize=15)
    ax1.plot(*t3, color='C1', marker='^', markersize=15)

    ax.set_ylim((0, 300))
    ax1.set_ylim((9.5, 11.5))
    ax.set_xlim((0, 10.2))

    lns = l1 + l2 + l3
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc=7)

    ax.yaxis.label.set_color(l1[0].get_color())
    ax1.yaxis.label.set_color(l3[0].get_color())

    ax.tick_params(axis='y', colors=l1[0].get_color())
    ax1.tick_params(axis='y', colors=l3[0].get_color())

    ax.set_xlabel("delay [ms]")
    ax.set_ylabel("frequency [Hz]")
    ax1.set_ylabel("frequency [Hz]")

    plt.tight_layout()
    plt.savefig("/home/spandans/skinner_lab_master/prm/figures/new_figs/fig100_a.pdf")

def plot_fig_100_b():
    os.chdir("/home/spandans/skinner_lab_master/prm/figures/PRM_figures/"
             "figure_100/panel_b_c_d/rPYR-rBiC/")

    tau = [5, 8, 10]
    colors = ['xkcd:light grey', 'xkcd:grey', 'xkcd:dark grey']

    fig, ax = plt.subplots(figsize=(8, 8))

    for t, col in zip(tau, colors):
        data = np.loadtxt(f"./rPYR-rBiC tau{t}.txt", skiprows=1)

        ax.plot(data[:, 0], data[:, 1], label=f"{t} ms", color=col)

    ax.set_xlim((0, 50))
    ax.set_ylim((0, 100))
    ax.set_ylabel("$r_{BiC}$ [Hz]")
    ax.set_xlabel("$r_{PYR}$ [Hz]")

    ax.legend(loc=2)
    plt.tight_layout()
    plt.savefig("/home/spandans/skinner_lab_master/prm/figures/new_figs/fig100_b.pdf")


def plot_fig_100_c():
    os.chdir("/home/spandans/skinner_lab_master/prm/figures/PRM_figures/"
             "figure_100/panel_b_c_d/r_PYR-r_PV/")

    tau = [2, 5, 8]
    colors = ['xkcd:light grey', 'xkcd:grey', 'xkcd:dark grey']

    fig, ax = plt.subplots(figsize=(8, 8))

    for t, col in zip(tau, colors):
        data = np.loadtxt(f"./rPYR-rPV tau{t}.txt", skiprows=1)

        ax.plot(data[:, 0], data[:, 1], label=f"{t} ms", color=col)

    ax.set_xlim((0, 40))
    ax.set_ylim((0, 60))
    ax.set_ylabel("$r_{PV}$ [Hz]")
    ax.set_xlabel("$r_{PYR}$ [Hz]")

    ax.legend(loc=2)
    plt.tight_layout()
    plt.savefig("/home/spandans/skinner_lab_master/prm/figures/new_figs/fig100_c.pdf")


def plot_fig_100_d():
    os.chdir("/home/spandans/skinner_lab_master/prm/figures/PRM_figures/"
             "figure_100/panel_b_c_d/r_CCK-r_PV/")

    tau = [2, 5, 8]
    colors = ['xkcd:light grey', 'xkcd:grey', 'xkcd:dark grey']

    fig, ax = plt.subplots(figsize=(8, 8))

    for t, col in zip(tau, colors):
        data = np.loadtxt(f"./rCCK-rPV tau{t}.txt", skiprows=1)

        ax.plot(data[:, 0], data[:, 1], label=f"{t} ms", color=col)

    ax.set_xlim((0, 20))
    ax.set_ylim((0, 15))
    ax.set_ylabel("$r_{PV}$ [Hz]")
    ax.set_xlabel("$r_{CCK}$ [Hz]")

    ax.legend(loc=2)
    plt.tight_layout()
    plt.savefig("/home/spandans/skinner_lab_master/prm/figures/new_figs/fig100_d.pdf")

def plot_supp_fig_7(conn_set=8, ctype='pyr'):
    with open("search_results/search_results_conn_10.json", "r") as f:
        conn_data = json.load(f)

    stim_range = {
        "pyr": [-0.033, 0.00, 0.033, 0.067],
        "cck": [-0.133, 0.0, 0.133, 0.233]
    }

    stim = {'pyr': np.zeros_like(time),
            'bic': np.zeros_like(time),
            'pv': np.zeros_like(time),
            'cck': np.zeros_like(time)
            }
    test_prm = PRM_v2(conn_data[conn_set])
    pst = 7 * len(time)//8

    fig, ax = plt.subplots(2, 4, figsize=(16, 9))

    for idx, stim_val in enumerate(stim_range[ctype]):
        stim[ctype] = stim_val + np.zeros_like(time)

        test_prm.set_init_state(len(time))
        test_prm = simulate(time, test_prm, stim=stim)
        psd_out = find_psd(test_prm.R, fs)[:2]

        ax[0][idx].plot(time[pst:], test_prm.R["pyr"][pst:])
        ax[1][idx].plot(psd_out[0], psd_out[1])

        ax[0][idx].set_title("$STIM_{CCK}=$"+str(stim_val))
        ax[1][idx].set_xlim(0, 60)

        ax[0][idx].set_xlabel("Time [s]")
        ax[1][idx].set_xlabel("Frequency [Hz]")

    ax[0][0].set_ylabel("$PYR$ Activity [Hz]")
    ax[1][0].set_ylabel("Power Spectral Density")

    fig.tight_layout()
    plt.savefig(f"figures/new_figs/supp_fig_7_{ctype}.pdf", dpi=400)

def plot_supp_fig_7b(conn_set="10_22"):
    conn_dataset, p_set = conn_set.split('_')
    with open(f"/home/spandans/skinner_lab_master/prm/search_results/search_results_conn_{conn_dataset}.json",
              "r") as f:
        conn_data = json.load(f)

    conns = conn_data[int(p_set)]

    P = PRM_v2(conns)
    P.set_init_state(len(time))
    P = simulate(time, P)

    plot_trace(time, P.R, P.labels, 'all')
    plt.xlim([7, 7.2])
    plt.savefig(f"figures/new_figs/supp_fig_7b_3.pdf", dpi=400)


# plot_fig_100_a()
# plot_fig_100_b()
# plot_fig_100_c()
# plot_fig_100_d()
# plot_fig_2()
# plot_fig_3(conn_set=13)
# plot_fig_4()
# plot_fig_5(8, "pyr")
# plot_fig_5a(8, "pv_pyr")
# plot_fig_6()
# plot_fig_7()
# plot_supp_fig_7(ctype='cck')
plot_supp_fig_7b()

plt.show()
