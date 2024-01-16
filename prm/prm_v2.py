import json
from copy import deepcopy

import numpy as np
import scipy.signal
import sympy
from matplotlib import pyplot as plt
from numpy.linalg import LinAlgError
from scipy import signal
from sympy import Symbol, symbols, nsolve

from f_filter_lfp import filter_lfp


class PRM_v2:
    """

    """

    def __init__(self, conns_dict="default", input_dict="default"):
        self.R = None

        self.labels = {
            "pyr": "PYR",
            "bic": "BiC",
            "pv": "PV",
            "cck": "CCK"
        }

        self._set_timescales()
        self.set_connections(conns_dict)
        self._set_inputs(input_dict)

    def _set_timescales(self):
        self.alpha = {
            "pyr": 40,
            "bic": 80,
            "pv": 80,
            "cck": 40
        }
        self.r_o = {
            "pyr": 40,
            "bic": 100,
            "pv": 100,
            "cck": 60
        }

    def set_connections(self, param_dict="default"):
        if param_dict == "default":
            self.conns = {
                "pyr": {
                    "pyr": 0.05, "bic": 0.04, "pv": 0.02, "cck": 0.0
                },
                "bic": {
                    "pyr": -0.02, "bic": 0.0, "pv": 0.0, "cck": 0.0
                },
                "pv": {
                    "pyr": -0.03, "bic": 0.0, "pv": -0.055, "cck": -0.075
                },
                "cck": {
                    "pyr": 0.0, "bic": 0.0, "pv": -0.15, "cck": -0.15
                },
            }
        else:
            self.conns = deepcopy(param_dict)

    def _set_inputs(self, param_dict='default'):
        self.D = {
            "pyr": 0.00,
            "bic": 0.00,
            "pv": 0.00,
            "cck": 0.00
        }
        if param_dict == "default":
            self.I = {
                "pyr": 0.03,
                "bic": -1.45,
                "pv": 0.5,
                "cck": 0.8,
            }
        else:
            self.I = deepcopy(param_dict)

    def set_init_state(self, n_samples,
                       ICs = [0, 0, 0, 0]):
        self.R = {
            "pyr": np.zeros(n_samples),
            "bic": np.zeros(n_samples),
            "pv": np.zeros(n_samples),
            "cck": np.zeros(n_samples),
        }

        for c, ic in zip(['pyr', 'bic', 'cck', 'pv'], ICs):
            self.R[c][0] = ic


def f(u, beta=20, h=0):
    return 1 / (1 + np.exp(-beta * (u - h)))

def f_sp(u, beta=20, h=0):
    return 1 / (1 + sympy.exp(-beta * (u - h)))


def simulate(time, P, dt=0.001, tau=5, stim=None):
    if stim == None:
        stim = {
            "pyr": 0 + np.zeros_like(time),
            "bic": 0 + np.zeros_like(time),
            "pv": 0 + np.zeros_like(time),
            "cck": 0 + np.zeros_like(time)
        }
    # euler_integrate
    for t in range(len(time) - 1):
        c_list = ["pyr", "bic", "pv", "cck"]
        for c1 in c_list:
            P.R[c1][t + 1] = P.R[c1][t] + dt * P.alpha[c1] * \
                             (-P.R[c1][t] + P.r_o[c1] * f(sum((P.conns[c2][c1] * P.R[c2][t - tau]) for c2 in c_list) + P.I[c1] + stim[c1][t])) + \
                             np.sqrt(2 * P.alpha[c1] * P.D[c1] * dt) * np.random.normal(0, 1)
    return P

def baseline_sim(P, IC = [0, 0, 0, 0]):
    # initialise and simulate PRM at baseline
    P.set_init_state(len(time), IC)
    P = simulate(time, P)
    ret = P.R
    return ret

def calc_spectral(R, fs, time,
                  band='theta', mode=None, plot_Fig=False, plot_Filter=False, labels=None):
    # choose which band of oscillations you want to filter
    if band == 'theta':
        cutoff = np.array([3, 12])
    elif band == 'gamma':
        # should this be changed - maybe?
        cutoff = np.array([20, 100])

    c_list = ["pyr", "bic", "pv", "cck"]

    if labels == None:
        labels = {
            "pyr": "PYR",
            "bic": "BiC",
            "pv": "PV",
            "cck": "CCK"
        }

    R_filt = {
        "pyr": filter_lfp(R["pyr"], fs, cutoff),
        "bic": filter_lfp(R["bic"], fs, cutoff),
        "pv": filter_lfp(R["pv"], fs, cutoff),
        "cck": filter_lfp(R["cck"], fs, cutoff),
    }

    if plot_Filter:
        plot_start_time = 3 * time.size // 4
        plt.figure(figsize=[13, 8])
        for c in c_list:
            plt.plot(time[plot_start_time:], R_filt[c][plot_start_time:], label=labels[c])

    segment = int(fs * 4)
    myhann = signal.get_window('hann', segment)

    myparams = dict(fs=fs, nperseg=segment, window=myhann,
                    noverlap=segment / 2, scaling='density', return_onesided=True)

    pgram = {
        "pyr": signal.welch(R_filt["pyr"], **myparams),
        "bic": signal.welch(R_filt["bic"], **myparams),
        "pv": signal.welch(R_filt["pv"], **myparams),
        "cck": signal.welch(R_filt["cck"], **myparams),
    }
    # pgram = {
    #     "pyr": signal.periodogram(R_filt["pyr"], fs),
    #     "bic": signal.periodogram(R_filt["bic"], fs),
    #     "pv": signal.periodogram(R_filt["pv"], fs),
    #     "cck": signal.periodogram(R_filt["cck"], fs),
    # }

    if plot_Fig:
        plt.figure(figsize=[13, 8])
        for c in c_list:
            plt.plot(pgram[c][0], pgram[c][1], label=labels[c])
        if band == 'theta':
            plt.xlim(0, 20)
        elif band == 'gamma':
            plt.xlim(0, 100)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Spectral Power')
        plt.legend()

    peaks = {
        "pyr": signal.find_peaks(pgram["pyr"][1])[0],
        "bic": signal.find_peaks(pgram["bic"][1])[0],
        "pv": signal.find_peaks(pgram["pv"][1])[0],
        "cck": signal.find_peaks(pgram["cck"][1])[0]
    }

    if mode == "peak_freq":
        fm = {
            "pyr": pgram["pyr"][0][peaks["pyr"]][np.argmax(pgram["pyr"][1][peaks["pyr"]])],
            "bic": pgram["bic"][0][peaks["bic"]][np.argmax(pgram["bic"][1][peaks["bic"]])],
            "pv": pgram["pv"][0][peaks["pv"]][np.argmax(pgram["pv"][1][peaks["pv"]])],
            "cck": pgram["cck"][0][peaks["cck"]][np.argmax(pgram["cck"][1][peaks["cck"]])],
        }
        return fm

    elif mode == "power":
        power = {
            "pyr": np.max(pgram["pyr"][1][peaks["pyr"]]),
            "bic": np.max(pgram["bic"][1][peaks["bic"]]),
            "pv": np.max(pgram["pv"][1][peaks["pv"]]),
            "cck": np.max(pgram["cck"][1][peaks["cck"]]),
        }
        return power

    else:
        fm = {
            "pyr": pgram["pyr"][0][peaks["pyr"]][np.argmax(pgram["pyr"][1][peaks["pyr"]])],
            "bic": pgram["bic"][0][peaks["bic"]][np.argmax(pgram["bic"][1][peaks["bic"]])],
            "pv": pgram["pv"][0][peaks["pv"]][np.argmax(pgram["pv"][1][peaks["pv"]])],
            "cck": pgram["cck"][0][peaks["cck"]][np.argmax(pgram["cck"][1][peaks["cck"]])],
        }
        power = {
            "pyr": np.max(pgram["pyr"][1][peaks["pyr"]]),
            "bic": np.max(pgram["bic"][1][peaks["bic"]]),
            "pv": np.max(pgram["pv"][1][peaks["pv"]]),
            "cck": np.max(pgram["cck"][1][peaks["cck"]]),
        }
        return fm, power


def plot_trace(time, R, labels, mode="pyr"):
    plot_start_time = 7 * time.size // 8
    plt.figure(figsize=[13, 8])
    if mode == "pyr":
        c_list = ["pyr"]
    else:
        c_list = ["pyr", "bic", "cck", "pv"]
    for c in c_list:
        plt.plot(time[plot_start_time:], R[c][plot_start_time:], label=labels[c])

    plt.xlabel('Time [s]')
    plt.ylabel('Activity [Hz]')
    plt.legend()


def valid_oscillation(R, fs, ref=None):
    if ref == None:
        ref = ref_power()

    tpp = find_pyr_power(R, fs, 'theta')[1]
    gpp = find_pyr_power(R, fs, 'gamma')[1]

    if (tpp >= (0.20 * ref[0])) and (gpp >= (0.2 * ref[1])):
        return [tpp, gpp]
    else:
        return [np.nan, np.nan]


def run_prm(conns=None, I=None, dt=0.001, T=8.0,
            stim=None, plot=False):
    if conns == None:
        conns = "default"
    if I == None:
        I = "default"

    if stim == None:
        stim = {"pyr": 0, "bic": 0, "cck": 0, "pv": 0}
    fs = 1 / dt
    new_prm = PRM_v2(conns, I)

    time = np.arange(0, T, dt)

    for c in ["pyr", "bic", "cck", "pv"]:
        stim[c] += np.zeros_like(time)

    new_prm.set_init_state(len(time))
    new_prm = simulate(time, new_prm, stim=stim)

    if plot:
        plot_trace(time, new_prm.R, new_prm.labels)

    tf, tpp = find_pyr_power(new_prm.R, fs, 'theta')
    gf, gpp = find_pyr_power(new_prm.R, fs, 'gamma')

    return [tf, tpp], [gf, gpp]

def import_conns(n=0):
    # import conns
    with open(f"/home/spandans/Documents/Skinner_Lab/skinner_lab_master/prm/search_results/search_results_conn_10.json",
              "r") as f:
        conn_data = json.load(f)

    # create PRM
    P = PRM_v2(conns_dict=conn_data[n])
    return P

def pv_bic_ratio(R):

    max_bic = np.max(R["bic"])
    max_pv = np.max(R["pv"])

    return max_pv/max_bic

def add_to_plot(conns, label, pcolor=None):
    v = []
    for c in [*conns]:
        v.append([*conns[c].values()])

    v = np.array(v).reshape(16)

def create_radar():
    ref = PRM_v2()
    v = []
    for c in [*ref.conns]:
        v.append([*ref.conns[c].values()])
    v = np.array(v).reshape(16)
    c_list = ["pyr", "bic", "pv", "cck"]
    conn_labels = []
    for c in c_list:
        for c2 in c_list:
            conn_labels.append(f"$w_{{{c.upper()} \\rightarrow {c2.upper()}}}$")

    invalid = np.where(v == 0)
    conn_labels = np.delete(conn_labels, invalid)
    num_labels = len(conn_labels)

    angles = np.linspace(0, 2 * np.pi, num_labels, endpoint=False)
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True), dpi=400)
    # line up first datapoint with vertical axis
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # clean up the labels so they don't overlap with the grid lines
    ax.set_thetagrids(np.degrees(angles), conn_labels)
    for label, angle in zip(ax.get_xticklabels(), angles):
        if angle in (0, np.pi):
            label.set_horizontalalignment('center')
        elif 0 < angle < np.pi:
            label.set_horizontalalignment('left')
        else:
            label.set_horizontalalignment('right')

    ax.set_ylim(0, 3)

    ax.set_rlabel_position(180 / num_labels)
    # Add some custom styling.
    # Change the color of the tick labels.
    ax.tick_params(colors='#222222')
    # Make the y-axis (0-100) labels smaller.
    ax.tick_params(axis='y', labelsize=8)
    # Change the color of the circular gridlines.
    ax.grid(color='#AAAAAA')
    # Change the color of the outermost gridline (the spine).
    ax.spines['polar'].set_color('#222222')
    # Change the background color inside the circle itself.
    ax.set_facecolor('#FAFAFA')
    plt.tight_layout()

    return ax

def plot_radar(in_conns, ax, mode="absolute", label=None, color=None):
    conns = deepcopy(in_conns)
    v = []

    for c in [*conns]:
        v.append([*conns[c].values()])
    v = np.array(v).reshape(16)

    if mode=="relative":
        ref = PRM_v2()
        ref_conns = deepcopy(ref.conns)
        v_ref = []

        for c in [*ref_conns]:
            v_ref.append([*ref_conns[c].values()])
        v_ref = np.array(v_ref).reshape(16)

    c_list = ["pyr", "bic", "pv", "cck"]

    conn_labels = []
    for c in c_list:
        for c2 in c_list:
            conn_labels.append(f"$w_{{{c.upper()} \\rightarrow {c2.upper()}}}$")

    invalid = np.where(v == 0)

    v = np.delete(v, invalid)
    if mode == "relative":
        v_ref = np.delete(v_ref, invalid)
    conn_labels = np.delete(conn_labels, invalid)

    num_labels = len(conn_labels)

    angles = np.linspace(0, 2 * np.pi, num_labels, endpoint=False)
    angles = np.append(angles, angles[0])
    if mode=="absolute":
        v = np.append(v, v[0])
        v_plot = 10*abs(v)
    elif mode=="relative":
        v_ratio = v/v_ref
        v_plot = np.append(v_ratio, v_ratio[0])
        # ax.set_ylim(0, 3)

    ax.plot(angles, v_plot, color=color, linewidth=0.1, label=label, alpha=0.2)
    # Fill it in.
    ax.fill(angles, v_plot, color=color, alpha=0.1)


def plot_conns(in_conns, in_ref_conns = None):
    if in_ref_conns == None:
        ref = PRM_v2()
        ref_conns = deepcopy(ref.conns)
    else:
        ref_conns = deepcopy(in_ref_conns)

    conns = deepcopy(in_conns)

    v, v_ref = [], []
    for c in [*conns]:
        v.append([*conns[c].values()])
        v_ref.append([*ref_conns[c].values()])

    v = np.array(v).reshape(16)
    v_ref = np.array(v_ref).reshape(16)

    c_list = ["pyr", "bic", "pv", "cck"]

    conn_labels = []
    for c in c_list:
        for c2 in c_list:
            conn_labels.append(f"$w_{{{c.upper()} \\rightarrow {c2.upper()}}}$")

    invalid = np.where(v_ref == 0)

    v = np.delete(v, invalid)
    v_ref = np.delete(v_ref, invalid)
    v_ratio = v/v_ref

    conn_labels = np.delete(conn_labels, invalid)

    num_labels = len(conn_labels)

    angles = np.linspace(0, 2 * np.pi, num_labels, endpoint=False)

    # append the first value to the end of the array to create a closed loop
    v_ratio = np.append(v_ratio, v_ratio[0])
    v_ref = np.append(v_ref, v_ref[0])
    angles = np.append(angles, angles[0])

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    # Draw the outline of our data.
    ax.plot(angles, v_ratio, linewidth=1)
    ax.plot(angles, v_ref/v_ref, linewidth=1)
    # Fill it in.
    ax.fill(angles, v_ratio, alpha=0.25)
    ax.fill(angles, v_ref/v_ref, alpha=0.25)

    # line up first datapoint with vertical axis
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # clean up the labels so they don't overlap with the grid lines
    ax.set_thetagrids(np.degrees(angles[:-1]), conn_labels)
    for label, angle in zip(ax.get_xticklabels(), angles):
        if angle in (0, np.pi):
            label.set_horizontalalignment('center')
        elif 0 < angle < np.pi:
            label.set_horizontalalignment('left')
        else:
            label.set_horizontalalignment('right')

    ax.set_ylim(0, 3)

    ax.set_rlabel_position(180 / num_labels)
    # Add some custom styling.
    # Change the color of the tick labels.
    ax.tick_params(colors='#222222')
    # Make the y-axis (0-100) labels smaller.
    ax.tick_params(axis='y', labelsize=8)
    # Change the color of the circular gridlines.
    ax.grid(color='#AAAAAA')
    # Change the color of the outermost gridline (the spine).
    ax.spines['polar'].set_color('#222222')
    # Change the background color inside the circle itself.
    ax.set_facecolor('#FAFAFA')

def make_boxplot(conn_dict):
    c_list = ["pyr", "bic", "pv", "cck"]

    conn_arr = np.zeros((len(conn_dict), 9))
    conn_labels = []
    for c in c_list:
        for c2 in c_list:
            conn_labels.append(f"$w_{{{c.upper()} \\rightarrow {c2.upper()}}}$")

    conn = conn_dict[0]
    v = []
    for c in [*conn]:
        v.append([*conn[c].values()])
    v = np.array(v).reshape(16)
    invalid = np.where(v == 0)
    conn_labels = np.delete(conn_labels, invalid)

    for idx, conn in enumerate(conn_dict):
        v = []
        for c in [*conn]:
            v.append([*conn[c].values()])
        v = np.array(v).reshape(16)
        v = np.delete(v, invalid)
        conn_arr[idx] = v

    # stats = np.zeros((9, 3))

    medianprops = dict(
        linewidth=4.5,
        solid_capstyle="butt"
    )
    boxprops = dict(
        linewidth=3
    )
    whiskerprops = dict(
        linewidth=3
    )
    capprops = dict(
        linewidth=3
    )

    plt.figure(figsize=(18, 10), dpi=400)
    plt.boxplot(conn_arr, labels=conn_labels,
                medianprops=medianprops,
                boxprops=boxprops,
                whiskerprops=whiskerprops,
                capprops=capprops)
    plt.ylabel("Synaptic Weight")

    # for i in range(9):
    #     stats[i] = [np.mean(conn_arr[:, i]), np.median(conn_arr[:, i]), np.std(conn_arr[:, i])]
    #     print(conn_labels[i], f"\nMean: {np.round(stats[i, 0], 7)}\nMedian: {np.round(stats[i, 1], 7)}"
    #                           f"\nSD: {stats[i, 2]:.3e}")
    #     print("="*60)
        # plt.text(i+0.75, 0.2,
        #          f"Mean: {np.round(stats[i, 0], 2)}\nSD: {np.round(stats[i, 1], 2)}")


    plt.tight_layout()
    # plt.savefig("./figures/conn_10/boxplot_conn_10.png")

def ref_power():
    ref_prm = PRM_v2()
    n_trials = 1

    temp_theta, temp_gamma = np.zeros(n_trials), np.zeros(n_trials)

    for i in range(1):
        ref_prm.set_init_state(len(time))
        ref_prm = simulate(time, ref_prm)

        temp_theta[i] = find_pyr_power(ref_prm.R, fs, "theta")[1]
        temp_gamma[i] = find_pyr_power(ref_prm.R, fs, "gamma")[1]

    ref_tpp = np.mean(temp_theta)
    ref_gpp = np.mean(temp_gamma)
    return ref_tpp, ref_gpp

def find_psd(R, fs):
    segment = int(fs * 4)
    myhann = signal.get_window('hann', segment)

    myparams = dict(fs=fs, nperseg=segment, window=myhann,
                    noverlap=segment / 2, scaling='density', return_onesided=True)

    signal_pyr = R["pyr"][int(fs):]
    signal_pv = R["pv"][int(fs):]

    fxx_pyr, Pxx_pyr = signal.welch(signal_pyr, **myparams)
    fxx_pv, Pxx_pv = signal.welch(signal_pv, **myparams)

    return fxx_pyr, Pxx_pyr, fxx_pv, Pxx_pv

def find_pyr_power(R, fs, band="theta"):
    if band == "theta":
        f_lo, f_hi = 3, 12
    elif band == "gamma":
        f_lo, f_hi = 20, 100
    else:
        raise IOError("Invalid band: Band must be 'theta' or 'gamma'")

    fxx_pyr, Pxx_pyr, fxx_pv, Pxx_pv = find_psd(R, fs)

    fxx_pyr_filt = fxx_pyr[(fxx_pyr > f_lo) & (fxx_pyr < f_hi)]
    Pxx_pyr_filt = Pxx_pyr[(fxx_pyr > f_lo) & (fxx_pyr < f_hi)]
    fxx_pv_filt = fxx_pv[(fxx_pv > f_lo) & (fxx_pv < f_hi)]
    Pxx_pv_filt = Pxx_pv[(fxx_pv > f_lo) & (fxx_pv < f_hi)]

    f_max, P_max = np.nan, np.nan

    if band == 'theta':
        f_max, P_max = fxx_pyr_filt[np.argmax(Pxx_pyr_filt)], np.max(Pxx_pyr_filt)

    if band == 'gamma':
        f_max, P_max = fxx_pv_filt[np.argmax(Pxx_pv_filt)], Pxx_pyr_filt[np.argmax(Pxx_pv_filt)]


    # testing (remove later)
    # plt.figure()
    # plt.plot(fxx_pyr_filt, Pxx_pyr_filt)
    # plt.plot(fxx_pv_filt, Pxx_pv_filt)
    # plt.plot(f_max, P_max, 'x')
    # plt.semilogy()

    return f_max, P_max

def theta_freq_vs_gamma(sr, tf, tp, gp):
    out_data = np.zeros((len(tf)-5, 2))
    # print(tp)
    # take five points for each point of stim range
    for idx, val in enumerate(tf[2:-3]):
        # print(idx)
        tf_range = tf[idx-2:idx+3]
        tp_range = tp[idx-2:idx+3]
        gp_range = gp[idx-2:idx+3]
        sr_range = sr[idx-2: idx+3]

        # print(tp_range)

        # check: is point valid? check theta and gamma power --> if 3 of 5 valid
        valid = [False]*5
        for jdx, vals in enumerate(zip(tp_range, gp_range)):
            # print(vals)
            if (vals[0] >= 48) and (vals[1] >= 0.31):
                valid[jdx] = True

        if sum(valid) >= 3:

            # find slope of theta freq
            try:
                a1, a0 = np.polyfit(sr_range[valid], tf_range[valid], 1)
            except LinAlgError:
                a1, a0 = np.nan, np.nan

            gp_mean = np.mean(gp_range[valid])
            out_data[idx] = np.array([gp_mean, a1])

        else:
            out_data[idx] = np.array([np.nan, np.nan])

    return out_data

def spike_raster(R):
    peaks = {}
    prominence_dict = {
        'pyr': 10,
        'pv': 8,
        'bic': 10,
        'cck': 4
    }

    for ctype in ['pyr', 'bic', 'cck', 'pv']:
        peaks[ctype], _ = scipy.signal.find_peaks(R[ctype], prominence=prominence_dict[ctype])

    return peaks

def save_signal(fname, time, R):
    DATA = np.column_stack((time, *R.values()))
    header = "t\tR_PYR\tR_BiC\tR_CCK\tR_PV"

    np.savetxt(fname, DATA, header=header)

# =============================================================
# Parameters
T = 8.0  # total time (units in sec)
dt = 0.001  # plotting and Euler timestep (parameters adjusted accordingly)
fs = 1/dt
time = np.arange(0, T, dt)

# # FI curve
# beta = 10
# tau = 5
# h = 0
# r_o = 30
# ===============================================================
# R = baseline_sim(import_conns(0))
# peaks = spike_raster(R)
#
# # plt.plot(time, R['pyr'])
# plt.plot(time[peaks['pyr']], 10*np.ones_like([peaks['pyr']][0]), 'x')
# plt.plot(time[peaks['bic']], 10*np.ones_like([peaks['bic']][0]), 'x')
# plt.plot(time[peaks['cck']], 10*np.ones_like([peaks['cck']][0]), 'x')
# # plt.plot(time[peaks['pv']], 40*np.ones_like([peaks['pv']][0]), 'x')
#
#
# plt.show()