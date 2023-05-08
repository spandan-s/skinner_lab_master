from copy import deepcopy

import numpy as np
from matplotlib import pyplot as plt
from scipy import signal
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
                "pyr": 0.01,
                "bic": -1.4,
                "pv": 0.5,
                "cck": 0.8,
            }
        else:
            self.I = deepcopy(param_dict)

    def set_init_state(self, n_samples):
        self.R = {
            "pyr": np.zeros(n_samples),
            "bic": np.zeros(n_samples),
            "pv": np.zeros(n_samples),
            "cck": np.zeros(n_samples),
        }


def f(u, beta=10, h=0):
    return 1 / (1 + np.exp(-beta * (u - h)))


def simulate(time, P, dt=0.001, tau=5, stim=None):
    if stim == None:
        stim = {
            "pyr": 0,
            "bic": 0,
            "pv": 0,
            "cck": 0
        }
    # euler_integrate
    for t in range(len(time) - 1):
        c_list = ["pyr", "bic", "pv", "cck"]
        for c1 in c_list:
            P.R[c1][t + 1] = P.R[c1][t] + dt * P.alpha[c1] * \
                             (-P.R[c1][t] + P.r_o[c1] * f(
                                 sum((P.conns[c2][c1] * P.R[c2][t - tau]) for c2 in c_list) + P.I[c1] + stim[c1])) + \
                             np.sqrt(2 * P.alpha[c1] * P.D[c1] * dt) * np.random.normal(0, 1)
    return P


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


def plot_trace(time, R, labels):
    plot_start_time = 3 * time.size // 4
    plt.figure(figsize=[13, 8])

    c_list = ["pyr", "bic", "cck", "pv"]
    for c in c_list:
        plt.plot(time[plot_start_time:], R[c][plot_start_time:], label=labels[c])

    plt.xlabel('Time (ms)')
    plt.ylabel('Activity')
    plt.legend()


def valid_oscillation(R, fs, time, ref=None):
    if ref == None:
        ref_prm = PRM_v2()
        ref_prm.set_init_state(len(time))
        ref_prm = simulate(time, ref_prm)
        ref_tpp = calc_spectral(ref_prm.R, fs, time, 'theta', 'power')["pyr"]
        ref_gpp = calc_spectral(ref_prm.R, fs, time, 'gamma', 'power')["pyr"]
        ref = [ref_tpp, ref_gpp]

    tpp = calc_spectral(R, fs, time, 'theta', 'power')["pyr"]
    gpp = calc_spectral(R, fs, time, 'gamma', 'power')["pyr"]

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
    fs = 1 / dt
    new_prm = PRM_v2(conns, I)

    time = np.arange(0, T, dt)

    new_prm.set_init_state(len(time))
    new_prm = simulate(time, new_prm, stim=stim)

    if plot:
        plot_trace(time, new_prm.R, new_prm.labels)

    tf = calc_spectral(new_prm.R, fs, time, 'theta', 'peak_freq', plot_Filter=False)["pyr"]
    gf = calc_spectral(new_prm.R, fs, time, 'gamma', 'peak_freq', plot_Filter=False)["pyr"]
    tpp = calc_spectral(new_prm.R, fs, time, 'theta', 'power', plot_Filter=False)["pyr"]
    gpp = calc_spectral(new_prm.R, fs, time, 'gamma', 'power', plot_Filter=False)["pyr"]

    return [tf, tpp], [gf, gpp]

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
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
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

    # ax.set_ylim(0, 3)

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

    return ax

def plot_radar(in_conns, ax, label=None):
    conns = deepcopy(in_conns)
    v = []

    for c in [*conns]:
        v.append([*conns[c].values()])
    v = np.array(v).reshape(16)
    c_list = ["pyr", "bic", "pv", "cck"]

    conn_labels = []
    for c in c_list:
        for c2 in c_list:
            conn_labels.append(f"$w_{{{c.upper()} \\rightarrow {c2.upper()}}}$")

    invalid = np.where(v == 0)

    v = np.delete(v, invalid)
    conn_labels = np.delete(conn_labels, invalid)

    num_labels = len(conn_labels)

    angles = np.linspace(0, 2 * np.pi, num_labels, endpoint=False)
    v = np.append(v, v[0])
    v = 10*abs(v)
    angles = np.append(angles, angles[0])
    # Draw the outline of our data.
    ax.plot(angles, v, linewidth=1, label=label)
    # Fill it in.
    ax.fill(angles, v, alpha=0.25)


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

def ref_power():
    ref_prm = PRM_v2()

    temp_theta, temp_gamma = np.zeros(5), np.zeros(5)

    for i in range(5):
        ref_prm.set_init_state(len(time))
        ref_prm = simulate(time, ref_prm)

        temp_theta[i] = calc_spectral(ref_prm.R, fs, time, 'theta', 'power')["pyr"]
        temp_gamma[i] = calc_spectral(ref_prm.R, fs, time, 'gamma', 'power')["pyr"]

    ref_tpp = np.mean(temp_theta)
    ref_gpp = np.mean(temp_gamma)
    return ref_tpp, ref_gpp
# =============================================================
# Parameters
T = 8.0  # total time (units in sec)
dt = 0.001  # plotting and Euler timestep (parameters adjusted accordingly)
fs = 1/dt
time = np.arange(0, T, dt)

# FI curve
beta = 10
tau = 5
h = 0
# r_o = 30
# ===============================================================
# new_prm = PRM_v2()
# #
# for idx in new_prm.D:
#     new_prm.D[idx] = 0.0
# new_prm.set_init_state(len(time))
# new_prm = simulate(time, new_prm, dt, tau)
# print(new_prm.D)
# #
# plot_trace(time, new_prm.R, new_prm.labels)
# dps_tpp = calc_spectral(new_prm.R, fs, time, new_prm.labels, 'theta', 'power', plot_Fig=True)["pyr"]
# dps_gpp = calc_spectral(new_prm.R, fs, time, new_prm.labels, 'gamma', 'power', plot_Fig=True)["pyr"]
#
# print(dps_tpp, dps_gpp)

# ax = create_radar()
# plot_radar(new_prm.conns)

# plt.show()
