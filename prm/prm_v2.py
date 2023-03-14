import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import periodogram, find_peaks
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
            self.conns = param_dict

    def _set_inputs(self, param_dict='default'):
        self.D = {
            "pyr": 0.001,
            "bic": 0.001,
            "pv": 0.001,
            "cck": 0.001
        }
        if param_dict == "default":
            self.I = {
                "pyr": 0.001,
                "bic": -1.4,
                "pv": 0.5,
                "cck": 0.8,
            }
        else:
            self.I = param_dict

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


def calc_spectral(R, fs, time, labels,
                  band='theta', mode='peak_freq', plot_Fig=False, plot_Filter=False):
    # choose which band of oscillations you want to filter
    if band == 'theta':
        cutoff = np.array([3, 15])
    elif band == 'gamma':
        # should this be changed - maybe?
        cutoff = np.array([15, 100])

    c_list = ["pyr", "bic", "pv", "cck"]

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

    pgram = {
        "pyr": periodogram(R_filt["pyr"], fs),
        "bic": periodogram(R_filt["bic"], fs),
        "pv": periodogram(R_filt["pv"], fs),
        "cck": periodogram(R_filt["cck"], fs),
    }
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
        "pyr": find_peaks(pgram["pyr"][1])[0],
        "bic": find_peaks(pgram["bic"][1])[0],
        "pv": find_peaks(pgram["pv"][1])[0],
        "cck": find_peaks(pgram["cck"][1])[0]
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


def plot_trace(time, R, labels):
    plot_start_time = 3 * time.size // 4
    plt.figure(figsize=[13, 8])

    c_list = ["pyr", "bic", "pv", "cck"]
    for c in c_list:
        plt.plot(time[plot_start_time:], R[c][plot_start_time:], label=labels[c])

    plt.xlabel('Time (ms)')
    plt.ylabel('Activity')
    plt.legend()

# # =============================================================
# # Parameters
# T = 8.0  # total time (units in sec)
# dt = 0.001  # plotting and Euler timestep (parameters adjusted accordingly)
# fs = 1/dt
#
# # FI curve
# beta = 10
# tau = 5
# h = 0
# r_o = 30
# # ===============================================================
# new_prm = PRM_v2()
#
# time = np.arange(0, T, dt)
#
# new_prm.set_init_state(len(time))
# new_prm = simulate(time, new_prm, dt, tau)
#
# plot_trace(time, new_prm.R, new_prm.labels)
# dps_tpp = calc_spectral(new_prm.R, fs, time, new_prm.labels, 'theta', 'power', plot_Fig=True)["pyr"]
# dps_gpp = calc_spectral(new_prm.R, fs, time, new_prm.labels, 'gamma', 'power', plot_Fig=True)["pyr"]
#
# plt.show()
