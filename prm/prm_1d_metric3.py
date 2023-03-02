from prm_v2 import *

import numpy as np
import matplotlib.pyplot as plt

# ===============================================================
# Parameters
T = 2.0  # total time (units in sec)
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
P_ref = PRM_v2()
P_ref.set_init_state(len(time))
P_ref = simulate(time, P_ref, dt, tau, r_o)

ref = np.log10(
    calc_spectral(P_ref.R, fs, time, P_ref.labels, 'theta', 'power')["pyr"] /
    calc_spectral(P_ref.R, fs, time, P_ref.labels, 'gamma', 'power')["pyr"]
)

# cell type to look at
ctype = "cck"

num_pts = 40  # number of points to plot

max_inputs = len(c_list) + 1

plt.figure(figsize=[9, 3])

for i in range(max_inputs):
    P = PRM_v2()
    x_vec = np.linspace(0.5, 1.5, num_pts)

    log_power_ratio = np.zeros((max_inputs, num_pts))
    y_pct = np.zeros_like(log_power_ratio)

    if i == len(c_list):
        p_vec = x_vec * P.I[ctype]
        y_label = f"$I_{{{ctype.upper()}}}$"

        for idx, val in enumerate(p_vec):
            P.I[ctype] = val
            P.set_init_state(len(time))
            P = simulate(time, P, dt, tau, r_o)

            log_power_ratio[i, idx] = np.log10(
                calc_spectral(P.R, fs, time, P.labels, 'theta', 'power')["pyr"] /
                calc_spectral(P.R, fs, time, P.labels, 'gamma', 'power')["pyr"]
            )

        y_pct[i] = log_power_ratio[i] / ref * 100
        plt.plot(x_vec * 100, y_pct[i], label=y_label,
                 ls='--', marker='.')

    else:
        if P.conns[c_list[i]][ctype] == 0:
            pass
        else:
            p_vec = x_vec * P.conns[c_list[i]][ctype]
            y_label = f"$w_{{{c_list[i].upper()} \\rightarrow {ctype.upper()}}}$"

            for idx, val in enumerate(p_vec):
                P.conns[c_list[i]][ctype] = val
                P.set_init_state(len(time))
                P = simulate(time, P, dt, tau, r_o)

                log_power_ratio[i, idx] = np.log10(
                    calc_spectral(P.R, fs, time, P.labels, 'theta', 'power')["pyr"] /
                    calc_spectral(P.R, fs, time, P.labels, 'gamma', 'power')["pyr"]
                )

            y_pct[i] = log_power_ratio[i] / ref * 100
            plt.plot(x_vec * 100, y_pct[i], label=y_label,
                     ls='--', marker='.')

plt.ylabel("%$\Delta$ Log Power Ratio")
plt.xlabel("%$\Delta$ Parameter")
plt.title(f"inputs to {ctype} cell".upper())
plt.grid()
plt.legend()
plt.tight_layout()

plt.savefig(f"./figures/1d_metric3/inputs_to_{ctype}_cell.png")
# plt.show()