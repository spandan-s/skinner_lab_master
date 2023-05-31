import numpy as np

from prm_v2 import *

def f_plot(ctype, num_pts=61):
    P = PRM_v2()
    x_vec = np.linspace(-2, 2, num_pts)
    stim = {
        "pyr": 0.0,
        "bic": 0.0,
        "pv": 0.0,
        "cck": 0.0
    }

    Y = np.zeros_like(x_vec)

    for idx, val in enumerate(x_vec):
        stim[ctype] = val
        P.set_init_state(len(time))
        for t in range(len(time) - 1):
                P.R[ctype][t + 1] = P.R[ctype][t] + dt * P.alpha[ctype] * \
                                 (-P.R[ctype][t] + P.r_o[ctype] * f(
                                     P.I[ctype] + stim[ctype])) + \
                                 np.sqrt(2 * P.alpha[ctype] * P.D[ctype] * dt) * np.random.normal(0, 1)

        Y[idx] = np.max(P.R[ctype])

    ax.plot(x_vec, Y, label=ctype)

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
num_pts = 41
P = PRM_v2()
P.set_init_state(len(time))

c_list = ["pyr", "bic", "cck", "pv"]
stim = {
        "pyr": 0.0,
        "bic": 0.0,
        "pv": 0.0,
        "cck": 0.0
    }

for ctype in c_list:
    for t in range(len(time) - 1):
        P.R[ctype][t + 1] = P.R[ctype][t] + dt * P.alpha[ctype] * \
                                         (-P.R[ctype][t] + P.r_o[ctype] * f(
                                             P.I[ctype] + stim[ctype])) + \
                                         np.sqrt(2 * P.alpha[ctype] * P.D[ctype] * dt) * np.random.normal(0, 1)

fig, ax = plt.subplots()
ax.set_xlabel("Stimulation")
ax.set_ylabel("Activity")


for ctype in c_list:
    f_plot(ctype)

plt.legend()
plt.grid()
# plt.savefig("./figures/f_plot_ref_set.png")

plt.show()