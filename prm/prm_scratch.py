import json
import os

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from prm_v2 import *

with open("search_results/search_results_conn_10.json", "r") as f:
    conn_data = json.load(f)

# run_prm(conn_data[2], stim={"pyr": 0, "bic": 0, 'cck': -0.5, "pv": 0}, plot=True)
#
# stim_range = (-2, 2)
# num_pts = 61
#
# x_vec = np.linspace(stim_range[0], stim_range[1], num_pts)
# x_vec = x_vec[(x_vec > -0.5) & (x_vec < 0.5)]
#
# stim = {
#         "pyr": 0,
#         "bic": 0,
#         "pv": 0,
#         "cck": 0
#     }
# for idx, val in tqdm(enumerate(x_vec)):
#     stim["bic"] = val
#
#     theta, gamma = run_prm(conns=conn_data[0], stim=stim, plot=True)
#     plt.title(f"BiC STIM = {np.round(stim['bic'], 3)}")
#     plt.savefig(f"/home/spandans/skinner_lab_master/prm/figures/conn_8_0/bic_stim/bic_stim_{idx}.png")
# ============================================================================
n_stim = 20
stim_arr = np.linspace(-1, 0, n_stim)
stim_duration = 1.25

time_end = stim_duration * n_stim
t_vec = np.arange(0, time_end, dt)
stim_vec = np.zeros_like(t_vec)

for idx, val in enumerate(stim_arr):
    stim_vec[int(stim_duration/dt*idx):int(stim_duration/dt*(idx+1))] = val

stim_vec[0:int(stim_duration/dt)] = 0

# plt.plot(t_vec, stim_vec)
# ============================================================================
n = 0

stim = {
    "pyr": np.zeros_like(t_vec),
    "bic": np.zeros_like(t_vec),
    "cck": np.zeros_like(t_vec),
    "pv": np.zeros_like(t_vec),
}

stim["cck"] = stim_vec

new_prm = PRM_v2(conns_dict=conn_data[n])
new_prm.set_init_state(len(t_vec))

new_prm = simulate(t_vec, new_prm, stim=stim)

fig, ax = plt.subplots(2, sharex=True, figsize=[21, 9], dpi=400)
ax[0].plot(t_vec, new_prm.R["pyr"])
ax[0].grid(which='both', axis='x')
ax[0].set_ylabel("PYR Activity")

ax[1].plot(t_vec, stim_vec)
ax[1].grid(which='both', axis='both')
ax[1].set_ylabel("Stim to CCK")

ax[1].set_xlabel("Time [s]")
fig.suptitle(f"Conn 10-{n}")

plt.savefig("./figures/conn_10/tempfig.png")
# plt.savefig(f"./figures/conn_10/bic_varstim/varstim_conn_10_{n}.png")
# ============================================================================
# print(run_prm(conn_data[0], plot=True))

plt.show()

