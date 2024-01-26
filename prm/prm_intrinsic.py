import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from prm_v2 import *

new_prm = PRM_v2()

for c1 in new_prm.conns:
    for c2 in new_prm.conns[c1]:
        new_prm.conns[c1][c2] = 0.0


stim_arr = np.arange(-2, 2, 0.025)
time = np.arange(0, 2, dt)
new_prm.set_init_state(len(time))

stim = {
    'pyr': 0 + np.zeros_like(time),
    'bic': 0 + np.zeros_like(time),
    'cck': 0 + np.zeros_like(time),
    'pv': 0 + np.zeros_like(time)
}

activity = {
    'pyr': np.zeros_like(stim_arr),
    'bic': np.zeros_like(stim_arr),
    'cck': np.zeros_like(stim_arr),
    'pv': np.zeros_like(stim_arr),
}

for ctype in ['pyr', 'bic', 'cck', 'pv']:
    for idx, stim_val in tqdm(enumerate(stim_arr)):
        for c1 in stim:
            stim[c1] = 0 + np.zeros_like(time)
        stim[ctype] = stim_val + np.zeros_like(time)

        new_prm.set_init_state(len(time))
        new_prm = simulate(time, new_prm, stim=stim)
        activity[ctype][idx] = np.max(new_prm.R[ctype])

    plt.plot(stim_arr, activity[ctype], label = new_prm.labels[ctype])

plt.legend()
plt.xlabel('STIM [au]'
           '')
plt.ylabel('Activity [Hz]')
plt.grid()
plt.tight_layout()

plt.savefig('figures/new_figs/intrinsic.pdf', dpi=400)

plt.show()