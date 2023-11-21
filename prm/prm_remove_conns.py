import os

import matplotlib.pyplot as plt

from prm_v2 import *

with open("search_results/search_results_conn_10.json", "r") as f:
    conn_data = np.array(json.load(f))

n = 0
conns = conn_data[n]



sdir = f"./figures/conn_10/removed_conns/conn_10_{n}"
try:
    os.makedirs(sdir)
except FileExistsError:
    pass

conns_dict = {
    "pyr": ["pyr", "bic", "pv"],
    "bic": ["pyr"],
    "cck": ["pv", "cck"],
    "pv": ["pyr", "pv", "cck"],
}
for c1 in [*conns_dict]:
    for c2 in [*conns_dict[c1]]:
        P = PRM_v2(conns_dict=conns)
        P.conns[c1][c2] = 0

        P.set_init_state(len(time))
        P = simulate(time, P)

        plt.figure()
        for ctype in [*conns_dict]:
            plt.plot(time[7*len(time)//8:], P.R[ctype][7*len(time)//8:], label=ctype)
        plt.title(f"removed {c1} {c2} connection")
        plt.legend()
        plt.savefig(f"{sdir}/{c1}_{c2}.png")
        plt.close()


# plt.show()