import json
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from prm_v2 import *
from prm_htest import hypothesis_test

def l2_ref(conns, ref_conns):
    v1, v2 = [], []
    for c in [*conns]:
        v1.append([*conns[c].values()])
        v2.append([*ref_conns[c].values()])

    v1 = np.array(v1).reshape(16)
    v2 = np.array(v2).reshape(16)

    return np.linalg.norm(v1-v2)

with open("search_results/search_results_conn_6.json", "r") as f:
    conn_data = json.load(f)

# with open("search_results/search_results_i_4.json", "r") as f:
#     i_data = json.load(f)

ref_conns = {
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

c_list = {"pyr", "bic", "pv", "cck"}
true_count, false_count = 0, 0

# for idx, conn in tqdm(enumerate(conn_data)):
#     if hypothesis_test(conn):
#         true_count += 1
#         # print(conn, i)
#     else:
#         false_count += 1
# harmonic = np.zeros(len(conn_data), dtype=bool)
# freq_arr = np.zeros((2, len(conn_data)))
#
# for idx, conn in tqdm(enumerate(conn_data)):
#     [tf, _], [gf, _] = test_prm(conn)
#     freq_arr[:, idx] = tf, gf

# print(f"True: {sum(harmonic)}; False: {len(harmonic)-sum(harmonic)}")
# plt.scatter(freq_arr[0], freq_arr[1])

# test_list = [4, 14, 27, 33]
# test_list = [6, 15, 38, 79]
# test_list = [1, 2, 12, 58]
# valid_ish = [34, 46, 62, 66, 115
n = 26
# print(hypothesis_test(conn_data[n], plot=True))
stim = {
    "pyr": 0.0,
    "bic": 0.0,
    "pv": 0.0,
    "cck": 0.0
}


# print(test_prm())
# for i in range(10):
#     print(run_prm(conn_data[n], stim=stim))

# print(run_prm(plot=True))
#
# l2_list = np.zeros(len(conn_data))
#
# for idx, conn in enumerate(conn_data):
#     l2_list[idx] = l2_ref(conn, ref_conns)
#
# plt.plot(l2_list, 'o')

ax1 = create_radar()

plot_radar(ref_conns, ax1, label = "ref")

for j in [26, 71]:
    plot_radar(conn_data[j], ax1, label = f"conn_6_{j}")

plt.legend()
plt.tight_layout()
plt.show()
