import json
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

from prm_htest import hypothesis_test

def l2_ref(conns, ref_conns):
    v1, v2 = [], []
    for c in [*conns]:
        v1.append([*conns[c].values()])
        v2.append([*ref_conns[c].values()])

    v1 = np.array(v1).reshape(16)
    v2 = np.array(v2).reshape(16)

    return np.linalg.norm(v1-v2)

with open("search_results/search_results_conn_4.json", "r") as f:
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
#     # for c in c_list:
#         # conn[c]["pyr"] = ref_conns[c]["pyr"]
#     if hypothesis_test(conn):
#         true_count += 1
#         # print(conn, i)
#     else:
#         false_count += 1
#
# print(f"True: {true_count}; False: {false_count}")

# test_list = [4, 14, 27, 33]
# test_list = [6, 15, 38, 79]
# test_list = [1, 2, 12, 58]
#
print(hypothesis_test(conn_data[89], plot=True))

# l2_list = np.zeros(len(conn_data))
#
# for idx, conn in enumerate(conn_data):
#     l2_list[idx] = l2_ref(conn, ref_conns)
#
# plt.plot(l2_list, 'o')
plt.show()
