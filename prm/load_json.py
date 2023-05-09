import json
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from prm_v2 import *
from prm_htest import hypothesis_test

from kneed import KneeLocator
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

def k_means(conn_data, k):

    features = np.zeros((len(conn_data), 16))
    for idx, conns in enumerate(conn_data):
        v1 = []
        for c in [*conns]:
            v1.append([*conns[c].values()])

        features[idx] = np.array(v1).reshape(16)

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    kmeans = KMeans(
        init="random",
        n_clusters=k,
        n_init=10,
        max_iter=300,
        random_state=42
    )

    kmeans.fit(scaled_features)

    return kmeans
def l2_ref(conns, ref_conns):
    v1, v2 = [], []
    for c in [*conns]:
        v1.append([*conns[c].values()])
        v2.append([*ref_conns[c].values()])

    v1 = np.array(v1).reshape(16)
    v2 = np.array(v2).reshape(16)

    return np.linalg.norm(v1-v2)

# with open("search_results/search_results_conn_6.json", "r") as f:
#     conn_data = json.load(f)
#
# with open("search_results/search_results_conn_7.json", "r") as f:
#     conn_data_2 = json.load(f)

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
#
# pst = 7 * len(time) // 8
#
# prm_0 = PRM_v2()
# prm_1 = PRM_v2(conns_dict=conn_data[26])
# prm_2 = PRM_v2(conns_dict=conn_data[71])
# prm_3 = PRM_v2(conns_dict=conn_data_2[92])
#
#
# run_prm(prm_0.conns, plot=True)
# plt.savefig("figures/set_0/6_26_activity_plot_nn.png")
# run_prm(prm_1.conns, plot=True)
# plt.savefig("figures/conn_6_26/6_26_activity_plot_nn.png")
#
# # run_prm(prm_2.conns, plot=True)
# run_prm(prm_3.conns, plot=True)
# plt.savefig("figures/conn_7_92/6_26_activity_plot_nn.png")

# prm_list = [prm_0, prm_1, prm_2, prm_3]
#
# fig, ax = plt.subplots(2, 2, True, True, figsize=[12, 4.8])
#
# line_style = ['solid', 'dotted', 'dashed', 'dashdot']
# for i in range(4):
#     prm_list[i].set_init_state(len(time))
#     prm_list[i] = simulate(time, prm_list[i])
#     ax[i//2, i%2].plot(time[pst:], prm_list[i].R["pyr"][pst:],
#              color="C0")
#     ax[i//2, i%2].set_title(f"Set {i}")


# ax[1, :].set_xlabel("Time [s]")
# ax[:, 0].set_ylabel("PYR cell activity")
# print(test_prm())
# for i in range(10):
#     print(run_prm(conn_data[n], stim=stim))

# print(run_prm(plot=True))
#
with open("search_results/search_results_conn_8.json", "r") as f:
    conn_data = json.load(f)
# l2_list = np.zeros(len(conn_data))
#
# for idx, conn in enumerate(conn_data):
#     l2_list[idx] = l2_ref(conn, ref_conns)
# #
# clusters = k_means(conn_data, 12)

# sse = []
# for k in range(1, 25):
#     clusters = k_means(conn_data, k)
#     sse.append(clusters.inertia_)
#
# plt.style.use("fivethirtyeight")
# plt.plot(range(1, 25), sse)
# plt.xticks(range(1, 25))
# plt.xlabel("Number of Clusters")
# plt.ylabel("SSE")
#
# kl = KneeLocator(
#     range(1, 25), sse, curve="convex", direction="decreasing"
# )
#
# print(kl.elbow)
# for idx in range(clusters.n_clusters):
#     conn_cluster_idx = conn_data[clusters.labels_ == idx]
#     ax1 = create_radar()
#     plot_radar(ref_conns, ax1, label="ref", relative_to_ref=False, color='black')
#     for i in range(len(conn_cluster_idx)):
#         plot_radar(conn_cluster_idx[i], ax1, relative_to_ref=True, color=f"C{idx}")

# for i in range(clusters.n_clusters):
#     with open("search_results/clusters.txt", "a") as f:
#         f.write(str(np.arange(len(l2_list))[clusters.labels_ == i]))
#         f.write("\n")
#     plt.plot(np.arange(len(l2_list))[clusters.labels_ == i], 0*l2_list[clusters.labels_ == i] + i,
#              color=f"C{i}", marker='.', ls="")

# 149**, 139**, 18, 3*, 0, 17, 37*, 64*!, 12, 2*, 8*, 5*
# 18, 0, 17, 12, 147, 137, 49??, 86,
# print(run_prm(conn_data[149], plot=True))

# plt.plot(l2_list, '.')

# ax1 = create_radar()
#
# plot_radar(ref_conns, ax1, label = "ref", relative_to_ref=True, color='black')
# for i in range(len(conn_data)):
#     plot_radar(conn_data[i], ax1, relative_to_ref=True, color=f"C{clusters.labels_[i]}")

for idx, val in enumerate([18, 0, 17, 12, 147, 137, 49, 86]):
    ax1 = create_radar()
    plot_radar(ref_conns, ax1, label="ref", relative_to_ref=True, color='black')
    plot_radar(conn_data[val], ax1, relative_to_ref=True, color='red')
    ax1.set_title(f"conn 8-{val}")

# with open("search_results/search_results_conn_7.json", "r") as f:
#     conn_data = json.load(f)
# for i in range(len(conn_data)):
#     plot_radar(conn_data[i], ax1, relative_to_ref=True)

# with open("search_results/search_results_conn_8.json", "r") as f:
#     conn_data = json.load(f)

# for i in range(len(conn_data)):
    # plot_radar(conn_data[i], ax1, relative_to_ref=False, color='black')

# plot_radar(conn_data[92], ax1, label=f"conn_7_92")

# plt.legend()
# ax1.set_ylim(0, 3.5)
plt.tight_layout()
plt.show()
