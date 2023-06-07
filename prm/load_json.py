import json

import matplotlib.pyplot as plt
import numpy as np

from prm_v2 import *


from kneed import KneeLocator
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

def make_boxplot(conn_dict):
    c_list = ["pyr", "bic", "pv", "cck"]

    conn_arr = np.zeros((len(conn_dict), 9))
    conn_labels = []
    for c in c_list:
        for c2 in c_list:
            conn_labels.append(f"$w_{{{c.upper()} \\rightarrow {c2.upper()}}}$")

    conn = conn_dict[0]
    v = []
    for c in [*conn]:
        v.append([*conn[c].values()])
    v = np.array(v).reshape(16)
    invalid = np.where(v == 0)
    conn_labels = np.delete(conn_labels, invalid)

    for idx, conn in enumerate(conn_dict):
        v = []
        for c in [*conn]:
            v.append([*conn[c].values()])
        v = np.array(v).reshape(16)
        v = np.delete(v, invalid)
        conn_arr[idx] = v

    stats = np.zeros((9, 2))

    plt.figure(figsize=(18, 10), dpi=250)
    plt.boxplot(conn_arr, labels=conn_labels)

    for i in range(9):
        stats[i] = [np.mean(conn_arr[:, i]), np.std(conn_arr[:, i])]
        plt.text(i+0.75, 0.2,
                 f"Mean: {np.round(stats[i, 0], 2)}\nSD: {np.round(stats[i, 1], 2)}")

    plt.savefig("./figures/conn_10/boxplot_conn_10.png")
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

    return np.linalg.norm(v1 - v2)


# with open("search_results/search_results_conn_6.json", "r") as f:
#     conn_data = json.load(f)
#
# with open("search_results/search_results_conn_7.json", "r") as f:
#     conn_data_2 = json.load(f)

with open("search_results/search_results_conn_8.json", "r") as f:
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

# print(run_prm(conn_data_2[41], plot=True))

# possible new sets: 7-1, 7-14, 7-25, 7-27*, 7-31, 7-32, 7-37, 7-41

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

pst = 7 * len(time) // 8

prm_0 = PRM_v2()
# prm_1 = PRM_v2(conns_dict=conn_data[26])
# prm_2 = PRM_v2(conns_dict=conn_data[71])
# prm_3 = PRM_v2(conns_dict=conn_data_2[92])

# =============================================================
# cck_stim_vec = np.linspace(-2, 2, 61)
# cck_stim_vec = cck_stim_vec[(cck_stim_vec > -0.14) & (cck_stim_vec < 0.7)]
#
# count = 0
#
# for cck_stim in cck_stim_vec:
#     stim["cck"] = cck_stim
#     prm_0.set_init_state(len(time))
#     prm_0 = simulate(time, prm_0, stim=stim)
#
#     segment = int(fs * 4)
#     myhann = signal.get_window('hann', segment)
#
#     myparams = dict(fs=fs, nperseg=segment, window=myhann,
#                     noverlap=segment / 2, scaling='density', return_onesided=True)
#
#     pyr_filt_lo = filter_lfp(prm_0.R["pyr"], fs, [3, 12])
#     pyr_filt_hi = filter_lfp(prm_0.R["pyr"], fs, [20, 100])
#
#     fft_lo = signal.welch(pyr_filt_lo, **myparams)
#     fft_hi = signal.welch(pyr_filt_hi, **myparams)
#
#     fig, ax = plt.subplots(2)
#     ax2 = ax[1].twinx()
#
#     ax[0].plot(time[pst:], prm_0.R["pyr"][pst:],
#                color="C0")
#     ax[0].set_xlabel("Time[s]")
#     ax[0].set_ylabel("Activity")
#
#     ax[1].plot(fft_lo[0], fft_lo[1])
#     ax2.plot(fft_hi[0], fft_hi[1], color="C1")
#     # ax[1].semilogy()
#     ax[1].set_xlim(-0.5, right=100)
#     # ax[1].set_ylim(bottom=1e-7)
#     ax[1].set_xlabel("Frequency [Hz]")
#     ax[1].set_ylabel("Theta PSD")
#     ax2.set_ylabel("Gamma PSD")
#     ax[1].yaxis.label.set_color("C0")
#     ax2.yaxis.label.set_color("C1")
#     ax[1].tick_params(axis='y', colors="C0")
#     ax2.tick_params(axis='y', colors="C1")
#
#     plt.suptitle(f"CCK STIM = {cck_stim.round(2)}")
#     plt.savefig(f"./figures/cck_stim_2/cck_stim_{count}.png")
#     count += 1
# plt.savefig("figures/set_0/6_26_activity_plot_nn.png")
# run_prm(prm_1.conns, plot=True)
# plt.savefig("figures/conn_6_26/6_26_activity_plot_nn.png")
#
# # run_prm(prm_2.conns, plot=True)
# run_prm(prm_3.conns, plot=True)
# plt.savefig("figures/conn_7_92/6_26_activity_plot_nn.png")
# =============================================================


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
# =============================================================


# ax[1, :].set_xlabel("Time [s]")
# ax[:, 0].set_ylabel("PYR cell activity")
# print(test_prm())
# for i in range(10):
#     print(run_prm(conn_data[n], stim=stim))

# print(run_prm(plot=True))
#
with open("search_results/search_results_conn_10.json", "r") as f:
    conn_data = np.array(json.load(f))

# make_boxplot(conn_data)

# theta_base_power, gamma_base_power = [], []
#
for n in [0, 2, 3, 7, 8, 22, 38, 50, 103, 135]:
    run_prm(conn_data[n], plot=True)
    plt.ylim((0, 42))
    plt.savefig(f"./figures/conn_10/pyr_activity/pyr_activity_10_{n}.png")
#     theta_base_power.append(theta[1])
#     gamma_base_power.append(gamma[1])
#
# print(f"Mean theta power = {np.mean(theta_base_power)}")
# print(f"Mean gamma power = {np.mean(gamma_base_power)}")
# l2_list = np.zeros(len(conn_data))
#
# for idx, conn in enumerate(conn_data):
#     l2_list[idx] = l2_ref(conn, ref_conns)
# # # # #
# clusters = k_means(conn_data, 13)
# print(np.arange(len(conn_data))[clusters.labels_ == 2])

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
#     plot_radar(ref_conns, ax1, label="ref", mode="absolute", color='black')
#     for i in range(len(conn_cluster_idx)):
#         plot_radar(conn_cluster_idx[i], ax1, mode="absolute", color=f"C{idx}")

# for i in range(clusters.n_clusters):
#     with open("search_results/clusters_10.txt", "a") as f:
#         f.write(str(np.arange(len(l2_list))[clusters.labels_ == i]))
#         f.write("\n")
#     plt.plot(np.arange(len(l2_list))[clusters.labels_ == i], 0*l2_list[clusters.labels_ == i] + i,
#              color=f"C{i}", marker='.', ls="")

# 149**, 139**, 18, 3*, 0, 17, 37*, 64*!, 12, 2*, 8*, 5*
# 18, 0, 17, 12, 147, 137, 49??, 86,
# print(run_prm(conn_data[149], plot=True))

# plt.plot(l2_list, '.')
#
# plt.hist(l2_list, bins=20)

# ax1 = create_radar()
#
# plot_radar(ref_conns, ax1, label = "ref", relative_to_ref=True, color='black')
# for i in range(len(conn_data)):
#     plot_radar(conn_data[i], ax1, relative_to_ref=True, color=f"C{clusters.labels_[i]}")

# for idx in range(clusters.n_clusters):
#     val = np.arange(len(l2_list))[clusters.labels_ == idx][0]

    # with open("new_sets.txt", "a") as f:
    #     f.write(f"Conn 10-{val}\n")
    #     f.write(str(conn_data[val]))
    #     f.write("\n"+"="*60 + "\n")

    # new_prm = PRM_v2(conn_data[val])
    # new_prm.set_init_state(len(time))
    # new_prm = simulate(time, new_prm)
    #
    # max_pv = np.max(new_prm.R["pv"][pst:])
    # max_bic = np.max(new_prm.R["bic"][pst:])
    # max_cck = np.max(new_prm.R["cck"][pst:])
    # print(f"Conn 10-{val}: ")
    # print(f"Max PV = {np.round(max_pv, 3)}")
    # print(f"Max BiC = {np.round(max_bic, 3)}")
    # print(f"Max CCK = {np.round(max_cck, 3)}")
    # print(f"PV-BiC ratio = {np.round(max_pv/max_bic, 3)}")
    # print(""*60)
    # print(run_prm(conn_data[val], plot=True))
    # plt.title(f"Conn 10-{val}")
    # plt.savefig(f"./figures/conn_10/activity_conn_10_{val}.png", dpi=300)

    # ax1 = create_radar()
    # plot_radar(ref_conns, ax1, label="ref", mode='relative', color='black')
    # plot_radar(conn_data[val], ax1, mode='relative', color='red')
    # ax1.set_title(f"conn 10-{val}")

# run_prm(conn_data[254], plot=True)

# with open("search_results/search_results_conn_7.json", "r") as f:
#     conn_data = json.load(f)
# for i in range(len(conn_data)):
#     plot_radar(conn_data[i], ax1, mode="absolute", color='black')

# with open("search_results/search_results_conn_8.json", "r") as f:
#     conn_data = json.load(f)

# for i in range(len(conn_data)):
    # plot_radar(conn_data[i], ax1, relative_to_ref=False, color='black')

# plot_radar(conn_data[92], ax1, label=f"conn_7_92")

# plt.legend()
# ax1.set_ylim(0, 3)
# plt.tight_layout()
plt.show()
