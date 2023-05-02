import json
from prm_v2 import *

with open("search_results/search_results_conn_6.json", "r") as f:
    conn_data_1 = json.load(f)

with open("search_results/search_results_conn_7.json", "r") as f:
    conn_data_2 = json.load(f)

conn_data = conn_data_1 + conn_data_2[1:]

num_conns = len(conn_data)
# print(num_conns)


def create_conn_vector(conns):
    v = []
    for c in [*conns]:
        v.append([*conns[c].values()])
    v = np.array(v).reshape(16)  # convert the connections into a vector
    v = np.delete(v, np.where(v == 0))  # delete the connections that are zero

    return v

#
# conn_labels = ["w_pyrpyr", "w_pyrbic", "w_pyrpv",
#                "w_bicpyr",
#                "w_pvpyr", "w_pvpv", "w_pvcck",
#                "w_cckpv", "w_cckcck"]

ref = PRM_v2()
v_ref = []

for c in [*ref.conns]:
    v_ref.append([*ref.conns[c].values()])
v_ref = np.array(v_ref).reshape(16)

c_list = ["pyr", "bic", "pv", "cck"]

conn_labels = []
for c in c_list:
    for c2 in c_list:
        conn_labels.append(f"$w_{{{c.upper()} \\rightarrow {c2.upper()}}}$")

invalid = np.where(v_ref == 0)
conn_labels = np.delete(conn_labels, invalid)

# print(create_conn_vector(conn_data_2[0]))

c0 = create_conn_vector(conn_data_1[0])
#
c_array = np.zeros([num_conns, len(c0)])
#
for idx in range(num_conns):
    c_array[idx] = create_conn_vector(conn_data[idx])
#
# print(c_array[:,0])

def conn_stats(conn_vector, label=None, Plot=False):
    conn_mean = np.mean(conn_vector)
    conn_std = np.std(conn_vector)

    conn_hist = np.histogram(conn_vector, bins=6)

    if Plot:
        plt.stairs(*conn_hist, fill=True)
        plt.title(label)

    return conn_mean, conn_std, conn_hist

# conn_stats(c_array[:, 0], conn_labels[0], True)

c_stats = np.zeros((len(conn_labels), 2))

for idx in range(len(c0)):
    # plt.figure()
    c_stats[idx] = conn_stats(c_array[:, idx], conn_labels[idx])[:-1]

for idx, [label, stats] in enumerate(zip(conn_labels, c_stats)):
    print(f"{label} Mean: {stats[0].round(3)}, SD: {stats[1].round(3)}")

# fig, ax = plt.subplots(figsize=[16, 9])
# bp = ax.boxplot(c_array)
#
# ax.set_xticklabels(conn_labels)

plt.show()