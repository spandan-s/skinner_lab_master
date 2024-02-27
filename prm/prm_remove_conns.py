import os

import matplotlib.pyplot as plt
import pandas as pd

from prm_v2 import *

def quantify_conns_removal(conn_set_num):
    """
    in_conns: index of connection set in "search_results_conn_10.json"

    remove each synaptic weight and simulate the PRM
    find the theta and gamma frequency and power for each case

    return:
    data_out: dataframe with entry for each conn_removal
    """

    ref_theta = 244.92
    ref_gamma = 1.55

    with open("search_results/search_results_conn_10.json", "r") as f:
        conn_data = np.array(json.load(f))

    conns = conn_data[conn_set_num]

    df_out = pd.DataFrame(columns=[
        "conn_set", "removed_conn", "theta_freq", "theta_power", "gamma_freq", "gamma_power"
    ])

    data_out = []

    conns_nz = {
        "pyr": ["pyr", "bic", "pv"],
        "bic": ["pyr"],
        "cck": ["pv", "cck"],
        "pv": ["pyr", "pv", "cck"],
    }

    P = PRM_v2(conns_dict=conns)
    P.set_init_state(len(time))
    P = simulate(time, P)
    theta_freq, theta_power = find_pyr_power(P.R, fs=fs, band='theta')
    gamma_freq, gamma_power = find_pyr_power(P.R, fs=fs, band='gamma')
    data_out.append({
        "conn_set": f"10-{conn_set_num}",
        "removed_conn": f"none",
        "theta_freq": theta_freq,
        "theta_power": theta_power,
        "theta_power_ratio": theta_power/ref_theta,
        "gamma_freq": gamma_freq,
        "gamma_power": gamma_power,
        "gamma_power_ratio": gamma_power/ref_gamma,
    })

    for c1 in [*conns_nz]:
        for c2 in [*conns_nz[c1]]:
            P = PRM_v2(conns_dict=conns)
            P.conns[c1][c2] = 0

            P.set_init_state(len(time))
            P = simulate(time, P)
            theta_freq, theta_power = find_pyr_power(P.R, fs=fs, band='theta')
            gamma_freq, gamma_power = find_pyr_power(P.R, fs=fs, band='gamma')

            data_out.append({
                "conn_set": f"10-{conn_set_num}",
                "removed_conn": f"{c1}_{c2}",
                "theta_freq": theta_freq,
                "theta_power": theta_power,
                "theta_power_ratio": theta_power/ref_theta,
                "gamma_freq": gamma_freq,
                "gamma_power": gamma_power,
                "gamma_power_ratio": gamma_power/ref_gamma
            })

            # plt.figure()
            # for ctype in ["pyr"]:
            #     plt.plot(time[7*len(time)//8:], P.R[ctype][7*len(time)//8:], label=ctype)
            # plt.title(f"removed {c1} {c2} connection")
            # # plt.legend()
            # plt.savefig(f"{sdir}/{c1}_{c2}.png")
            # plt.close()
    df_out = pd.concat([df_out, pd.DataFrame(data_out)], ignore_index=True)

    return df_out

big_df = pd.DataFrame()

for idx in [0, 2, 3, 7, 8, 22, 38, 50, 103, 135]:
    output = quantify_conns_removal(idx)
    big_df = pd.concat([big_df, output], ignore_index=True)

big_df.to_csv("sheets/quantifying_conns_removal_v2.csv")

# plt.show()