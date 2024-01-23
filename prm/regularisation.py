import numpy as np

from prm_v2 import *

def spike_timing(DATA):
    R = {}
    for idx, ctype in enumerate(['pyr', 'bic', 'pv', 'cck']):
        R[ctype] = DATA[:, idx + 1]

    SR = spike_raster(R)
    spikes = {}
    for idx, ctype in enumerate(['pyr', 'bic', 'pv', 'cck']):
        spikes[ctype] = np.array([DATA[:, 0][SR[ctype]], DATA[:, idx+1][SR[ctype]]])

    isi = {}
    for idx, ctype in enumerate(['pyr', 'bic', 'cck', 'pv']):
        isi[ctype] = np.diff(spikes[ctype][0])

    def spike_timing_stats():
        isi_stats = {}
        spike_stats = {}
        for idx, ctype in enumerate(['pyr', 'bic', 'cck', 'pv']):
            if len(isi[ctype]) == 0:
                isi_stats[ctype] = {
                    'mean': np.nan,
                    'median': np.nan,
                    'std': np.nan
                }
                spike_stats[ctype] = {
                    'mean': np.nan,
                    'median': np.nan,
                    'std': np.nan
                }
            else:
                isi_stats[ctype] = {
                    'mean': np.mean(isi[ctype]),
                    'median': np.median(isi[ctype]),
                    'std': np.std(isi[ctype])
                }
                spike_stats[ctype] = {
                    'mean': np.mean(spikes[ctype][1]),
                    'median': np.median(spikes[ctype][1]),
                    'std': np.std(spikes[ctype][1])
                }
        return isi_stats, spike_stats

    isi_stats, spike_stats = spike_timing_stats()

    return isi_stats, spike_stats


DATA = np.loadtxt("signals/silence_CCK_10_38.dat",
                  skiprows=1)

time = DATA[:, 0]

DATA_1 = DATA[int(1*fs):int(5*fs)]
DATA_2 = DATA[int(6*fs):]
#
# # plt.plot(DATA_1[:, 0], DATA_1[:, 1])
# # plt.plot(DATA_1[:, 0], DATA_2[:, 1])
#
# R1, R2 = {}, {}
# for idx, ctype in enumerate(['pyr', 'bic', 'cck', 'pv']):
#     R1[ctype] = DATA_1[:, idx+1]
#     R2[ctype] = DATA_2[:, idx+1]
#
# SR1 = spike_raster(R1)
# SR2 = spike_raster(R2)
#
# plt.plot(DATA_1[:, 0][SR1['pyr']], DATA_1[:, 1][SR1['pyr']], 'x')
# plt.plot(DATA_1[:, 0][SR2['pyr']], DATA_2[:, 1][SR2['pyr']], 'x')

# print(spike_timing(DATA_1))
# print(spike_timing(DATA_2))

stats1 = spike_timing(DATA_1)
stats2 = spike_timing(DATA_2)

for ctype in ['pyr', 'bic']:
    print(ctype.upper())
    print("mean amp", np.round(stats1[1][ctype]['mean'], 3), np.round(stats2[1][ctype]['mean'], 3))
    print("median amp", np.round(stats1[1][ctype]['median'], 3), np.round(stats2[1][ctype]['median'], 3))
    print("std amp", np.round(stats1[1][ctype]['std'], 3), np.round(stats2[1][ctype]['std'], 3))
    print("mean isi", np.round(stats1[0][ctype]['mean'], 3), np.round(stats2[0][ctype]['mean'], 3))
    print("median isi", np.round(stats1[0][ctype]['median'], 3), np.round(stats2[0][ctype]['median'], 3))
    print("std isi", np.round(stats1[0][ctype]['std'], 3), np.round(stats2[0][ctype]['std'], 3))

plt.show()