import json

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch, get_window, iirnotch

from prm_v2 import *
from f_filter_lfp import notch_filt, filter_lfp

with open("search_results/search_results_conn_8.json", "r") as f:
    conn_data = json.load(f)

# new_prm = PRM_v2(conn_data[0])
#
# new_prm.set_init_state(len(time))
# new_prm = simulate(time, new_prm)

print(ref_power())

# print(find_pyr_power(conn_data[0], fs, band="theta"))
# print(find_pyr_power(conn_data[0], fs, band="gamma"))

# # remove first second of signal
# one_sec = int(1/dt)
#
# signal_pyr = new_prm.R["pyr"][one_sec:]
# signal_pv = new_prm.R["pv"][one_sec:]
#
# pv_filt = filter_lfp(signal_pv, fs, [20, 100])
#
# fxx_pyr, Pxx_pyr = welch(signal_pyr, **myparams)
# fxx_pv, Pxx_pv = welch(signal_pv, **myparams)
# # plt.plot(time[one_sec:], signal_pyr)
# plt.plot(time[one_sec:], signal_pv)
# plt.plot(time[one_sec:], pv_filt)
#
#
#
# plt.figure()
#
# plt.plot(fxx_pyr[(fxx_pyr < 100)],
#          Pxx_pyr[(fxx_pyr < 100)])
# #
# #
# plt.plot(fxx_pv[(fxx_pv < 100)],
#          Pxx_pv[(fxx_pv < 100)])
#
# # peak_freq = [fxx[np.argmax(Pxx)], np.max(Pxx)]
# #
# # harmonics = np.arange(20, 100, peak_freq[0])
# # print(harmonics)
#
# # for hf in harmonics:
# #     signal = notch_filt(signal, fs, hf, Q=10)
#
# # fxx, Pxx = welch(signal, **myparams)
# # peak_freq_2 = [fxx[np.argmax(Pxx[(fxx > 20) & (fxx < 100)])], np.max(Pxx[(fxx > 20) & (fxx < 100)])]
# #
# # print(peak_freq_2)
# #
# # plt.plot(fxx[(fxx > 20) & (fxx < 100)], Pxx[(fxx > 20) & (fxx < 100)])
# plt.semilogy()

plt.show()