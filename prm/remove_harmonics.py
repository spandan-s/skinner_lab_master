import json

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch, get_window, iirnotch

from prm_v2 import *
from f_filter_lfp import notch_filt

with open("search_results/search_results_conn_8.json", "r") as f:
    conn_data = json.load(f)

new_prm = PRM_v2(conn_data[12])

new_prm.set_init_state(len(time))
new_prm = simulate(time, new_prm)

# remove first second of signal
one_sec = int(1/dt)

signal = new_prm.R["pyr"][one_sec:]

# plt.plot(time[one_sec:], signal)

segment = int(fs * 4)
myhann = get_window('hann', segment)

myparams = dict(fs=fs, nperseg=segment, window=myhann,
                noverlap=segment / 2, scaling='density', return_onesided=True)

fxx, Pxx = welch(signal, **myparams)
plt.plot(fxx[(fxx > 20) & (fxx < 100)], Pxx[(fxx > 20) & (fxx < 100)])

peak_freq = [fxx[np.argmax(Pxx)], np.max(Pxx)]

harmonics = np.arange(20, 100, peak_freq[0])
print(harmonics)

for hf in harmonics:
    signal = notch_filt(signal, fs, hf, Q=10)

fxx, Pxx = welch(signal, **myparams)
peak_freq_2 = [fxx[np.argmax(Pxx[(fxx > 20) & (fxx < 100)])], np.max(Pxx[(fxx > 20) & (fxx < 100)])]

print(peak_freq_2)

plt.plot(fxx[(fxx > 20) & (fxx < 100)], Pxx[(fxx > 20) & (fxx < 100)])
plt.semilogy()

plt.show()