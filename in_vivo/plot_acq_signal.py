import os
from typing import List

import bioread
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import iirnotch, filtfilt, periodogram

# plt.style.use("fivethirtyeight")

def pre_filter(Y, fs, cutoff=[0.5, 120], w0=60, Q=80):
    # y_filt_60Hz = notch_filt(Y, fs, w0, Q=20)
    b, a = iirnotch(w0, Q, fs)
    y_filt = filtfilt(b, a, Y)
    # y_filt = filter_lfp(y_filt_60Hz, fs, cutoff)

    return y_filt

file_dir = "/media/spandans/One Touch/"

metadata_xl = pd.read_excel(f"{file_dir}/{[file for file in os.listdir(file_dir) if file.endswith('xlsx')][0]}")
sub_dirs: list[str] = [file for file in os.listdir(file_dir) if file[0].isdigit()]

file_list: dict = {}

for sub_dir in sub_dirs:
    file_list[sub_dir] = os.listdir(f"{file_dir}/{sub_dir}")

sub_dir_num = 5

sub_dir = sub_dirs[sub_dir_num]
file_nums = np.arange(11, len(file_list[sub_dirs[sub_dir_num]]))

# print(file)

channel_num = np.array([40, 41, 42, 44, 45, 46, 47])
channel_num -= 24

for file_num in file_nums[:20]:
    file = file_list[sub_dir][file_num]
    DATA = bioread.read_file(f"{file_dir}/{sub_dir}/{file}")

    fig, ax = plt.subplots(len(channel_num),
                           figsize=[16, 9])

    for idx, ch in enumerate(channel_num):
        time = DATA.channels[ch].time_index
        signal = DATA.channels[ch].data
        fs = DATA.samples_per_second

        signal_filt = pre_filter(signal, fs, w0=50)
        signal_filt = pre_filter(signal_filt, fs, w0=30)

        ax[idx].plot(time, signal_filt)
        ax[idx].axhline(0.5, ls='--', color='C1')
        ax[idx].axhline(-0.5, ls='--', color='C1')
        ax[idx].set_ylim((-1, 1))
        ax[idx].set_title(f"Channel No: {ch+24}")
    # fxx, Pxx = periodogram(signal_filt, fs)
    # plt.plot(fxx, Pxx)
    # plt.semilogx()

    # plt.plot(time, signal_filt)

    save_dir = f"/home/spandans/skinner_lab_master/in_vivo/TBI_Aylin_2023/{sub_dir}"
    try:
        os.makedirs(save_dir)
    except FileExistsError:
        pass

    plt.tight_layout()
    plt.savefig(f"{save_dir}/{file.split('.')[0]}.png")
    print(f"Completed for file: {file}")
    plt.close()
# plt.show()



