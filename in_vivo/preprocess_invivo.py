import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from scipy.signal import periodogram
from f_filter_lfp import notch_filt, filter_lfp

def pre_filter(Y, fs, cutoff=[0.5, 120], w0=60):
    y_filt_60Hz = notch_filt(Y, fs, w0, Q=20)
    # y_filt = filter_lfp(y_filt_60Hz, fs, cutoff)

    return y_filt_60Hz

def find_jumps(Y, threshold=0.03, mask_len=10):
    # find element-wise difference of signal
    diff_Y = np.diff(Y)

    # find where abs(difference signal) is greater than the threshold
    isValid = abs(diff_Y) > threshold

    isValid = np.convolve(isValid, np.ones(mask_len, ), mode='same') > 0
    isValid = np.clip(isValid, -1, 1)

    d = abs(np.diff(isValid))
    jumps = np.where(d > 0.5)

    return jumps

def cut_signal(X, Y, fs):

    jumps = find_jumps(Y)

    time_jumps = X[jumps]

    eps = 0.0001
    cut_pts = np.zeros(len(time_jumps), dtype=int)

    for idx, val in enumerate(time_jumps):
        x = np.where(abs(X - val) <= eps)[0][0]
        cut_pts[idx] = x

    cut_pts = np.append(cut_pts, len(X) - 1)
    num_sections = len(cut_pts)
    min_length = int(fs * 30)
    max_length = int(fs * 60)
    cut_signals_X, cut_signals_Y = [], []

    start, stop = 0, cut_pts[0]
    count = 0
    for idx in range(num_sections):
        while (stop - start >= max_length):
            cutX = np.array(X[start: start + max_length])
            cutY = np.array(Y[start: start + max_length])
            cut_signals_X.append(cutX)
            cut_signals_Y.append(cutY)
            start = start + max_length
        if (stop - start >= min_length):
            cutX = np.array(X[start:stop])
            cutY = np.array(Y[start:stop])
            cut_signals_X.append(cutX)
            cut_signals_Y.append(cutY)
        count += 1
        start = stop
        if count < len(cut_pts):
            stop = cut_pts[count]

    return cut_signals_X, cut_signals_Y

path_in = "/media/spandans/Transcend/TBI/first group/"
path_out = ""

run_name = "2-17_and_1-17_bot_amplifer2022-11-16T11_32_25"
# specify subject number (1 or 2)
subject_no = 1

# specify channel number (1 - 4)
channel_no = 1

if subject_no == 2:
    channel_no += 8

# if bool(re.search(r"top", run_name)):
#     if subject_no == 2:
#         channel_no += 8
# elif bool(re.search(r"bot", run_name)):
#     if subject_no == 1:
#         channel_no += 39
#     elif subject_no == 2:
#         channel_no += 47

fname = f"{path_in}{run_name}/Subject_{subject_no}/channel_{channel_no}.dat"

try:
    os.mkdir(f"{path_in}{run_name}/Subject_{subject_no}/channel_{channel_no}_cut")
except FileExistsError:
    pass

DATA = np.loadtxt(fname)
dt = DATA[1, 0] - DATA[0, 0]
Fs = np.round(1/dt)

DATA[:, 1] = pre_filter(DATA[:, 1], Fs)

cut_X, cut_Y = cut_signal(DATA[:, 0], DATA[:, 1], Fs)
num_signals = len(cut_X)

fxx, Pxx = periodogram(DATA[:, 1], Fs)

fig, ax = plt.subplots(2, 1)

# create dataframe to store cut signal details
df = pd.DataFrame(columns=["Filename", "Duration"])

# ax[0].plot(DATA[:, 0], DATA[:, 1])
for idx in range(num_signals):
    # ax[0].plot(cut_X[idx], cut_Y[idx], 'C0')
    sname = f"channel_{channel_no}_cut{idx}.dat"
    new_row = pd.DataFrame({"Filename": sname, "Duration": f"{[int(cut_X[idx][0]), int(cut_X[idx][-1])]}"},
                           index=[0])
    df = pd.concat([new_row, df.loc[:]]).reset_index(drop=True)
    save_data = np.zeros((len(cut_X[idx]), 2))
    save_data[:, 0] = cut_X[idx]
    save_data[:, 1] = cut_Y[idx]
    np.savetxt(f"{path_in}{run_name}/Subject_{subject_no}/channel_{channel_no}_cut/{sname}",
               save_data)

# ax[0].plot(DATA[:, 0], DATA[:, 1], 'C1', alpha=0.3)
# ax[1].set_yscale("log")
# ax[1].plot(fxx, Pxx)
# ax[1].set_xlim(0, 120)

df.to_csv(f"{path_in}{run_name}/Subject_{subject_no}/channel_{channel_no}_cut/run_details.csv")
# plt.show()

