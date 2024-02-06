import sys

import numpy as np
import matplotlib.pyplot as plt
import pyabf
import os
from f_filter_lfp import filter_lfp, notch_filt
from scipy.signal import decimate, find_peaks, periodogram, get_window, welch


def pre_filter(Y, fs, cutoff=[0.5, 100], w0=[]):
    """
    filter the signal to remove any components below a low and high cutoff
    as well as remove 60 Hz DC component
    """
    y_filt_60Hz = notch_filt(Y, fs, 60, Q=50)
    for ws in w0:
        y_filt_60Hz = notch_filt(y_filt_60Hz, fs, ws, Q=20)

    y_filt = filter_lfp(y_filt_60Hz, fs, cutoff)

    return y_filt


def downsample_signal(X, Y, M=50):
    """
    downsample signal
    :param M: downsample factor - one sample per M samples is retained in the downsampled signal
    """
    # down-sample signal to make difference calculation easier
    ds_Y = decimate(Y, M)
    newX = np.linspace(X[0], X[-1], len(ds_Y), endpoint=False)

    return newX, ds_Y


def find_jumps(Y, fs, threshold=0.025, mask_len=5):
    """
    find where there are large "jumps" in the signal
    uses a convolution with a constant function to "smooth out" discontinuities
    returns indices of jumps
    """

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
    """
    cuts signal into sections of length < 30s that don't contain jumps
    """
    newX, ds_Y = downsample_signal(X, Y)

    jumps = find_jumps(ds_Y, fs)

    time_jumps = newX[jumps]

    eps = 0.0001
    cut_pts = np.zeros(len(time_jumps), dtype=int)

    for idx, val in enumerate(time_jumps):
        x = np.where(abs(X - val) <= eps)[0][0]
        cut_pts[idx] = x

    cut_pts = np.append(cut_pts, len(X) - 1)
    # print(X[cut_pts])
    num_sections = len(cut_pts)
    min_length = fs * 10
    max_length = fs * 60
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


def count_spikes(X, Y):
    plt.figure()
    plt.plot(X, Y)

    # find the mean voltage
    mean_Y = np.mean(Y)
    plt.axhline(mean_Y + 0.01, color='C1')
    # find peaks in the delta range that are a certain height above the mean
    peaks, _ = find_peaks(Y, height=mean_Y + 0.01, distance=6667)
    plt.plot(X[peaks], Y[peaks], 'x')

    # check if there are at least x events depending on the length of the section
    num_peaks = len(peaks)
    text_loc = [120 + 0.75 * np.ptp(X), np.min(Y)]
    # required number of peaks = 1 peak per 2 seconds => 1 peak per 2*fs samples
    # for signal of x seconds = x*fs samples => 0.5*x/fs samples
    req_peaks = round(0.5 * np.ptp(X))
    plt.title(f'no. of peaks: {num_peaks}/{req_peaks}')

    print(f'no. of peaks: {num_peaks}/{req_peaks}')

    if num_peaks >= req_peaks:
        return True
    else:
        return False

def plot_periodograms(X, Y, fs):
    delta_range = [0.5, 3]
    theta_range = [3, 15]
    gamma_range = [15, 100]

    Y_filt_delta = filter_lfp(Y, fs, delta_range)
    Y_filt_theta = filter_lfp(Y, fs, theta_range)
    Y_filt_gamma = filter_lfp(Y, fs, gamma_range)

    segment = int(fs * 10)
    myhann = get_window('hann', segment)

    myparams = dict(fs=fs, nperseg=segment, window=myhann,
                    noverlap=segment / 4, scaling='density', return_onesided=True)

    # fxx_delta, pxx_delta = periodogram(Y_filt_delta, fs)
    # fxx_theta, pxx_theta = periodogram(Y_filt_theta, fs)
    # fxx_gamma, pxx_gamma = periodogram(Y_filt_gamma, fs)

    fxx_delta, pxx_delta = welch(Y_filt_delta, **myparams)
    fxx_theta, pxx_theta = welch(Y_filt_theta, **myparams)
    fxx_gamma, pxx_gamma = welch(Y_filt_gamma, **myparams)

    peak_delta = [fxx_delta[np.argmax(pxx_delta)], np.max(pxx_delta)]
    peak_theta = [fxx_theta[np.argmax(pxx_theta)], np.max(pxx_theta)]
    peak_gamma = [fxx_gamma[np.argmax(pxx_gamma)], np.max(pxx_gamma)]

    fig, ax = plt.subplots(1, 3, figsize=[16, 9])
    ax[0].set_title('Delta Periodogram')
    ax[0].plot(fxx_delta, pxx_delta)
    ax[0].set_xlim(-0.5, 7)
    ax[0].set_xlabel('Frequency [Hz]')
    ax[0].set_ylabel('Theta Spectral Density')

    ax[1].set_title('Theta Periodogram')
    ax[1].plot(fxx_theta, pxx_theta)
    ax[1].set_xlim(1.5, 20)
    ax[1].set_xlabel('Frequency [Hz]')
    ax[1].set_ylabel('Theta Spectral Density')

    ax[2].set_title('Gamma Periodogram')
    ax[2].plot(fxx_gamma, pxx_gamma)
    ax[2].set_xlim(10, 120)
    ax[2].set_xlabel('Frequency [Hz]')
    ax[2].set_ylabel('Gamma Spectral Density')

    plt.tight_layout()
    return peak_delta, peak_theta, peak_gamma

def check_validity(X, Y, fs):
    valid = count_spikes(X, Y)

    if valid:
        peak_delta, peak_theta, peak_gamma = plot_periodograms(X, Y, fs)

        print(f'Peak Theta: {peak_theta[0]:.2E} Hz; {peak_theta[1]:.2E} mV^2/Hz')
        print(f'Peak Delta: {peak_delta[0]:.2E} Hz; {peak_delta[1]:.2E} mV^2/Hz')
        print(f'Peak Gamma: {peak_gamma[0]:.2E} Hz; {peak_gamma[1]:.2E} mV^2/Hz')
    return valid

# ==========================================================
# MAIN PROGRAM

# load abf file and X and Y data
dir = '/media/spandans/Transcend/Liang_TBI_09_11_23/REID DATA/'
# os.chdir(dir)

id = "20230330 ID2B33 TBI/"
fn = '230330_0000'

# load signal from abf file
abf = pyabf.ABF(f'{dir}{id}{fn}.abf')

try:
    os.makedirs(f"./TBI_analysis_results/{id}/")
except FileExistsError:
    pass

sys.stdout = open(f'./TBI_analysis_results/{id}{fn}.txt')

print('==================================================')
print(f'Processing signal from file: {fn}')

sweepX = abf.sweepX
sweepY = abf.sweepY

plt.figure()
plt.plot(sweepX, sweepY)

fs = abf.sampleRate

Y_filt = pre_filter(sweepY, fs, w0=[40, 80])
# plot_periodograms(sweepX[int(300*fs):int(500*fs)], Y_filt[int(300*fs):int(500*fs)], fs=fs)

# cut signal into "valid" chunks
cut_signal_X, cut_signal_Y = cut_signal(sweepX, Y_filt, fs)

# check validity of each section and plot frequency domain stuff
num_signals = len(cut_signal_X)
# print(num_signals)
for idx in range(num_signals):
    signal_X, signal_Y = cut_signal_X[idx], cut_signal_Y[idx]

    print('==================================================')
    print(f'Processing section {idx +1} from {round(signal_X[0], 2)} to {round(signal_X[-1], 2)} seconds')

    validity = check_validity(signal_X, signal_Y, fs)
    print(f'Signal validity: {validity}')
    print('==================================================\n')

# plt.show()
plt.close()
sys.stdout.close()