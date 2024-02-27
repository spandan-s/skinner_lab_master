import numpy as np
import matplotlib.pyplot as plt
from pactools.utils import BandPassFilter, Spectrum
from scipy.signal import welch, get_window

def calc_spectral(signal, fs, xlim=[0, 10]):
    segment = int(fs * 4)
    myhann = get_window('hann', segment)

    myparams = dict(fs=fs, nperseg=segment, window=myhann,
                    noverlap=segment / 2, scaling='density', return_onesided=True)

    pgram = welch(signal, **myparams)

    plt.figure()
    plt.plot(pgram[0], pgram[1])
    plt.grid()
    # plt.semilogy()
    plt.xlim(xlim)

fs = 10000.

t, signal = np.loadtxt("C:/Users/spand/Documents/Skinner_Lab_Analysis/FSM_LFP/lfp_bez_sca_5013.dat",
                       unpack=True)

f_theta = BandPassFilter(fs=fs, fc=7.5, bandwidth=4.5, n_cycles=None)
f_gamma = BandPassFilter(fs=fs, fc=70, n_cycles=7.0, bandwidth=None)

theta = f_theta.transform(signal)
gamma = f_gamma.transform(signal)

calc_spectral(theta, fs=fs, xlim=[2, 15])

calc_spectral(gamma, fs=fs, xlim=[20, 100])


# plt.plot(t, signal)
# plt.plot(t, theta)
# plt.plot(t, gamma)

plt.show()