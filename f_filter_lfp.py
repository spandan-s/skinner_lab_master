import numpy as np
from scipy.signal import butter, iirnotch, filtfilt, sosfiltfilt, firls, firwin2


def filter_lfp(y, fs, cutoff, order=3, type='sos'):
    """
    performs a forward-backward bandpass filtering of a signal using a butterworth filter
    :param y: numpy array containing the signal
    :param fs: sampling rate of the signal
    :param cutoff: low and high cutoff to filter signal between in the form of [lo, hi]
    :param order: order of the filter -->
    actual order of filter is double this due to forward-backward filtering (default = 3)
    :param type: type of filter output --> 'ba' or 'sos' (default = 'sos')
    :return: y_filt --> filtered signal
    """
    nyq = 0.5 * fs
    cutoff = np.array(cutoff)
    band = cutoff / nyq
    if type == 'ba':
        b, a = butter(order, band, btype='band', output=type)
        y_filt = filtfilt(b, a, y)
        return y_filt
    elif type == 'sos':
        sos = butter(order, band, btype='band', output=type)
        y_filt = sosfiltfilt(sos, y)
        return y_filt
    else:
        print('Invalid output type. Valid output types are ba and sos')
        print('Returning unfiltered signal')
        return y


def notch_filt(y, fs, w0, Q=30):
    """
    performs a notch filter using an IIR filter
    :param y: numpy array containing signal
    :param fs: sampling rate of signal
    :param w0: frequency to remove using notch filter
    :param Q: quality factor (default = 30)
    :return: y_filt: signal filtered to remove specified frequency
    """
    b, a = iirnotch(w0, Q, fs)
    y_filt = filtfilt(b, a, y)

    return y_filt
