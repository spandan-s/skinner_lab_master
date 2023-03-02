import numpy as np
import matplotlib.pyplot as plt
from f_filter_lfp import filter_lfp
from scipy.signal import welch

run_name = "1-15_and_2-13_top_amplifer"
subject_no = 1
channel_no = 1
cut_no = 3

path_in = f"/media/spandans/Transcend/TBI/first group/\
{run_name}/Subject_{subject_no}/channel_{channel_no}_cut/"

fname = f"{path_in}channel_{channel_no}_cut{cut_no}.dat"

time, signal = np.loadtxt(fname, unpack=True)

dt = time[1] - time[0]
Fs = np.round(1/dt)

cutoff_lo = [3, 15]
cutoff_hi = [20, 100]

V_lo = filter_lfp(signal, Fs, cutoff_lo)
V_hi = filter_lfp(signal, Fs, cutoff_hi)

periodogram_lo = welch(V_lo, Fs, nperseg=4*Fs)
periodogram_hi = welch(V_hi, Fs, nperseg=4*Fs)

peak_lo = [periodogram_lo[0][np.argmax(periodogram_lo[1])],
           np.max(periodogram_lo[1])]
peak_hi = [periodogram_hi[0][np.argmax(periodogram_hi[1])],
           np.max(periodogram_hi[1])]

cfc_valid = False

if (3 <= peak_lo[0] <= 15) and (20 <= peak_hi[0] <= 100):
    if 0.01 <= peak_hi[1]/peak_lo[1] <= 100:
        cfc_valid = True

if cfc_valid:
    print("Recommended for CFC Analysis")
else:
    print("Not recommended for CFC Analysis")

fig, ax = plt.subplot_mosaic(
    [["top row"]*2,
     ["bottom left", "bottom right"]]
)

ax["top row"].plot(time, signal, label='signal')
ax["top row"].plot(time, V_lo, label="$\\theta$ filtered")
ax["top row"].plot(time, V_hi, label="$\gamma$ filtered")
ax["top row"].set_xlabel("Time [s]")
ax["top row"].set_ylabel("EEG Signal [mV]")
ax["top row"].legend()

ax["bottom left"].plot(periodogram_lo[0], periodogram_lo[1])
ax["bottom right"].plot(periodogram_hi[0], periodogram_hi[1])

ax["bottom left"].set_xlabel("$\\theta$ Frequency (Hz)")
ax["bottom left"].set_xlim((0, 20))
ax["bottom left"].set_ylabel("PSD [$mV^2/Hz$]")
ax["bottom left"].set_title(f"[{peak_lo[0]:.3f} Hz, {peak_lo[1]:.3e}]")

ax["bottom right"].set_xlabel("$\gamma$ Frequency (Hz)")
ax["bottom right"].set_xlim((10, 100))
ax["bottom right"].set_title(f"[{peak_hi[0]:.3f} Hz, {peak_hi[1]:.3e}]")

plt.tight_layout()
plt.show()