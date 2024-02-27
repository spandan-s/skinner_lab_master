from scipy.signal import spectrogram, ShortTimeFFT
from scipy.signal.windows import gaussian
import numpy as np
import matplotlib.pyplot as plt


def create_test_signal(t_end=120, fs=5e3,
                       w1=2.0, w2=6.0, w3=80.0,
                       D=0.01):
    dt = 1 / fs
    t = np.arange(0, t_end, dt)

    y1_phase = 2 * np.pi * np.random.random()
    y2_phase = 2 * np.pi * np.random.random()
    y3_phase = 2 * np.pi * np.random.random()

    y1_amp = np.abs(np.random.normal(1))
    y2_amp = np.abs(np.random.normal(1))
    y3_amp = 0.01*np.abs(np.random.normal(1))

    y1 = y1_amp * np.sin(2 * np.pi * w1 * (t + y1_phase))
    y2 = y2_amp * np.sin(2 * np.pi * w2 * (t + y2_phase))
    y3 = y3_amp * np.sin(2 * np.pi * w3 * (t + y3_phase))

    Y = y1 + y2 + y3 + np.random.normal() * D

    return t, Y


t, data = create_test_signal()

# f, t, Sxx = spectrogram(data[3000:10000], fs=5e3,
                        # window='hamming', nperseg=1200, noverlap=128)

# plt.pcolormesh(t, f, Sxx, shading='gouraud')
# plt.ylabel('Frequency [Hz]')
# plt.xlabel('Time [sec]')
# plt.ylim((0, 120))

g_std = 8  # standard deviation for Gaussian window in samples
w = gaussian(5000, std=g_std, sym=True)  # symmetric Gaussian window

SFT = ShortTimeFFT(w, hop=10, fs=5000, mfft=10000, scale_to='magnitude')

Sx = SFT.stft(data[3000:25000])  # perform the STFT
N = 9000

fig1, ax1 = plt.subplots(figsize=(6., 4.))  # enlarge plot a bit
t_lo, t_hi = SFT.extent(N)[:2]  # time range of plot
ax1.set_title(rf"STFT ({SFT.m_num*SFT.T:g}$\,s$ Gaussian window, " +
              rf"$\sigma_t={g_std*SFT.T}\,$s)")
ax1.set(xlabel=f"Time $t$ in seconds ({SFT.p_num(N)} slices, " +
               rf"$\Delta t = {SFT.delta_t:g}\,$s)",
        ylabel=f"Freq. $f$ in Hz ({SFT.f_pts} bins, " +
               rf"$\Delta f = {SFT.delta_f:g}\,$Hz)",
        xlim=(t_lo, t_hi))
im1 = ax1.imshow(abs(Sx), origin='lower', aspect='auto',
                 extent=SFT.extent(N), cmap='viridis')
fig1.colorbar(im1, label="Magnitude $|S_x(t, f)|$")
ax1.legend()
fig1.tight_layout()

# plt.plot(t, data)
#
plt.show()
