import numpy as np
import matplotlib.pyplot as plt

from pactools import Comodulogram, REFERENCES
from pactools import simulate_pac
from pactools.dar_model import DAR, extract_driver

fs = 10000.  # Hz
high_fq = 50.0  # Hz
low_fq = 8.0  # Hz
low_fq_width = 1.0  # Hz

n_points = 10000
noise_level = 0.4

# signal = simulate_pac(n_points=n_points, fs=fs, high_fq=high_fq, low_fq=low_fq,
#                       low_fq_width=low_fq_width, noise_level=noise_level,
#                       random_state=0)

t, signal = np.loadtxt("C:/Users/spand/Documents/Skinner_Lab_Analysis/FSM_LFP/lfp_bez_sca_5013.dat",
                       unpack=True)

# ============== COMODULOGRAM ==============================
low_fq_range = np.linspace(1, 10, 50)
high_fq_range = np.linspace(20, 120, 100)

method = 'tort'

estimator = Comodulogram(fs=fs, low_fq_range=low_fq_range,
                         low_fq_width=low_fq_width,
                         high_fq_range=high_fq_range,
                         n_surrogates=200, n_jobs=8,
                         method=method, progress_bar=True)
estimator.fit(signal)
estimator.plot(contour_method='comod_max', contour_level=0.05)

# plt.ylim([20, 120])
# ============================================================

# ====================== DAR MODEL ===========================
#
# # Prepare the plot for the two figures
# fig, axs = plt.subplots(1, 2, figsize=(10, 4))
# axs = axs.ravel()
#
# sigdriv, sigin, sigdriv_imag = extract_driver(
#     sigs=signal, fs=fs, low_fq=low_fq, bandwidth=low_fq_width,
#     extract_complex=True, random_state=0, fill=2)
#
# # Create a DAR model
# # Here we use BIC selection to get optimal hyperparameters (ordar, ordriv)
# dar = DAR(ordar=20, ordriv=8, criterion='bic')
# # Fit the DAR model
# dar.fit(sigin=sigin, sigdriv=sigdriv, sigdriv_imag=sigdriv_imag, fs=fs)
#
#
# # Plot the BIC selection
# bic_array = dar.model_selection_criterions_['bic']
# lines = axs[0].plot(bic_array)
# axs[0].legend(lines, ['ordriv=%d' % d for d in [0, 1, 2]])
# axs[0].set_xlabel('ordar')
# axs[0].set_ylabel('BIC / T')
# axs[0].set_title('BIC order selection')
# axs[0].plot(dar.ordar_, bic_array[dar.ordar_, dar.ordriv_], 'ro')
#
# # Plot the modulation extracted by the optimal model
# plt.figure()
# dar.plot()
# plt.title(dar.get_title(name=True))
# ==================================================================

plt.show()