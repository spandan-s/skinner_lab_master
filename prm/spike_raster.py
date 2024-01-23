import matplotlib.pyplot as plt

from prm_v2 import *
plt.style.use("seaborn-v0_8-poster")
plt.rcParams.update({"font.size": 20})

def plot_spike_raster(n, xmin, xmax, ymin, ymax):
    # P = import_conns(n)
    #
    # R = baseline_sim(P)
    DATA = np.loadtxt(f"signals/stim_CCK_10_{n}_v2.dat",
                      skiprows=1)
    R = {}
    for idx, ctype in enumerate(['pyr', 'bic', 'pv', 'cck']):
        R[ctype] = DATA[:, idx + 1]
    R_section = {}

    for ctype in ['pyr', 'bic', 'cck', 'pv']:
        R_section[ctype] = R[ctype][int(xmin*fs):int(xmax*fs)]

    SR = spike_raster(R_section)

    plot_colours = {
        'pyr': 'C0',
        'bic': 'C1',
        'cck': 'C2',
        'pv': 'C3'
    }

    # plt.figure(figsize=[16, 10])
    for ctype in ['cck', 'pv']:
        for spike in SR[ctype]:
            plt.axvline(time[int(xmin*fs):int(xmax*fs)][spike], ymin, ymax,
                        color=plot_colours[ctype], alpha=0.8, lw=4)

plt.figure(figsize=[16, 9])
for idx, c_set in enumerate([0, 2, 3, 7, 8, 22, 38, 50, 103, 135]):
    plot_spike_raster(c_set, 3.85, 4.25, (1 - 0.1*idx), (1.01-0.1*(idx+1)))

plt.savefig("figures/spike_raster_cck_stim.pdf")
plt.show()