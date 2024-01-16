import numpy as np

from prm_v2 import *


# def baseline_sim(P):
#     # initialise and simulate PRM at baseline
#     IC = [0, 10, 0, 0]
#     P.set_init_state(len(time), IC)
#     P = simulate(time, P)
#     ret = P.R
#     return ret


def silence_CCK_start(P):
    # silence CCK cell at start of simulation
    stim = {
        'pyr': np.zeros_like(time),
        'bic': np.zeros_like(time),
        'cck': -1 + np.zeros_like(time),
        'pv': np.zeros_like(time)
    }
    P.set_init_state(len(time))
    P.R['bic'][0] = 10
    P = simulate(time, P, stim=stim)
    ret = P.R

    return ret


def silence_CCK_mid(P,
                    stim_start=5.0):
    # silence CCK mid simulation
    new_time = np.arange(0, 10, dt)
    stim = {
        'pyr': np.zeros_like(new_time),
        'bic': np.zeros_like(new_time),
        'cck': np.zeros_like(new_time),
        'pv': np.zeros_like(new_time)
    }
    # stim_start = 4.0
    stim['cck'][int(stim_start * fs):] = -1

    P.set_init_state(len(new_time))
    P = simulate(new_time, P, stim=stim)
    ret = P.R

    return new_time, ret


def stim_CCK_mid(P,
                 stim_cck=0.25, stim_start=4.0, stim_stop=4.25):
    # silence CCK at start, add stimulation to it mid simulation
    stim = {
        'pyr': np.zeros_like(time),
        'bic': np.zeros_like(time),
        'cck': -1 + np.zeros_like(time),
        'pv': np.zeros_like(time)
    }

    # stim_start = 4.0
    # stim_stop = 4.25
    stim['cck'][int(stim_start * fs):int(stim_stop * fs)] = stim_cck
    # stim['cck'][int(stim_stop * fs):] = 0

    P.set_init_state(len(time))
    P = simulate(time, P, stim=stim)
    ret = P.R

    return ret


def silence_plot(P):
    R1 = baseline_sim(P)
    R2 = silence_CCK_start(P)
    R3 = silence_CCK_mid(P)
    R4 = stim_CCK_mid(P, stim_cck=0.5)

    # plot the three activities
    fig, ax = plt.subplots(4, 1, sharex=True, sharey=True, figsize=[14, 10])

    for idx, Rx in enumerate([R1, R2, R3, R4]):
        for ctype in ['pyr', 'bic', 'cck', 'pv']:
            ax[idx].plot(time, Rx[ctype])
        ax[idx].set_ylabel('Activity [Hz]')

    plt.xlim((3.5, 5.5))
    plt.xlabel('Time [s]')
    plt.savefig(f"figures/silence_cck/silenced_t425/silence_CCK_10_0.pdf")


# silence_plot(import_conns(0))
t, R = silence_CCK_mid(import_conns(135))
save_signal("signals/silence_CCK_10_135.dat", time=t, R=R)

# IC = [0, 20, 20, 0]
# R = silence_CCK_start(import_conns(7))
# R = baseline_sim(import_conns(38), IC)
#
# for ctype in ['pyr', 'bic', 'cck', 'pv']:
#     print(R[ctype][0])
#     plt.plot(time, R[ctype])
#
# plt.xlim(0, 2)

plt.show()
