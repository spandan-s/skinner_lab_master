import numpy as np

from prm_v2 import *
plt.style.use("seaborn-poster")
plt.rcParams.update({"font.size": 20})


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
                    stim_start=4.0, stim_cck=-1):
    # silence CCK mid simulation
    time = np.arange(0, 8, dt)
    stim = {
        'pyr': np.zeros_like(time),
        'bic': np.zeros_like(time),
        'cck': np.zeros_like(time),
        'pv': np.zeros_like(time)
    }
    # stim_start = 4.0
    stim['cck'][int(stim_start * fs):] = stim_cck

    P.set_init_state(len(time))
    P = simulate(time, P, stim=stim)
    ret = P.R

    return time, ret


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
    stim['cck'][int(stim_stop * fs):] = 0

    P.set_init_state(len(time))
    P = simulate(time, P, stim=stim)
    ret = P.R

    return time, ret


def silence_plot(n):
    labels = ["PYR", "BiC", "CCK", "PV"]

    P = import_conns(n)
    # R1 = baseline_sim(P)
    # R2 = silence_CCK_start(P)
    t3, R3 = silence_CCK_mid(P)
    t4, R4 = stim_CCK_mid(P, stim_cck=0.5)
    t5, R5 = silence_CCK_mid(P, stim_cck=20)

    # plot the three activities
    fig, ax = plt.subplots(3, 1, sharex=True, sharey=True, figsize=[14, 10])

    # for idx, Rx in enumerate([R1, R2, R3, R4]):
    for idx, Rx in enumerate([R3, R4, R5]):
        for ctype in ['pyr', 'bic', 'cck', 'pv']:
            ax[idx].plot(time, Rx[ctype], label=P.labels[ctype])
        ax[idx].set_ylabel('Activity [Hz]')
        # ax[idx].legend(loc=1)

    ax[0].axvline(4.0, color='black', linestyle='--')
    ax[1].axvline(4.0, color='black', linestyle='--')
    ax[2].axvline(4.0, color='black', linestyle='--')
    ax[1].axvline(4.25, color='black', linestyle='--')


    plt.xlim((3.75, 4.5))
    plt.xlabel('Time [s]')
    plt.savefig(f"figures/silence_cck/fig_7/silence_CCK_10_{n}_v3_1.pdf")


# silence_plot(22)
# for idx, c_set in enumerate([0, 2, 3, 7, 8, 22, 38, 50, 103, 135]):
#     t, R = stim_CCK_mid(import_conns(c_set))
#     save_signal(f"signals/stim_CCK_10_{c_set}_v2.dat", time=t, R=R)
#     print(f"Completed for Set 10-{c_set}")

t, R = silence_CCK_mid(import_conns(135), stim_cck=20)
# save_signal("signals/silence_CCK_10_38.dat", time=t, R=R)

# IC = [0, 20, 20, 0]
# R = silence_CCK_start(import_conns(7))
# R = baseline_sim(import_conns(38), IC)
#
for ctype in ['pyr', 'bic', 'cck', 'pv']:
    plt.plot(time, R[ctype])
plt.xlim(3.75, 4.5)
plt.axvline(4.0, color='black', linestyle='--')
#
# plt.xlim(0, 2)

plt.show()
