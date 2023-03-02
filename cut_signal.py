# cut_signal.py
# date: 06_09_22
# cut a signal into sections given start and stop epochs

import numpy as np
import matplotlib.pyplot as plt
from os import listdir

# file containing signal
path_in = '/home/spandans/Documents/FSM_LFP/'
# fname = 'lfp_bez_sca_5501.dat'
save_path = '/home/spandans/Documents/FSM_LFP/cut_signals/'

skiprows = 0
unpack = False

files = sorted(file for file in listdir(path_in) if file.endswith('.dat'))

for fname in files:
	DATA = np.loadtxt(path_in + fname, skiprows = skiprows, unpack = unpack)

	channel_time = 0
	channel_data = 1

	dt = np.mean(DATA[1:6, channel_time] - DATA[0:5, channel_time])

	# list of sections to cut signal into in the form of [start time, stop time]
	cut_sections = [[500, 4000]]
	n_sections = len(cut_sections)

	for section in cut_sections:
		start = int(section[0]/dt)
		stop = int(section[1]/dt)
		cut_data = DATA[start:stop]

		# save_fname = f'{save_path}{fname.split(".")[0]}_cut_{str(int(start/10))}_{str(int(stop/10))}.dat'
		save_fname = f'{save_path}{fname.split(".")[0]}_cut.dat'
		title = ' '.join(fname.split('.')[:-1])

		# plt.figure()
		# plt.plot(cut_data[:,0], cut_data[:,1])
		# plt.xlabel('Time [ms]')
		# plt.ylabel('LFP [mV]')
		# plt.title(title)
		# plt.show()

		np.savetxt(save_fname, cut_data)

