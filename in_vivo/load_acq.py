import os

import numpy as np
import bioread
import re


def save_channels(acq_file, ch1, ch2, sdir, extension='.dat'):
    '''

    :param acq_file: acq file read using bioread
    :param ch1: list of channels for first subject
    :param ch2: list of channels for second subject
    :param sdir: path to directory results are saved in
    :param extension: extension for saved files, default = .dat
    :return: None
    '''

    os.chdir(sdir)
    try:
        os.mkdir("Subject_1")
        os.mkdir("Subject_2")
    except FileExistsError:
        pass

    if len(ch1) > 0:
        for channel in ch1:
            time = acq_file.channels[channel].time_index
            signal = acq_file.channels[channel].data

            save_data = np.zeros((len(time), 2))
            save_data[:, 0] = time
            save_data[:, 1] = signal
            save_data = np.float32(save_data)

            np.savetxt(f"./Subject_1/channel_{channel + 1}{extension}", save_data)

    if len(ch2) > 0:
        for channel in ch2:
            time = acq_file.channels[channel].time_index
            signal = acq_file.channels[channel].data

            save_data = np.zeros((len(time), 2))
            save_data[:, 0] = time
            save_data[:, 1] = signal
            save_data = np.float32(save_data)

            np.savetxt(f"./Subject_2/channel_{channel + 1}{extension}", save_data)

path_in = '/media/spandans/Transcend/TBI/first group/'
path_out = '/home/spandans/Documents/'

files = sorted(file for file in os.listdir(path_in) if file.endswith('.acq'))

k = 69

filename = files[k]
save_dir = filename

REGEX_REPLACEMENTS = [
    # (r"[-T_:\d]{10,}", ""),
    (r"\.[^\.]+$", ""),
    (r"\s", "_")
]

for old, new in REGEX_REPLACEMENTS:
    save_dir = re.sub(old, new, save_dir)

if bool(re.search(r"top", save_dir)):
    channels1, channels2 = [0, 1, 2, 3], [8, 9, 10, 11]
elif bool(re.search(r"bot", save_dir)):
    channels1, channels2 = [39, 40, 41, 42], [47, 48, 49, 50]
else:
    channels1, channels2 = [], []

# loop through files in directory
for k in range(69, 70):

    # read file
    print(f"Reading file: {files[k]}")
    data = bioread.read_file(f'{path_in}{files[k]}')

    # create save directory
    save_dir = files[k]
    for old, new in REGEX_REPLACEMENTS:
        save_dir = re.sub(old, new, save_dir)

    try:
        os.mkdir(f"{path_in}{save_dir}")
    except FileExistsError:
        pass

    # now redundant
    # # check if top or bottom amplifier
    # if bool(re.search(r"top", save_dir)):
    #     channels1, channels2 = [0, 1, 2, 3], [8, 9, 10, 11]
    # elif bool(re.search(r"bot", save_dir)):
    #     channels1, channels2 = [39, 40, 41, 42], [47, 48, 49, 50]
    # else:
    #     channels1, channels2 = [], []

    channels1, channels2 = [0, 1, 2, 3], [8, 9, 10, 11]

    # save desired channels to .dat files
    save_channels(data, channels1, channels2, f"{path_in}{save_dir}")
