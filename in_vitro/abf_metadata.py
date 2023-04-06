#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 12:32:14 2022

@author: spandans
"""

import pyabf
import os
import matplotlib.pyplot as plt
from time import time

def get_abf_metadata(fn, write_to_file=True, Plot=False):
    save_fn = f'metadata/{fn.split(".")[0]}_metadata.txt'

    start_time = time()
    abf = pyabf.ABF(fn)

    header = abf.headerText
    soln = '(in mM): NaCl 125, KCl 3.5, NaH2PO4 1.25, CaCl2 1.5, MgSO4 1.5, NaHCO3 26 \
    and glucose 10. Aerated with 5%CO2-95%O2 (pH 7.35-7.4).'

    lines = {
        'RECORDING TYPE': 'monopolar',
        'SOLUTION': soln,
        'SAMPLE RATE': str(abf.sampleRate),
        'FILE COMMENT': abf.abfFileComment,
        'X LABEL': abf.sweepLabelX,
        'Y LABEL': abf.sweepLabelY,
        'X UNITS': abf.sweepUnitsX,
        'Y UNITS': abf.sweepUnitsY
    }

    labels = []
    for key, value in lines.items():
        labels.append(key)

    if write_to_file:
        with open(save_fn, 'w') as f:
            print('=========================================')
            print('Writing to file:', save_fn)

            f.write('### IMPORTANT STUFF ###\n')

            for label in labels:
                f.write(label + ': ')
                f.write(lines[label])
                f.write('\n')

            f.write('\n### FULL HEADER ###')
            f.write(header)
    print(f'Completed in {(time() - start_time):.2f} seconds. Metadata file saved in metadata folder.')
    print('=========================================\n')

    sweepX = abf.sweepX
    sweepY = abf.sweepY
    sweepC = abf.sweepC

    if Plot:
        plt.figure(figsize=[12, 9])
        plt.plot(sweepX, sweepY)
        plt.xlabel(lines['X LABEL'])
        plt.ylabel(lines['Y LABEL'])
        plt.title(fn.split('.')[0])
        plt.show()
        # plt.xlim(175, 195)

dir = '/home/spandans/Documents/NFRF_testdata_Aylin_Liang//noise_test_march_20_2022/'
os.chdir(dir)

try:
    os.mkdir('metadata')
except FileExistsError:
    pass

for fn in sorted(i for i in os.listdir(dir) if i.endswith('.abf')):
    get_abf_metadata(fn)

# fn = '221004_5901cko_0026.abf'
# save_fn = fn.split('.')[0] + '_metadata.txt'
#
# start_time = time()
# abf = pyabf.ABF(fn)
#
# header = abf.headerText
# soln = '(in mM): NaCl 125, KCl 3.5, NaH2PO4 1.25, CaCl2 1.5, MgSO4 1.5, NaHCO3 26 \
# and glucose 10. Aerated with 5%CO2-95%O2 (pH 7.35-7.4).'
#
# lines = {
#     'RECORDING TYPE': 'monopolar',
#     'SOLUTION': soln,
#     'SAMPLE RATE': str(abf.sampleRate),
#     'FILE COMMENT': abf.abfFileComment,
#     'X LABEL': abf.sweepLabelX,
#     'Y LABEL': abf.sweepLabelY,
#     'X UNITS': abf.sweepUnitsX,
#     'Y UNITS': abf.sweepUnitsY
#     }
#
# labels = []
# for key, value in lines.items():
#     labels.append(key)
#
# write_to_file = True
# if write_to_file:
#     with open(save_fn, 'w') as f:
#         print('=========================================')
#         print('Writing to file:', save_fn)
#
#         f.write('### IMPORTANT STUFF ###\n')
#
#         for label in labels:
#             f.write(label + ': ')
#             f.write(lines[label])
#             f.write('\n')
#
#         f.write('\n### FULL HEADER ###')
#         f.write(header)
# print(f'Completed in {(time() - start_time):.2f} seconds')
# print('=========================================\n')
#
# sweepX = abf.sweepX
# sweepY = abf.sweepY
# sweepC = abf.sweepC
#
# Plot = True
# if Plot:
#     plt.figure(figsize=[12, 9])
#     plt.plot(sweepX, sweepY)
#     plt.xlabel(lines['X LABEL'])
#     plt.ylabel(lines['Y LABEL'])
#     plt.title(fn.split('.')[0])
#     plt.show()
#     # plt.xlim(175, 195)
