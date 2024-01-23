import numpy as np

def create_entry(arr1, arr2):
    pyr = {
        'mean_amp': arr1[0],
        'median_amp': arr1[1],
        'std_amp': arr1[2],
        'mean_isi': arr1[0],
        'median_isi': arr1[1],
        'std_amp': arr1[2],
    }