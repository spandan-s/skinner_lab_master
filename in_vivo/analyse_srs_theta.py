import numpy as np
import pandas as pd
import re

cagedirs = [
    "cage B #5 left cut",
    "Cage C #7 right cut",
    "cage D #12 no cut",
    "cage D 11 right cut"
]


def read_file(fname):
    entry_list = []
    with open(fname, "r") as f:
        file_txt = f.read()
    entry_list = file_txt.split("Detected Segments:\n")

    return entry_list

def extract_data(entry):
    nums, *peaks = entry.split("\n\n")

    #deal with the numbers --> remove empty strings and convert to floats
    segments_list = nums.split(" ")
    segments_list[:] = [float(x) for x in segments_list if x]

    peaks[:] = [x for x in peaks if x]
    peakInfo = [{} for _ in range(len(peaks))]
    # deal with the event data
    for idx, peak in enumerate(peaks):
        peakInfo[idx]["event_num"] = int(re.findall(r"Event number (\d+)", peak)[0])
        peakInfo[idx]["peak_freq"] = float(re.findall(r"frequency: (.*)", peak)[0])
        peakInfo[idx]["bandwidth"] = float(re.findall(r"Bandwidth: (.*)", peak)[0])
        peakInfo[idx]["peak_amp"] = float(re.findall(r"Amplitude: (.*)", peak)[0])
        peakInfo[idx]["segments"] = segments_list
    return peakInfo

cage_dir = "cage B #5 left cut"
rec_type = "pre-kindle"
file_num = "2017_11_08_0007"

entries = read_file(f"srs_analysis_results/{cage_dir}/{rec_type}/{file_num}.txt")
entries[:] = [x for x in entries if x]
# print(entries[1])

DATA = []
for entry in entries:
    data = extract_data(entry)

    for idx in data:
        idx["cage"] = cage_dir
        idx["rec_type"] = rec_type
        idx["file_num"] = file_num

    DATA.append(data)

print(DATA)