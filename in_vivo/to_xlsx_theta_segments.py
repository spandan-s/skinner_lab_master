import os

import numpy as np
import pandas as pd
import re

cagedirs = [
    "cage B #5 left cut",
    "Cage C #7 right cut",
    "cage D #12 no cut",
    "cage D 11 right cut"
]
df_cols = [
    "cage", "rec_type", "file_num",
    "event_num", "t1", "t2", "duration",
    "peak_1_freq", "peak_1_bw", "peak_1_amp",
    "peak_2_freq", "peak_2_bw", "peak_2_amp",
]

def process_file(cage_dir, rec_type, file):
    file_num = file.split('.txt')[0]
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
    return DATA


def read_file(fname):
    entry_list = []
    with open(fname, "r") as f:
        file_txt = f.read()
    entry_list = file_txt.split("Detected Segments:\n")

    return entry_list


def get_segments(X, m=1):
    segments = X.split("\n")
    segments[:] = [x for x in segments if x]
    nums = np.zeros((len(segments), 4))
    for i, s in enumerate(segments):
        nums[i] = [m * float(x1) for x1 in s.split(' ') if x1]

    return nums


def extract_data(entry):
    X = entry.split("\n\n")
    peak_start = 1

    # deal with the numbers --> remove empty strings and convert to floats
    if "*" in X[0]:
        segments = get_segments(X[1], 1e3)
        peak_start = 2
    else:
        segments = get_segments(X[0])

    peaks = X[peak_start:]
    peaks[:] = [x for x in peaks if x.startswith('Event')]
    peakInfo = [{} for _ in range(len(peaks))]
    # deal with the event data
    for idx, peak in enumerate(peaks):
        peakInfo[idx]["event_num"] = int(re.findall(r"Event number (\d+)", peak)[0])
        peakInfo[idx]["peak_freq"] = float(re.findall(r"frequency: (.*)", peak)[0])
        peakInfo[idx]["bandwidth"] = float(re.findall(r"Bandwidth: (.*)", peak)[0])
        peakInfo[idx]["peak_amp"] = float(re.findall(r"Amplitude: (.*)", peak)[0])

    segment_info = [{} for _ in range(len(segments))]
    for idx, segment in enumerate(segments):
        segment_info[idx]["event_num"] = peakInfo[0]["event_num"]
        segment_info[idx]["t1"] = segment[0]/10
        segment_info[idx]["t2"] = segment[1]/10
        segment_info[idx]["duration"] = segment[2]/10
        segment_info[idx]["peak_1_freq"] = peakInfo[0]["peak_freq"]
        segment_info[idx]["peak_1_bw"] = peakInfo[0]["bandwidth"]
        segment_info[idx]["peak_1_amp"] = peakInfo[0]["peak_amp"]
        if len(peaks) > 1:
            segment_info[idx]["peak_2_freq"] = peakInfo[1]["peak_freq"]
            segment_info[idx]["peak_2_bw"] = peakInfo[1]["bandwidth"]
            segment_info[idx]["peak_2_amp"] = peakInfo[1]["peak_amp"]

    return segment_info

df = pd.DataFrame(columns=df_cols)

cage_dir = "cage F #19 right 2 cuts"
rec_type = ["pre-kindle", 'post 100 stim']

for rt in rec_type:
    files = sorted(file for file in os.listdir(f"srs_analysis_results/{cage_dir}/{rt}") if file.endswith('.txt'))
    for file in files:
        DATA = process_file(cage_dir, rt, file)

        for data in DATA:
            df = pd.concat([df, pd.DataFrame(data)], ignore_index=True)

df.to_excel(f"srs_analysis_results/{cage_dir}.xlsx")
