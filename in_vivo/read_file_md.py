from datetime import datetime
import os
import numpy as np
import re
import pandas as pd

def extract_date_time(fn):
    for match in re.findall(r"[-T_:\d]{10,}", fn):
        date, time = match.split('T')
        time = re.sub("_", ":", time)
        date = datetime.strptime(date, "%Y-%m-%d")
        time = datetime.strptime(time, '%H:%M:%S')
        return date, time


def extract_amp(fn):
    if "top" in fn.lower():
        return "TOP"
    elif "bottom" in fn.lower():
        return "BOTTOM"
    else:
        return "Amp not specified"

def extract_ids(fn):
     ids_in_str = []
     for idx in data.id:
          if idx in fn:
               ids_in_str.append(idx)

     return ids_in_str

def create_entry(fn):
    date, time = extract_date_time(fn)
    amp = extract_amp(fn)
    ids = extract_ids(fn)

    entries = pd.DataFrame([])

    for id in ids:
        entry = {
            "fn": fn,
            "mouse_id": id,
            "amp": amp,
            "tbi": data.loc[data.id == id, "tbi"].item(),
            "proc_date": data.loc[data.id == id, "proc_date"].item(),
            "rec_date": date,
            "rec_time": time,
            "l_chan1": data.loc[data.id == id, "L_chan1"].item(),
            "l_chan1_type": data.loc[data.id == id, "L_chan1_type"].item(),
            "l_chan2": data.loc[data.id == id, "L_chan2"].item(),
            "l_chan2_type": data.loc[data.id == id, "L_chan2_type"].item(),
            "r_chan1": data.loc[data.id == id, "R_chan1"].item(),
            "r_chan1_type": data.loc[data.id == id, "R_chan1_type"].item(),
            "r_chan2": data.loc[data.id == id, "R_chan2"].item(),
            "r_chan2_type": data.loc[data.id == id, "R_chan2_type"].item(),
        }
        entry = pd.DataFrame([entry])
        entries = pd.concat([entries, entry], ignore_index=True)
    return entries


data = pd.read_excel("./aylin_tbi_spreadsheet.xlsx")

file_dir = "/media/spandans/One Touch/"
sub_dirs: list[str] = [file for file in os.listdir(file_dir) if file[0].isdigit()]
file_list: dict = {}
for sub_dir in sub_dirs:
    file_list[sub_dir] = os.listdir(f"{file_dir}/{sub_dir}")

sub_dirs.remove('13-20')
# sub_dir_num = 10


output = pd.DataFrame([])
for sub_dir_num, sub_dir in enumerate(sub_dirs):
# sub_dir = sub_dirs[sub_dir_num]
    file_nums = np.arange(len(file_list[sub_dirs[sub_dir_num]]))
    for file_num in file_nums:
        file = file_list[sub_dir][file_num]
        output = pd.concat([output, create_entry(file)], ignore_index=True)

output.to_excel(f"{file_dir}/spreadsheets/master.xlsx")




