import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
sns.set_theme()

data_in = "srs_analysis_results/theta_segments.ods"
srs_in = "srs_analysis_results/SRS_timings.ods"

DATA = pd.read_excel(data_in, sheet_name='SRS')
# SRS = pd.read_excel(srs_in)

def ictal_timing(DATA, srs_in = "srs_analysis_results/SRS_timings.ods"):
    SRS = pd.read_excel(srs_in)
    time_to_SRS = np.zeros(len(DATA))

    DATA['t_event'] = (DATA['event_num'] - 1) + (DATA.t1 / 60)

    for i in range(len(DATA)):
        time_to_SRS[i] = np.min(DATA['t_event'][i] - SRS.loc[SRS['file_num'] == DATA['file_num'][i]]['SRS_timing'])

    DATA['time_to_SRS'] = pd.Series(time_to_SRS)
    return DATA

# time_to_SRS = np.zeros(len(DATA))
#
# DATA['t_event'] = (DATA['event_num'] - 1) + (DATA.t1 / 60)
#
# for i in range(len(DATA)):
#     time_to_SRS[i] = np.min(DATA['t_event'][i] - SRS.loc[SRS['file_num'] == DATA['file_num'][i]]['SRS_timing'])
#
# DATA['time_to_SRS'] = pd.Series(time_to_SRS)

DATA = ictal_timing(DATA)

DATA.to_csv("./srs_analysis_results/SRS_events.csv")
filt_DATA = DATA.loc[(DATA['time_to_SRS'] < 0) | (DATA['time_to_SRS'] > 2)]

# print(DATA.head())
f, ax = plt.subplots()
sns.scatterplot(data=filt_DATA, x="time_to_SRS", y="peak_1_freq", ax=ax, hue='cage')
sns.scatterplot(data=filt_DATA, x="time_to_SRS", y="peak_2_freq", ax=ax, hue='cage')
# plt.savefig("./srs_analysis_results/figures/ictal_vs_freq.png")

# f, ax = plt.subplots()
# sns.scatterplot(data=filt_DATA, x="time_to_SRS", y="peak_1_amp", ax=ax, color='C0')
# sns.scatterplot(data=filt_DATA, x="time_to_SRS", y="peak_2_amp", ax=ax, color='C0')
# plt.savefig("./srs_analysis_results/figures/ictal_vs_amp.png")
#
# f, ax = plt.subplots()
# sns.scatterplot(data=filt_DATA, x="time_to_SRS", y="duration", ax=ax, color='C0')
# plt.savefig("./srs_analysis_results/figures/ictal_vs_duration.png")

# ax.set_xlim(left=5)

plt.show()