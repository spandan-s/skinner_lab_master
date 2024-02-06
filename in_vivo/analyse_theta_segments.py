import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
sns.set_theme()

def ictal_timing(DATA, srs_in = "srs_analysis_results/SRS_timings.ods"):
    SRS = pd.read_excel(srs_in)
    time_to_SRS = np.zeros(len(DATA))

    DATA['t_event'] = (DATA['event_num'] - 1) + (DATA.t1 / 60)

    for i in range(len(DATA)):
        time_to_SRS[i] = np.min(DATA['t_event'][i] - SRS.loc[SRS['file_num'] == DATA['file_num'][i]]['SRS_timing'])

    DATA['time_to_SRS'] = pd.Series(time_to_SRS)
    return DATA

data_in = "srs_analysis_results/theta_segments.ods"

# DATA = pd.read_excel(f_in)

DATA_baseline = pd.read_excel(data_in, sheet_name='baseline')
DATA_baseline['rec_type'] = 'baseline'
DATA_SRS = pd.read_excel(data_in, sheet_name='SRS')
DATA_baseline['rec_type'] = 'SRS'


# print(DATA_SRS)

remove_cages = ['Cage C #9 no cut', 'cage E #13 no cut', 'cage E #15 right cut']
filt_DATA_baseline = DATA_baseline[~DATA_baseline['cage'].isin(remove_cages)]

#filter amplitude under 1e-5 out
filt_DATA_baseline = filt_DATA_baseline.loc[(filt_DATA_baseline['peak_1_amp'] > 1e-4) | (filt_DATA_baseline['peak_2_amp'] > 1e-4)]
filt_DATA_SRS = DATA_SRS.loc[(DATA_SRS['peak_1_amp'] > 5e-5) | (DATA_SRS['peak_2_amp'] > 5e-5)]

filt_DATA_SRS = ictal_timing(filt_DATA_SRS)

filt_DATA_SRS = filt_DATA_SRS.loc[(filt_DATA_SRS['time_to_SRS'] < 0) | (filt_DATA_SRS['time_to_SRS'] > 2)]

# with open("./srs_analysis_results/duration.txt", 'w') as f:
#     f.write("BASELINE\n")
#     f.write(str(filt_DATA_baseline['duration'].describe()))
#     f.write("\nSRS\n")
#     f.write(str(filt_DATA_SRS['duration'].describe()))

print(filt_DATA_baseline['peak_1_amp'].describe())
print(filt_DATA_SRS['peak_1_amp'].describe())

# filt_DATA = pd.concat([filt_DATA_baseline, filt_DATA_SRS])
# filt_DATA['SRS'] = filt_DATA['rec_type'] != 'pre-kindle'

# print(filt_DATA.loc[(filt_DATA['rec_type'] == 'baseline')]['peak_1_amp'].describe())
# print(filt_DATA.loc[(filt_DATA['rec_type'] == 'SRS')]['peak_1_amp'].describe())

# filt_DATA['rec_type'] = ['baseline' if filt_DATA['rec_type'] == 'pre-kindle' else 'SRS']

# ========================= FREQUENCY ===============================================
sns.histplot(data=filt_DATA_baseline, x='peak_1_freq', kde=True, stat='proportion', binwidth=1)
sns.histplot(data=filt_DATA_SRS, x='peak_1_freq', kde=True, stat='proportion', binwidth=1)
# plt.savefig("./srs_analysis_results/figures/peak_1_freq.png")

# ================= DURATION ================================================
# sns.histplot(data=filt_DATA_baseline, x='duration', kde=True, stat='proportion')
# sns.histplot(data=filt_DATA_SRS, x='duration', kde=True, stat='proportion')
# plt.savefig("./srs_analysis_results/figures/duration.png")

# ==================== AMPLITUDE =============================================
# fig, [ax1, ax2] = plt.subplots(1, 2)
# ax1 = sns.histplot(data=filt_DATA_baseline, x='peak_1_amp', kde=True, stat='proportion')
# ax2 = sns.histplot(data=filt_DATA_SRS, x='peak_1_amp', kde=True, stat='proportion')
# sns.displot(filt_DATA, x='peak_1_amp', col='SRS')
# # plt.tight_layout()
# plt.savefig("./srs_analysis_results/figures/peak_1_amp.png")

# sns.histplot(data=filt_DATA_baseline, x='peak_1_amp')
# plt.figure()
# sns.displot(filt_DATA, col='rec_type')

# sns.histplot(data=filt_DATA_baseline, x='peak_1_amp', stat='proportion')
# plt.figure()
# sns.histplot(data=filt_DATA_SRS, x='peak_1_amp', stat='proportion')

# plt.title("baseline")
# print(filt_DATA_baseline['peak_1_freq'].describe())
# print(filt_DATA_SRS['peak_1_freq'].describe())

# plt.figure()
# print(DATA_SRS['peak_1_freq'].hist(bins=20))
# plt.title("SRS")

plt.show()