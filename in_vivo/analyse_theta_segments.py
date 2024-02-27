import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu

sns.set_theme()

# def ictal_timing(DATA, srs_in = "srs_analysis_results/SRS_timings.ods"):
#     SRS = pd.read_excel(srs_in)
#     time_to_SRS = np.zeros(len(DATA))
#
#     DATA['t_event'] = (DATA['event_num'] - 1) + (DATA.t1 / 60)
#
#     for i in range(len(DATA)):
#         time_to_SRS[i] = np.min(DATA['t_event'][i] - SRS.loc[SRS['file_num'] == DATA['file_num'][i]]['SRS_timing'])
#
#     DATA['time_to_SRS'] = pd.Series(time_to_SRS)
#     return DATA
def ictal_timing(DATA, srs_in="srs_analysis_results/SRS_timings.ods"):
    SRS = pd.read_excel(srs_in)
    time_to_SRS = np.zeros(len(DATA))
    DATA['t_event'] = (DATA['event_num'] - 1) + (DATA.t1 / 60)
    for i in range(len(DATA)):
        var1 = DATA['t_event'][i] - SRS.loc[SRS['file_num'] == DATA['file_num'][i]]['SRS_timing']
        if len(var1.values) > 0:
            var2 = np.argmin(np.abs(var1))
            time_to_SRS[i] = var1.values[var2]
        else:
            time_to_SRS[i] = np.nan
    DATA['time_to_SRS'] = pd.Series(time_to_SRS)
    return DATA

data_in = "srs_analysis_results/theta_segments.ods"

# DATA = pd.read_excel(f_in)

DATA_baseline = pd.read_excel(data_in, sheet_name='baseline')
DATA_baseline['rec_type'] = 'baseline'
DATA_SRS = pd.read_excel("srs_analysis_results/Theta_dominants_in_SRS.xlsx")
DATA_SRS['rec_type'] = 'SRS'


# print(DATA_SRS)

remove_cages = ['Cage C #9 no cut', 'cage E #13 no cut', 'cage E #15 right cut']
filt_DATA_baseline = DATA_baseline[~DATA_baseline['cage'].isin(remove_cages)]

#filter amplitude under 1e-5 out
filt_DATA_baseline = filt_DATA_baseline.loc[(filt_DATA_baseline['peak_1_amp'] > 1e-4) | (filt_DATA_baseline['peak_2_amp'] > 1e-4)]

# print(filt_DATA_baseline.head())

cage_bl = 'Cage C #7 right cut'
cage_SRS = "Cage C #7 right cut"
cage_filt_baseline = filt_DATA_baseline.loc[filt_DATA_baseline['cage'] == cage_bl]
cage_filt_SRS = DATA_SRS.loc[DATA_SRS['Cage'] == cage_SRS]

if cage_filt_SRS.count()['Start Time (min)'] > 0:
    fig, ax = plt.subplots(2, 2)

    g1 = np.array(cage_filt_SRS["Peak frequency 1"].dropna())
    g2 = np.array(cage_filt_baseline["peak_1_freq"].dropna())
    print(mannwhitneyu(g1, g2))
    sns.histplot(data=cage_filt_baseline, x='peak_1_freq', ax=ax[0, 0], kde=True, stat='proportion', binwidth=0.5)
    sns.histplot(data=cage_filt_SRS, x='Peak frequency 1', ax=ax[0, 0], kde=True, stat='proportion', binwidth=0.5)

    g1 = np.array(cage_filt_SRS["Peak Amplitude 1"].dropna())
    g2 = np.array(cage_filt_baseline["peak_1_amp"].dropna())
    print(mannwhitneyu(g1, g2))
    sns.histplot(data=cage_filt_baseline, x='peak_1_amp', ax=ax[0, 1], kde=True, stat='proportion')
    sns.histplot(data=cage_filt_SRS, x='Peak Amplitude 1', ax=ax[0, 1], kde=True, stat='proportion')

    g1 = np.array(cage_filt_SRS["Duration (s)"].dropna())
    g2 = np.array(cage_filt_baseline["duration"].dropna())
    print(mannwhitneyu(g1, g2))
    sns.histplot(data=cage_filt_baseline, x='duration', ax=ax[1, 0], kde=True, stat='proportion', binwidth=0.5)
    sns.histplot(data=cage_filt_SRS, x='Duration (s)', ax=ax[1, 0], kde=True, stat='proportion', binwidth=0.5)

    g1 = np.array(cage_filt_SRS["Bandwidth 1"].dropna())
    g2 = np.array(cage_filt_baseline["peak_1_bw"].dropna())
    print(mannwhitneyu(g1, g2))
    sns.histplot(data=cage_filt_baseline, x='peak_1_bw', ax=ax[1, 1], kde=True, stat='proportion', binwidth=0.5)
    sns.histplot(data=cage_filt_SRS, x='Bandwidth 1', ax=ax[1, 1], kde=True, stat='proportion', binwidth=0.5)
    fig.suptitle(cage_bl)

    plt.tight_layout()


# filt_DATA_SRS = DATA_SRS.loc[(DATA_SRS['peak_1_amp'] > 5e-5) | (DATA_SRS['peak_2_amp'] > 5e-5)]

# filt_DATA_SRS = ictal_timing(filt_DATA_SRS)
#
# filt_DATA_SRS = filt_DATA_SRS.loc[(filt_DATA_SRS['time_to_SRS'] < 0) | (filt_DATA_SRS['time_to_SRS'] > 2)]
# filt_DATA_SRS.to_excel("srs_analysis_results/SRS_filtered_v2.xlsx")

# with open("./srs_analysis_results/duration.txt", 'w') as f:
#     f.write("BASELINE\n")
#     f.write(str(filt_DATA_baseline['duration'].describe()))
#     f.write("\nSRS\n")
#     f.write(str(filt_DATA_SRS['duration'].describe()))

# filt_DATA_SRS = pd.read_excel("srs_analysis_results/SRS_filtered.xlsx")
# filt_DATA_SRS = filt_DATA_SRS.loc[filt_DATA_SRS['for_analysis']]

# print(filt_DATA_baseline['peak_1_amp'].describe())
# print(filt_DATA_SRS['peak_1_amp'].describe())

# filt_DATA = pd.concat([filt_DATA_baseline, filt_DATA_SRS])
# filt_DATA['SRS'] = filt_DATA['rec_type'] != 'pre-kindle'

# print(filt_DATA.loc[(filt_DATA['rec_type'] == 'baseline')]['peak_1_amp'].describe())
# print(filt_DATA.loc[(filt_DATA['rec_type'] == 'SRS')]['peak_1_amp'].describe())

# filt_DATA['rec_type'] = ['baseline' if filt_DATA['rec_type'] == 'pre-kindle' else 'SRS']

# ========================= FREQUENCY ===============================================
# sns.histplot(data=filt_DATA_baseline, x='peak_1_freq', kde=True, stat='proportion', binwidth=0.5)
# sns.histplot(data=filt_DATA_SRS, x='peak_1_freq', kde=True, stat='proportion', binwidth=0.5)
# plt.savefig("./srs_analysis_results/figures/peak_1_freq.png")

# ================= DURATION ================================================
# sns.histplot(data=filt_DATA_baseline, x='duration', kde=True, stat='proportion')
# sns.histplot(data=filt_DATA_SRS, x='duration', kde=True, stat='proportion')
# plt.savefig("./srs_analysis_results/figures/duration.png")

# ==================== AMPLITUDE =============================================
# sns.histplot(data=filt_DATA_baseline, x='peak_1_amp')
# plt.figure()
# sns.displot(filt_DATA, col='rec_type')

# sns.histplot(data=filt_DATA_baseline, x='peak_1_amp', stat='proportion')
# plt.figure()
# sns.histplot(data=filt_DATA_SRS, x='peak_1_amp', stat='proportion')
# ================================== PAIRPLOTS =============================================
# sns.pairplot(filt_DATA_baseline, vars=['peak_1_amp', 'duration', 'peak_1_freq'], kind='hist')
# plt.tight_layout()
# plt.savefig("srs_analysis_results/figures/baseline_pairplot.png", dpi=300)

# sns.pairplot(filt_DATA_SRS, vars=['peak_1_amp', 'duration', 'peak_1_freq'], kind='scatter', diag_kind='hist')
# plt.tight_layout()
# plt.savefig("srs_analysis_results/figures/SRS_pairplot.png", dpi=300)
# ============================== ICTAL TIMING ====================================================
# plt.scatter(x=filt_DATA_SRS['time_to_SRS'], y=filt_DATA_SRS['peak_1_amp'], color='C0')
# plt.scatter(x=filt_DATA_SRS['time_to_SRS'], y=filt_DATA_SRS['peak_2_amp'], color='C0')
# plt.xlabel("time to SRS [min]")
# plt.ylabel("theta power [$mV^2/Hz$]")
# plt.tight_layout()
# plt.savefig('srs_analysis_results/figures/timing_vs_amp.png')

# plt.scatter(x=filt_DATA_SRS['time_to_SRS'], y=filt_DATA_SRS['peak_1_freq'], color='C0')
# plt.scatter(x=filt_DATA_SRS['time_to_SRS'], y=filt_DATA_SRS['peak_2_freq'], color='C0')
# plt.xlabel("time to SRS [min]")
# plt.ylabel("theta frequency [Hz]")
# plt.tight_layout()
# plt.savefig('srs_analysis_results/figures/timing_vs_freq.png')
#
# plt.scatter(x=filt_DATA_SRS['time_to_SRS'], y=filt_DATA_SRS['duration'], color='C0')
# plt.xlabel("time to SRS [min]")
# plt.ylabel("duration [s]")
# plt.tight_layout()
# plt.savefig('srs_analysis_results/figures/timing_vs_duration.png')

plt.show()