import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
sns.set_theme()

f_in = "srs_analysis_results/theta_segments.ods"

DATA = pd.read_excel(f_in)

DATA_baseline = DATA.loc[DATA['rec_type'] == 'pre-kindle']
DATA_SRS = DATA.loc[DATA['rec_type'] != 'pre-kindle']

# print(DATA_SRS)

#filter amplitude under 1e-5 out
filt_DATA_baseline = DATA_baseline.loc[(DATA_baseline['peak_1_amp'] > 1e-5) | (DATA_baseline['peak_2_amp'] > 1e-5)]
filt_DATA_SRS = DATA_SRS.loc[(DATA_SRS['peak_1_amp'] > 1e-5) | (DATA_SRS['peak_2_amp'] > 1e-5)]

print(filt_DATA_baseline['peak_1_amp'].describe())
print(filt_DATA_SRS['peak_1_amp'].describe())


# print(DATA_baseline['peak_1_freq'].hist(bins=20))
# sns.kdeplot(data=filt_DATA_baseline, x='peak_1_freq')
# sns.kdeplot(data=filt_DATA_SRS, x='peak_1_freq')
#
# plt.figure()
# sns.kdeplot(data=filt_DATA_baseline, x='duration')
# sns.kdeplot(data=filt_DATA_SRS, x='duration')

sns.histplot(data=filt_DATA_baseline, x='peak_1_amp')
plt.figure()
sns.histplot(data=filt_DATA_SRS, x='peak_1_amp')
# sns.kdeplot(data=filt_DATA_SRS, x='peak_1_amp')

# plt.title("baseline")
# print(filt_DATA_baseline['peak_1_freq'].describe())
# print(filt_DATA_SRS['peak_1_freq'].describe())

# plt.figure()
# print(DATA_SRS['peak_1_freq'].hist(bins=20))
# plt.title("SRS")



plt.show()