import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statistics_1 import check_stationarity
from aeml.eda.statistics import computer_granger_causality_matrix
import warnings

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib as mpl
from matplotlib.font_manager import FontProperties

plt.style.reload_library()
plt.style.use('grid')
plt.rcParams['font.family'] = 'sans-serif'

warnings.filterwarnings("ignore")

df = pd.read_pickle('/home/lsmo/Desktop/aeml_project/aeml/DataDynamo/RawData/New_campaigns/202403 SCOPE data set dynamic campaign.pkl')

df = df.dropna()

df['TI-1213'] = np.where(df['valve position'] == 1, df['TI-13'], df['TI-12'])

TARGETS_clean = ["AMP-4", "PZ-4"] 

# MEAS_COLUMNS = [

# 'PI-2',
# 'TI-2',
# 'F-3',
# 'PI-3',
# 'TI-3',
# 'CO2-3',
# 'O2-3',
# 'TI-33',
# 'TI-35',
# 'TI-4',
# 'CO2-4',
# 'F-11',
# 'TI-13',
# 'FI-211',
# 'TI-8',
# 'TI-5',
# 'TI-28',
# 'PI-30',
# 'F-30',
# 'P-38',
# 'T-36',
# 'F-19',
# 'T-19',
# 'PI-1',
# 'F-40',
# 'F-23',
# 'Level Desorber',
# 'TI-24',
# 'FI-25',
# 'TI-16',
# 'TI-151',
# 'TI-212',
# 'TI-241',
# 'valve position'
# ]

MEAS_COLUMNS = [

'PI-2',
'TI-2',
'F-3',
'PI-3',
'TI-3',
'CO2-3',
'O2-3',
'TI-32',
'TI-33',
'TI-34',
'TI-35',
'PI-4',
'TI-4',
'F-4',
'CO2-4',
# 'AMP-4',
# 'PZ-4',
# 'NH3-4',
# 'ACA',
'F-11',
'TI-12',
'TI-13',
'FI-20',
'FI-211',
'TI-211',
'TI-8',
'TI-9',
'TI-5',
'TI-7',
'TI-28',
'PI-28',
'PI-30',
'TI-30',
'F-30',
'F-38',
'P-38',
'F-36',
'T-36',
'Reboiler duty',
'F-19',
'T-19',
'PI-1',
'TI-1',
'TI-40',
'F-40',
'TI-39',
'F-23',
'TI-22',
'Level Desorber',
'Level Reboiler',
'TI-24',
'TI-25',
'FI-25',
'FI-16',
'TI-16',
'FI-151',
'TI-151',
'TI-152',
'TI-212',
'FI-241',
'TI-241',
'TI-242',
'valve position',
'T-15',
'dp-32',
'dp-33',
'dp-34',
'dp-35',
'dp-36',
'Level Adsorber',
'TI-071',
'TI-072',
'TI-070',
'dp-071',
'dp-072',
'dp-073',
'flow process water',
'temperature processwater inlet acid wash',
'pH process water',
'demin water flow',
'H2SO4 flow',
'level column',

]


def average_timeseries(df, skip):
    return df.rolling(window=skip, min_periods=1).mean()[::skip]

# stationary_test_result = check_stationarity(series=df['TI-3'])
# print(stationary_test_result['adf']['stationary'] == False)

# for column in df.columns:
#     stationary_test_result = check_stationarity(series=df[column])

#     # print(df.iloc[0,0:4])
#     # print(df.info())
#     if (stationary_test_result['adf']['stationary'] == True or stationary_test_result['kpss']['stationary'] == True):
#         print(column, end='\n')
#         print(stationary_test_result)
#         print('-' * 50)



result = pd.DataFrame()

for skip in [1, 24, 48, 96]:

    dfnew = average_timeseries(df, skip)
    
    causality_matrix = computer_granger_causality_matrix(dfnew, xs=MEAS_COLUMNS, ys=[TARGETS_clean[1]])
    
    if skip == 1:
        causality_matrix.columns = [f'Pz without averaging']
    else:
        causality_matrix.columns = [f'Average Pz over {skip} points']    

    # Append the results column-wise
    if result.empty:
        result = causality_matrix
    else:
        result = pd.concat([result, causality_matrix], axis=1)


# Get top 30 values
top_30_rows = result.loc[result.max(axis=1).nlargest(30).index]
top_30_rows_sorted = top_30_rows.loc[result.index.intersection(top_30_rows.index)]
filtered_result = top_30_rows_sorted


plt.imshow(filtered_result, cmap='coolwarm', aspect='auto')
plt.colorbar(label='Values')

# Setting axis labels and ticks
plt.xticks(np.arange(len(filtered_result.columns)), filtered_result.columns, rotation=90)
plt.yticks(np.arange(len(filtered_result.index)), filtered_result.index)

'''Plot Information and decoration'''
# =============================================================================
# Setting the font properties
fpLegend = '/home/lsmo/.local/share/fonts/calibri-regular.ttf'
fpLegendtitle = '/home/lsmo/.local/share/fonts/coolvetica rg.otf'
fpTitle = '/home/lsmo/.local/share/fonts/coolvetica rg.otf'
fpLabel = '/home/lsmo/.local/share/fonts/Philosopher-Bold.ttf'
fpTicks = '/home/lsmo/.local/share/fonts/Philosopher-Regular.ttf'

fLegend = FontProperties(fname=fpLegend)
fLegendtitle = FontProperties(fname=fpLegendtitle)
fTitle = FontProperties(fname=fpTitle)
fLabel = FontProperties(fname=fpLabel)
fTicks = FontProperties( fname=fpTicks)

# =============================================================================
# Add a frame around the plot area
plt.gca().spines['top'].set_visible(True)
plt.gca().spines['right'].set_visible(True)
plt.gca().spines['bottom'].set_visible(True)
plt.gca().spines['left'].set_visible(True)

# =============================================================================
# Adjust font size for tick labels
plt.xticks(fontproperties = fTicks)
plt.yticks(fontproperties = fTicks)

# =============================================================================
plt.legend(frameon=False)
plt.tight_layout()
plt.grid(False)

plt.show()

