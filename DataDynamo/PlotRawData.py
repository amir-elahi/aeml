import pandas as pd 
import numpy as np
from darts import TimeSeries

from darts.dataprocessing.transformers import Scaler

from aeml.utils.Plot import *

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

plt.style.reload_library()
plt.style.use('grid')
plt.rcParams['font.family'] = 'sans-serif'


#! Script to plot the raw data with different averaging (each panel of Figure A3)
# =============================================================================
# Load the data
df = pd.read_pickle('./DataDynamo/RawData/New_campaigns/202403 SCOPE data set dynamic campaign.pkl')

df = df.dropna()

df['TI-1213'] = np.where(df['valve position'] == 1, df['TI-13'], df['TI-12'])

TARGETS_clean = ['AMP-4', 'PZ-4'] 

MEAS_COLUMNS = [ 'T-19', 'TI-3', 'F-19','F-11', 'TI-1213','TI-35']

Target = MEAS_COLUMNS[3]
ylabel = r'Flow rate $[\mathrm{kg/h}]$'
title = 'FI-11'
# =============================================================================
# Convert to TimeSeries and scale the data
plt.figure(figsize=(4.9, 2.1))

def average_timeseries(df, skip):
    return df.rolling(window=skip, min_periods=1).mean()[::skip]


for skip in [1, 24, 48, 96]:

    y = TimeSeries.from_dataframe(df, value_cols=TARGETS_clean, time_col='Date')
    x = TimeSeries.from_dataframe(df, value_cols=MEAS_COLUMNS, time_col='Date')

    transformer = Scaler()
    x = transformer.fit_transform(x)

    y_transformer = Scaler()
    y = y_transformer.fit_transform(y)

    scal = y_transformer.transform(y)

    # Convert TimeSeries to DataFrame for processing
    df_y = y.pd_dataframe()
    df_x = x.pd_dataframe()

# Apply averaging function
    temp_y = average_timeseries(df_y, skip)
    temp_y.columns = y.pd_dataframe().columns
    ts_y = TimeSeries.from_dataframe(temp_y)
    df_y= temp_y

    y = ts_y

    temp_x = average_timeseries(df_x, skip)
    temp_x.columns = x.pd_dataframe().columns
    ts_x = TimeSeries.from_dataframe(temp_x)
    df_x = temp_x

    x = ts_x

    Ts3 = y

    #* Inverse Transforming the series
    y_actual = y_transformer.inverse_transform(Ts3)
    x_actual = transformer.inverse_transform(x)

    plt.plot(x_actual.pd_dataframe()[Target].index, x_actual.pd_dataframe()[Target].values, lw=0.8)

# plt.plot(Ts3.pd_dataframe()[Target].index, Ts3.pd_dataframe()[Target].values, lw=0.8, color = 'red')

'''Plotting'''
'''Plot Information and decoration'''
# =============================================================================
# Setting the font properties
fpLegend = './DataDynamo/Fonts/calibri-regular.ttf'
fpLegendtitle = './DataDynamo/Fonts/coolvetica rg.otf'
fpTitle = './DataDynamo/Fonts/coolvetica rg.otf'
fpLabel = './DataDynamo/Fonts/Philosopher-Bold.ttf'
fpTicks = './DataDynamo/Fonts/Philosopher-Regular.ttf'

fLegend = FontProperties(fname=fpLegend)
fLegendtitle = FontProperties(fname=fpLegendtitle)
fTitle = FontProperties(fname=fpTitle)
fLabel = FontProperties(fname=fpLabel)
fTicks = FontProperties( fname=fpTicks)

# =============================================================================
# Add labels and title and ticks
plt.ylabel(ylabel, fontproperties = fLabel)
plt.xlabel('Time', fontproperties = fLabel)

if title is not None:
    plt.title(title, fontproperties = fTitle)

for label in (plt.gca().get_xticklabels() + plt.gca().get_yticklabels()):
    label.set_fontproperties(fTicks)

# =============================================================================
# Add a frame around the plot area
plt.gca().spines['top'].set_visible(True)
plt.gca().spines['right'].set_visible(True)
plt.gca().spines['bottom'].set_visible(True)
plt.gca().spines['left'].set_visible(True)

# =============================================================================
# Add a legend
# legend = plt.legend(prop = fLegend)

# =============================================================================
# Set the date format
date_format = mdates.DateFormatter('%b-%d')
plt.gca().xaxis.set_major_formatter(date_format)

# =============================================================================
# Adjust font size for tick labels
plt.xticks(rotation='vertical', fontproperties = fTicks)
plt.yticks(fontproperties = fTicks)
# plt.ylim(0,200)
# =============================================================================
plt.legend(frameon=False)
plt.tight_layout()
plt.show()
