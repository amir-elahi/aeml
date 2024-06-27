import pandas as pd 
import numpy as np
from darts import TimeSeries
from aeml.models.gbdt.settings import *


from darts.dataprocessing.transformers import Scaler
import pickle, subprocess

import matplotlib.pyplot as plt

plt.style.reload_library()
# plt.style.use('grid')

from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'

from chronos import ChronosPipeline
import torch
from transformers import set_seed
from datetime import datetime
import os

from aeml.utils.Plot import *


#! This is the script to train the Chronos model and make predictions.
#! It can either save to a .pkl file and plot with PlotHistoricalForecast.py or you can plot here 
#! (Figure 5a, Figure 6a, Figure A5 )

"""
# =============================================================================
TL;DR:
    This script is used to train the Chronos model on the data and make predictions.

Description:
    This Python script does the histrical forecast of the data using Chronos model.
    It can average and skip the data points to reduce the number of points to be fed to the model.

Dependencies:
    Refer to the *.yml file based on your operating system.

Version History:
    <Date>, <Author>, <Description of Changes>
    6  April 2022 AE Add the historical forecast and averaging of the data using Chronos model.
    8  April 2022 AE Configure the plotting and saving of the output. Adding flags to make it easier to use.
    14 April 2022 AE Minor changes in plotting.
    27 June  2022 AE Ready for the first release.

# =============================================================================
"""

df = pd.read_pickle('./DataDynamo/RawData/New_campaigns/202403 SCOPE data set dynamic campaign.pkl')

df = df.dropna()

df['TI-1213'] = np.where(df['valve position'] == 1, df['TI-13'], df['TI-12'])

TARGETS_clean = ['AMP-4', 'PZ-4'] 

MEAS_COLUMNS = [ 'T-19', 'TI-3', 'F-19','F-11', 'TI-1213','TI-35']


y = TimeSeries.from_dataframe(df, value_cols=TARGETS_clean, time_col='Date')
x = TimeSeries.from_dataframe(df, value_cols=MEAS_COLUMNS, time_col='Date')

transformer = Scaler()
x = transformer.fit_transform(x)

y_transformer = Scaler()
y = y_transformer.fit_transform(y)

#* Set the seed for reproducibility
set_seed(42)

#! This part is where we define the torch tensor to feed it to the model. Note that it is not scaled
prediction_length = 64
startPoint = 0
endPoint = len(y)
skip = 158
savePickles = True # True will save pickles and don't plot, False will plot and don't save pickles
historic = True # True will perform historical forecast, False will perform the normal forecast

#* Load the model (You can use cuda if you have a GPU)
pipeline = ChronosPipeline.from_pretrained(
    "amazon/chronos-t5-tiny",
    device_map="cpu",
    torch_dtype=torch.bfloat16,
)

#* Extract the time series
# =============================================================================
# The normal time series
Ts1 = y[startPoint:endPoint]
# =============================================================================
# Skip is the number of points to skip and it selects the last point of every skip points
Ts2 = y[startPoint:endPoint:skip]
# =============================================================================
# This averages skip point in the time series. example: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10: skip=2, then 1, 2.5, 4.5, 6.5, 8.5
def average_timeseries(df, skip):
    return df.rolling(window=skip, min_periods=1).mean()[::skip]

# Convert TimeSeries to DataFrame for processing
df_y = y.pd_dataframe()

# Apply averaging function
temp_y = average_timeseries(df_y, skip)
temp_y.columns = y.pd_dataframe().columns
ts_y = TimeSeries.from_dataframe(temp_y)
df_y= temp_y

# Split the dataset because we are not very interested in the zero values at the end
ds = int(92241/skip)

Ts3 = ts_y[:ds]
# =============================================================================

#* Select the time series to use
Ts = Ts3

FullForecast = []
FullLow = []
FullHigh = []

for step in range(0, (len(Ts) - 512 - prediction_length)):
    if historic:
        TheSeries = Ts[startPoint+step:startPoint + 512 + prediction_length + step]
    else:
        TheSeries = Ts

    selected_df2 = pd.Series(np.ravel(y_transformer.inverse_transform(TheSeries)[TARGETS_clean[0]][:-prediction_length].values()))
    selected_df2 = selected_df2.astype('float32')
    context = torch.tensor(selected_df2.values)

    forecast = pipeline.predict(
        context,
        prediction_length,
        num_samples = 3,
        limit_prediction_length=True) 

    low, median, high = np.quantile(forecast[0].numpy(), [0.1, 0.5, 0.9], axis=0)

    forecast = TimeSeries.from_times_and_values(times = TheSeries[TARGETS_clean[0]][-prediction_length:].time_index, 
                                                values = median,
                                                columns=[TARGETS_clean[0]]
                                                )

    FullForecast.append(forecast)
    FullLow.append(low)
    FullHigh.append(high)

    if not historic:
        break

FullForecast_df = [ts.pd_dataframe() for ts in FullForecast]
for df in FullForecast_df:
    df.columns = [TARGETS_clean[0]]



if savePickles:

    #* Save the output
    # =============================================================================
    # Naming the output file
    now = datetime.now()
    date_string = now.strftime("%d%m%Y_%H%M%S")
    commit_id = subprocess.check_output(["git", "describe", "--always"]).strip().decode('utf-8')
    # =============================================================================
    # Save the output
    output_path = os.getcwd() + '/DataDynamo/Output/'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    with open(output_path + f'{commit_id}_FullForecast_{date_string}_Skip{skip}.pkl', 'wb') as f:
        pickle.dump(FullForecast_df, f)
    with open(output_path + f'{commit_id}_FullLow_{date_string}_Skip{skip}.pkl', 'wb') as f:
        pickle.dump(FullLow, f)
    with open(output_path + f'{commit_id}_FullHigh_{date_string}_Skip{skip}.pkl', 'wb') as f:
        pickle.dump(FullHigh, f)

else:
    if historic:
        # =============================================================================
        # Extract the prediction point from the historical forecast
        # Which prediction point in histoical forecast to get and plot. from 0 to len(FullForecast[0]) - 1
        prediction_point = len(FullForecast[0]) - 1
        prediction_time = (prediction_point + 1) * skip * 10 / 60 # In minutes

        # Extract the point of each time series
        point_selected_values = [ts.values()[prediction_point] for ts in FullForecast]
        point_selected_times = [ts.time_index[prediction_point] for ts in FullForecast]

        # Create a new time series from the points
        point_selected_ts = TimeSeries.from_times_and_values(pd.DatetimeIndex(point_selected_times), point_selected_values)

        point_selected_values_Low = [array[prediction_point] for array in FullLow]
        point_selected_values_High = [array[prediction_point] for array in FullHigh]


        point_selected_df = point_selected_ts.pd_dataframe()
        point_selected_df.columns = [TARGETS_clean[0]]
        point_selected_df.index.name = 'Date'
        # Choose between the first, middle and last points
        historical_forecast = TimeSeries.from_dataframe(point_selected_df)
        # =============================================================================
    else:
        Forecast = FullForecast[0]


    TheSeries = y_transformer.inverse_transform(TheSeries)

    if historic:
        '''Plotting'''
        # =============================================================================
        plot_historical_forecast(df = TheSeries.pd_dataframe(),
                        forecast = historical_forecast.pd_dataframe(),
                        lower_percentile = point_selected_values_Low,
                        higher_percentile = point_selected_values_High,
                        target_col = TARGETS_clean[0],
                        ShowEvent=False,
                        labels = None,
                        )    
    else: 
        '''Plotting'''
        # =============================================================================
        # Plot the time series
        plot_historical_forecast(df = TheSeries.pd_dataframe(),
                        forecast = Forecast.pd_dataframe(),
                        lower_percentile = FullLow[0],
                        higher_percentile = FullHigh[0],
                        target_col = TARGETS_clean[0],
                        ShowEvent=False,
                        labels = None,
                        )  