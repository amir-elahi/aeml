import pandas as pd 
import numpy as np
from darts import TimeSeries
from darts.models import RegressionModel, LightGBMModel
from darts.metrics import mape, mae, ope, mase
from sklearn.linear_model import BayesianRidge
from aeml.models.gbdt.gbmquantile import LightGBMQuantileRegressor
from aeml.models.gbdt.run import run_ci_model, run_model
from aeml.models.gbdt.settings import *
from aeml.models.gbdt.plot import make_forecast_plot


from darts.dataprocessing.transformers import Scaler
import joblib, pickle, subprocess

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

plt.style.reload_library()
# plt.style.use('grid')

from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'

from chronos import ChronosPipeline
import torch
from transformers import set_seed
from datetime import datetime
import os

"""
# =============================================================================
Script Name: ModelTrain_Choronos.py
Author(s) <Abrevation>: Amir Elahi <AE>
Date: 6 April 2022

TL;DR:
    This script is used to train the Chronos model on the data and make predictions.

Description:
    This Python script does the histrical forecast of the data using Chronos model.
    It can average and skip the data points to reduce the number of points to be fed to the model.

Usage:
    Produce histroical forecast of the data using Chronos model.

Dependencies:
    Refer to the *.yml file based on your operating system.

Notes:
    < >.

#TODO:
    < >.

Version History:
    <Date>, <Author>, <Description of Changes>
    6  April 2022 AE Add the historical forecast and averaging of the data using Chronos model.
    8  April 2022 AE Configure the plotting and saving of the output. Adding flags to make it easier to use.
    14 April 2022 AE Minor changes in plotting.

# =============================================================================
"""



df = pd.read_pickle('./DataDynamo/RawData/New_campaigns/202403 SCOPE data set dynamic campaign.pkl')

df = df.dropna()

TARGETS_clean = ['AMP-4', 'PZ-4'] 

MEAS_COLUMNS = [
     'TI-3',
     'TI-35',
     'F-11',
     'F-19',
     'T-19']

y = TimeSeries.from_dataframe(df, value_cols=TARGETS_clean, time_col='Date')
x = TimeSeries.from_dataframe(df, value_cols=MEAS_COLUMNS, time_col='Date')

transformer = Scaler()
x = transformer.fit_transform(x)

y_transformer = Scaler()
y = y_transformer.fit_transform(y)

scal = y_transformer.transform(y)

#* Set the seed for reproducibility
set_seed(42)

#! This part is where we define the torch tensor to feed it to the model. Note that it is not scaled
prediction_length = 50
startPoint = 0
endPoint = len(y)
skip = 1
savePickles = True # True will save pickles and don't plot, False will plot and don't save pickles
historic = True # True will perform historical forecast, False will perform the normal forecast

pipeline = ChronosPipeline.from_pretrained(
    "amazon/chronos-t5-tiny",
    device_map="cuda",
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
# Averaging is used to average the points in the time series instead of skipping
def average_timeseries(y, startPoint, endPoint, skip):
    averaged_values = []
    for i in range(startPoint, endPoint, skip):
        avg = sum(y.values()[i:i+skip]) / skip
        averaged_values.append(avg)
    return averaged_values

Ts3 = average_timeseries(y, startPoint, endPoint, skip)
Ts3 = TimeSeries.from_times_and_values(times = y[startPoint:endPoint:skip].time_index, values = Ts3)
Ts3 = Ts3.pd_dataframe()
Ts3.columns = y.pd_dataframe().columns
Ts3 = TimeSeries.from_dataframe(Ts3)
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
        num_samples = 20,
        limit_prediction_length=True) 

    forecast_index = range(len(df), len(df) + prediction_length)

    low, median, high = np.quantile(forecast[0].numpy(), [0.1, 0.5, 0.9], axis=0)

    forecast = TimeSeries.from_times_and_values(times = TheSeries[TARGETS_clean[0]][-prediction_length:].time_index, values = median)
    # forecast.plot(label = f'step {step} Forecast')
    # plt.fill_between(TheSeries[TARGETS_clean[0]][-prediction_length:].time_index, low, high, color="tomato", alpha=0.3)
    FullForecast.append(forecast)
    FullLow.append(low)
    FullHigh.append(high)
    if not historic:
        break


FullForecast_df = [ts.pd_dataframe() for ts in FullForecast]

# for i in range(0, len(FullForecast)):
#     FullForecast[i][-1].plot(label = f'step {i} Forecast')
#     plt.fill_between(TheSeries[TARGETS_clean[0]][-prediction_length:].time_index, FullLow[i][-1], FullHigh[i][-1], color="tomato", alpha=0.3)


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
        # Extract the first point of each time series
        first_points_values = [ts.first_value() for ts in FullForecast]
        first_points_times = [ts.time_index[0] for ts in FullForecast]

        # Create a new time series from the first points
        first_points_ts = TimeSeries.from_times_and_values(pd.DatetimeIndex(first_points_times), first_points_values)

        first_points_values_Low = [array[0] for array in FullLow]
        first_points_values_High = [array[0] for array in FullHigh]

        # Extract the middle point of each time series
        middle_points_values = [ts.values()[len(ts.values()) // 2] for ts in FullForecast]
        middle_points_times = [ts.time_index[len(ts.values()) // 2] for ts in FullForecast]

        # Create a new time series from the middle points
        middle_points_ts = TimeSeries.from_times_and_values(pd.DatetimeIndex(middle_points_times), middle_points_values)

        middle_points_values_Low = [array[len(array) // 2] for array in FullLow]
        middle_points_values_High = [array[len(array) // 2] for array in FullHigh]

        # Extract the last point of each time series
        last_points_values = [ts.last_value() for ts in FullForecast]
        last_points_times = [ts.time_index[-1] for ts in FullForecast]

        # Create a new time series from the last points
        last_points_ts = TimeSeries.from_times_and_values(pd.DatetimeIndex(last_points_times), last_points_values)

        last_points_values_Low = [array[-1] for array in FullLow]
        last_points_values_High = [array[-1] for array in FullHigh]

        # Choose between the first, middle and last points
        historical_forecast = last_points_ts
        # =============================================================================
    else:
        Forecast = FullForecast[0]


    TheSeries = y_transformer.inverse_transform(TheSeries)

    if historic:
        '''Plotting'''
        # =============================================================================
        # Plot the time series
        fig = plt.figure(figsize=(3.5*3, 0.6*3.5*3))
        fig.subplots_adjust(bottom=0.2, left= 0.1)

        TheSeries[TARGETS_clean[0]].plot(label=f'True values averaging every {skip - 1} points')
        TheSeries[TARGETS_clean[0]][:512].plot(label=f'First 512 true values')

        historical_forecast.plot(label='Historical Forecast')
        plt.fill_between(last_points_times, last_points_values_Low, last_points_values_High, color="tomato", alpha=0.3, label="80% prediction interval")
    else: 
        '''Plotting'''
        # =============================================================================
        # Plot the time series
        fig = plt.figure(figsize=(2.5*3.5/1.2, 1.8*3.5/1.2))
        fig.subplots_adjust(bottom=0.2, left= 0.15)

        TheSeries[TARGETS_clean[0]].plot(label=f'True values')
        # TheSeries[TARGETS_clean[0]][-prediction_length-512:-prediction_length].plot(label=f'True input values')

        Forecast.plot(label='Forecast')
        plt.fill_between(Forecast.time_index, FullLow[0], FullHigh[0], color="tomato", alpha=0.3, label="80% prediction interval")


    '''Plot Information and decoration'''
    # =============================================================================
    # Setting the font properties
    plt.rcParams['font.family'] = 'sans-serif'

    fpLegend = '/home/lsmo/.local/share/fonts/calibri-regular.ttf'
    fpLegendtitle = '/home/lsmo/.local/share/fonts/coolvetica rg.otf'
    fpTitle = '/home/lsmo/.local/share/fonts/coolvetica rg.otf'
    fpLabel = '/home/lsmo/.local/share/fonts/Philosopher-Bold.ttf'
    fpTicks = '/home/lsmo/.local/share/fonts/Philosopher-Regular.ttf'

    fLegend = FontProperties(fname=fpLegend, size = 13)
    fLegendtitle = FontProperties(fname=fpLegendtitle, size = 14)
    fTitle = FontProperties(fname=fpTitle, size = 18)
    fLabel = FontProperties(fname=fpLabel, size = 16)
    fTicks = FontProperties(fname=fpTicks, size = 15)

    # =============================================================================
    # Add labels and title and ticks
    plt.ylabel(r'Emissions $[\mathrm{mg/nm^3}]$', fontproperties = fLabel)
    plt.xlabel('Date', fontproperties = fLabel)
    adsorbent_in_plot = r'2-Amino-2-methylpropanol $(\mathrm{C_4H_{11}NO})$'
    plt.title( f'{adsorbent_in_plot}'
            , fontproperties = fTitle)

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
    legend = plt.legend(fontsize = 12, prop = fLegend)
    legend.set_title('Legend', prop = fLegendtitle)

    # =============================================================================
    # Set the date format
    date_format = mdates.DateFormatter('%b-%d')
    plt.gca().xaxis.set_major_formatter(date_format)

    # =============================================================================
    # Adjust font size for tick labels
    plt.xticks(rotation='vertical', fontproperties = fTicks)
    plt.yticks(fontproperties = fTicks)
    # =============================================================================
    # Adjust the y and x limits
    # plt.ylim(99.52176895141602, 122.39603195190428)
    # plt.xlim(19787.72662037037, 19788.139120370368 )

    plt.show()