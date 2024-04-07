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

import matplotlib.pyplot as plt
plt.style.reload_library()
plt.style.use('science')
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
    Make the figures more readable and better looking.
    Calculate the MASE for the model.

Version History:
    <Date>, <Author>, <Description of Changes>
    6 April 2022 AE Add the historical forecast and averaging of the data using Chronos model.

# =============================================================================
"""



df = pd.read_pickle('/home/lsmo/Desktop/aeml_project/aeml/DataDynamo/RawData/New_campaigns/202403 SCOPE data set dynamic campaign.pkl')

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
prediction_length = 64
startPoint = 0
endPoint = len(y)
skip = 96 * 3

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
    TheSeries = Ts[startPoint+step:startPoint + 512 + prediction_length + step]

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

FullForecast_df = [ts.pd_dataframe() for ts in FullForecast]

# for i in range(0, len(FullForecast)):
#     FullForecast[i][-1].plot(label = f'step {i} Forecast')
#     plt.fill_between(TheSeries[TARGETS_clean[0]][-prediction_length:].time_index, FullLow[i][-1], FullHigh[i][-1], color="tomato", alpha=0.3)


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


# # Extract the last point of each time series
# last_points_values = [ts.last_value() for ts in FullForecast]
# last_points_times = [ts.time_index[-1] for ts in FullForecast]

# # Create a new time series from the last points
# last_points_ts = TimeSeries.from_times_and_values(pd.DatetimeIndex(last_points_times), last_points_values)

# # Plot the new time series
# last_points_ts.plot(label='Last points')


# #* Inverse Transforming the series
# TheSeries = y_transformer.inverse_transform(TheSeries)
# y = y_transformer.inverse_transform(y)

# # MASE = mase(TheSeries[TARGETS_clean[0]][-prediction_length:], forecast, TheSeries[TARGETS_clean[0]][-prediction_length-512:-prediction_length])
# # print(MASE)

# '''Plotting'''
# # plt.figure()

# y[TARGETS_clean[0]][startPoint:endPoint:skip].plot(label=f'True Values {skip - 1} Skip')
# y[TARGETS_clean[0]][startPoint:startPoint + 512*skip: skip].plot(label=f'First 512 True Values')

# # (TheSeries[-prediction_length-512:-prediction_length])[TARGETS_clean[0]].plot(label='Input Values')
# # forecast.plot(label='Prediction by Chronos')
# # plt.fill_between(TheSeries[TARGETS_clean[0]][-prediction_length:].time_index, low, high, color="tomato", alpha=0.3, label="80\% prediction interval")


# '''Plot Information and decoration'''
# plt.ylabel(r'Emissions [$\mathrm{mg/nm^3}$]', fontsize=14)
# plt.xlabel('Date', fontsize=14)
# adsorbent_in_plot = r'2-Amino-2-methylpropanol $\mathrm{C_4H_{11}NO}$'
# plt.title( f'{adsorbent_in_plot}'
#           , fontsize=18, fontweight='extra bold')

# # Add a frame around the plot area
# plt.gca().spines['top'].set_visible(True)
# plt.gca().spines['right'].set_visible(True)
# plt.gca().spines['bottom'].set_visible(True)
# plt.gca().spines['left'].set_visible(True)
# plt.legend()

# # Adjust font size for tick labels
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=16)

# plt.show()