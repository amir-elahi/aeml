import pandas as pd 
import numpy as np
from darts import TimeSeries
from darts.models import RegressionModel, LightGBMModel
from darts.metrics import mape, mae, ope
from sklearn.linear_model import BayesianRidge
from models.gbdt.gbmquantile import LightGBMQuantileRegressor
from models.gbdt.run import run_ci_model, run_model
from models.gbdt.settings import *
from models.gbdt.plot import make_forecast_plot


from darts.dataprocessing.transformers import Scaler
import joblib

import matplotlib.pyplot as plt
plt.style.reload_library()
plt.style.use('science')
from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'

from chronos import ChronosPipeline
import torch
from transformers import set_seed

df = pd.read_pickle('/home/lsmo/Desktop/aeml_project/aeml/DataDynamo/RawData/New_campaigns/202403 SCOPE data set dynamic campaign.pkl')

# df = pd.read_excel('./Vapour Dry Bed 18-21 NOV 2022.xlsx', sheet_name='PI-Daten', header=4, nrows = 1700)

df = df.dropna()

TARGETS_clean = ['AMP-4', 'PZ-4'] 

MEAS_COLUMNS = [

    # 'Date',
    # 'PI-2',
    # 'TI-2',
    # 'F-2',
    # 'PI-3',
     'TI-3',
    # 'CO2-3',
    # 'O2-3',
    # 'TI-32',
    # 'TI-33',
    # 'TI-34',
     'TI-35',
    # 'PI-4',
    # 'TI-4',
    # 'F-4',
    # 'CO2-4',
    # 'AMP-4',
    # 'PZ-4',
    # 'NH3-4',
     'F-11',
    # 'TI-12',
    # 'TI-13',
    # 'FI-20',
    # 'FI-211',
    # 'TI-211',
    # 'TI-8',
    # 'TI-9',
    # 'TI-5',
    # 'TI-7',
    # 'TI-28',
    # 'PI-28',
    # 'PI-30',
    # 'TI-30',
    # 'F-30',
    # 'F-38',
    # 'P-38',
    # 'F-36',
    # 'T-36',
    # 'Reboiler duty',
     'F-19',
     'T-19',
    # 'PI-1',
    # 'TI-1',
    # 'TI-35b',
    # 'F-35',
    # 'TI-39',
    # 'F-23',
    # 'TI-22',
    # 'Level Desorber',
    # 'Level Reboiler',
    # 'TI-24',
    # 'TI-25',
    # 'FI-25',
    # 'FI-16',
    # 'TI-16',
    # 'FI-151',
    # 'TI-151',
    # 'TI-152',
    # 'TI-212',
    # 'FI-241',
    # 'TI-241',
    # 'TI-242',
    # 'TI-39',
    # 'TI-35B',
    # 'TI-36',
    # 'Voltage WESP',
    # 'CAPTURE RATE',
    # 'TI-12/TI-13',	
    # 'TI-35 - TI-4'

]

y = TimeSeries.from_dataframe(df, value_cols=TARGETS_clean, time_col='Date')
x = TimeSeries.from_dataframe(df, value_cols=MEAS_COLUMNS, time_col='Date')

# y = TimeSeries.from_dataframe(df, value_cols=TARGETS_clean)
# x = TimeSeries.from_dataframe(df, value_cols=MEAS_COLUMNS)

transformer = Scaler()
x = transformer.fit_transform(x)

y_transformer = Scaler()
y = y_transformer.fit_transform(y)

scal = y_transformer.transform(y)

#* Set the seed for reproducibility
set_seed(42)

#! This part is where we define the torch tensor to feed it to the model. Note that it is not scaled
print(len(y))
prediction_length = 50
startPoint = 41000
endPoint = len(y) -148000
TheSeries = y[startPoint:endPoint]
print(TheSeries[0].time_index)
print(TheSeries[-1].time_index)
TheSeries2 = y[startPoint:endPoint:2]
TheSeries3 = y[startPoint:endPoint:3]
TheSeries4 = y[startPoint:endPoint:4]
TheSeries5 = y[startPoint:endPoint:5]
TheSeries6 = y[startPoint:endPoint:6]
print(len(TheSeries))
# selected_df = df[TARGETS_clean[0]][:len(train_yWESPoff[TARGETS_clean[0]])].reset_index(drop=True)
selected_df2 = pd.Series(np.ravel(y_transformer.inverse_transform(TheSeries)[TARGETS_clean[0]][:-prediction_length].values()))
# selected_df = selected_df.astype('float32')
selected_df2 = selected_df2.astype('float32')
context = torch.tensor(selected_df2.values)



pipeline = ChronosPipeline.from_pretrained(
    "amazon/chronos-t5-tiny",
    device_map="cpu",
    torch_dtype=torch.bfloat16,
)

forecast = pipeline.predict(
    context,
    prediction_length,
    num_samples = 20,
    limit_prediction_length=True)  # shape [num_series, num_samples, prediction_length]


forecast_index = range(len(df), len(df) + prediction_length)

low, median, high = np.quantile(forecast[0].numpy(), [0.1, 0.5, 0.9], axis=0)

forecast = TimeSeries.from_times_and_values(times = TheSeries[TARGETS_clean[0]][-prediction_length:].time_index, values = median)

'''Plotting'''
plt.figure()
TheSeries = y_transformer.inverse_transform(TheSeries)
TheSeries2 = y_transformer.inverse_transform(TheSeries2)
TheSeries3 = y_transformer.inverse_transform(TheSeries3)
TheSeries4 = y_transformer.inverse_transform(TheSeries4)
TheSeries5 = y_transformer.inverse_transform(TheSeries5)
TheSeries6 = y_transformer.inverse_transform(TheSeries6)

TheSeries[TARGETS_clean[0]].plot(label='True Values No Skip')
TheSeries2[TARGETS_clean[0]].plot(label='True Values Skip 2')
TheSeries3[TARGETS_clean[0]].plot(label='True Values Skip 3')
TheSeries4[TARGETS_clean[0]].plot(label='True Values Skip 4')
TheSeries5[TARGETS_clean[0]].plot(label='True Values Skip 5')
TheSeries6[TARGETS_clean[0]].plot(label='True Values Skip 6')

# y_transformer.inverse_transform(TheSeries[-prediction_length-512:-prediction_length])[TARGETS_clean[0]].plot(label='Input Values')

# forecast.plot(label='Prediction by Chronos')
# plt.fill_between(TheSeries[TARGETS_clean[0]][-prediction_length:].time_index, low, high, color="tomato", alpha=0.3, label="80\% prediction interval")


'''Plot Information and decoration'''
plt.ylabel(r'Emissions [$\mathrm{mg/nm^3}$]', fontsize=14)
plt.xlabel('Date', fontsize=14)
adsorbent_in_plot = r'2-Amino-2-methylpropanol $\mathrm{C_4H_{11}NO}$'
plt.title( f'{adsorbent_in_plot}'
          , fontsize=18, fontweight='extra bold')

# Add a frame around the plot area
plt.gca().spines['top'].set_visible(True)
plt.gca().spines['right'].set_visible(True)
plt.gca().spines['bottom'].set_visible(True)
plt.gca().spines['left'].set_visible(True)
plt.legend()

# Adjust font size for tick labels
plt.xticks(fontsize=12)
plt.yticks(fontsize=16)

plt.show()