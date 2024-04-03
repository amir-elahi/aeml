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
import joblib

import matplotlib.pyplot as plt
plt.style.reload_library()
plt.style.use('science')
from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'

from chronos import ChronosPipeline
import torch
from transformers import set_seed

'''
In this version I tried to make the forecast for the first timesteps witout skipping any timestep.
Then for the next forecast I skip 1 timestep and so on. The idea is to see how the model behaves when it is done
#TODO: instead of the end time going further, the prediction time goes back.
#TODO: The looping over models and storing the last value is not working.
#TODO: The MASE is not working properly. It is not calculating the correct value when we want to do the loop. 
#! Not having the same frequency is something to think about, maybe we can interpolate the data to have the same frequency.
#! Maybe Working with the same frequency is important for the MASE calculation.
'''



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
print(len(y))
prediction_length = 50
startPoint = 0
endPoint = len(y) - 185900

output = []
for i in range(1,3):
    TheSeries = y[startPoint:endPoint:i]
    print(TheSeries[0].time_index)
    print(TheSeries[-1].time_index)
    # TheSeries2 = y[startPoint:endPoint:2]
    # TheSeries3 = y[startPoint:endPoint:3]
    # TheSeries4 = y[startPoint:endPoint:4]
    # TheSeries5 = y[startPoint:endPoint:5]
    # TheSeries6 = y[startPoint:endPoint:6]
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

    output.append([forecast])


# Assuming output is your list of forecasts

for i in range(0, len(output) - 1):
    # Append the current forecast to the combined_series
    last_time_index = output[i][0].time_index[-1]
    end_time_index = output[i+1][0].time_index[-1]
    output[i+1][0] = output[i+1][0].slice(last_time_index, end_time_index)
    
for i in range(0, len(output)):
    output[i][0].plot()

#* Inverse Transforming the series
TheSeries = y_transformer.inverse_transform(TheSeries)

MASE = mase(TheSeries[TARGETS_clean[0]][-prediction_length:], forecast, TheSeries[TARGETS_clean[0]][-prediction_length-512:-prediction_length])
print(MASE)

'''Plotting'''
plt.figure()
# TheSeries2 = y_transformer.inverse_transform(TheSeries2)
# TheSeries3 = y_transformer.inverse_transform(TheSeries3)
# TheSeries4 = y_transformer.inverse_transform(TheSeries4)
# TheSeries5 = y_transformer.inverse_transform(TheSeries5)
# TheSeries6 = y_transformer.inverse_transform(TheSeries6)

TheSeries[TARGETS_clean[0]].plot(label='True Values No Skip')
# TheSeries2[TARGETS_clean[0]].plot(label='True Values Skip 2')
# TheSeries3[TARGETS_clean[0]].plot(label='True Values Skip 3')
# TheSeries4[TARGETS_clean[0]].plot(label='True Values Skip 4')
# TheSeries5[TARGETS_clean[0]].plot(label='True Values Skip 5')
# TheSeries6[TARGETS_clean[0]].plot(label='True Values Skip 6')


(TheSeries[-prediction_length-512:-prediction_length])[TARGETS_clean[0]].plot(label='Input Values')
forecast.plot(label='Prediction by Chronos')
plt.fill_between(TheSeries[TARGETS_clean[0]][-prediction_length:].time_index, low, high, color="tomato", alpha=0.3, label="80\% prediction interval")


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