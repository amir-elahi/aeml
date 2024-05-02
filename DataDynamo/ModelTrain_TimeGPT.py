import pandas as pd 
import numpy as np
from darts import TimeSeries
from nixtlats import NixtlaClient
import os

from darts.dataprocessing.transformers import Scaler
from datetime import datetime


import sys
sys.path.insert(0, '/home/lsmo/Desktop/aeml_project/aeml/DataDynamo/Utils2/')
from Plot import plot_historical_forecast
from metrics import get_metrics
from Save_and_Load import save_to_pickle, load_from_pickle

"""
# =============================================================================
Script Name: ModelTrain_lgbm.py
Author(s) <Abrevation>: Amir Elahi <AE>
Date: 17 April 2022

TL;DR:
    This script is used to train the TimeGPT on the data.

Description:
    < >.

Usage:
    This script will use the TimeGPT model to predict the emissiosn. 

Dependencies:
    Refer to the *.yml file based on your operating system.

Notes:
    < >.

#TODO:
    < >.

Version History:
    <Date>, <Author>, <Description of Changes>
    22  April 2022 AE Add header to the script. Make TI-1213 and minor fixes.

# =============================================================================
"""

df = pd.read_pickle('./DataDynamo/RawData/New_campaigns/202403 SCOPE data set dynamic campaign.pkl')

df = df.dropna()

df['TI-1213'] = np.where(df['valve position'] == 1, df['TI-13'], df['TI-12'])

TARGETS_clean = ['AMP-4', 'PZ-4'] 

MEAS_COLUMNS = [ 'T-19', 'TI-3', 'F-19','F-11', 'TI-1213','TI-35']

startPoint = 0
endPoint = len(df)
skip = 48

y = TimeSeries.from_dataframe(df, value_cols=TARGETS_clean, time_col='Date')
x = TimeSeries.from_dataframe(df, value_cols=MEAS_COLUMNS, time_col='Date')


transformer = Scaler()
x = transformer.fit_transform(x)

y_transformer = Scaler()
y = y_transformer.fit_transform(y)

# This averages skip point in the time series. example: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10: skip=2, then 1, 2.5, 4.5, 6.5, 8.5
def average_timeseries(df, skip):
    return df.rolling(window=skip, min_periods=1).mean()[::skip]

# Convert TimeSeries to DataFrame for processing
df_y = y.pd_dataframe()
df_x = x.pd_dataframe()

# Apply averaging function
temp_y = average_timeseries(df_y, skip)
temp_y.columns = y.pd_dataframe().columns
ts_y = TimeSeries.from_dataframe(temp_y)
df_y= temp_y

y = df_y

temp_x = average_timeseries(df_x, skip)
temp_x.columns = x.pd_dataframe().columns
ts_x = TimeSeries.from_dataframe(temp_x)
df_x = temp_x

x = df_x

ds = int(92241 / skip)

# Break the dataset so that we don't consider the zero values in the end. The time index is roughly 2024-03-01 16:13:20
y = y[:ds]
x = x[:ds]

y = y.reset_index()
y = y[['Date', TARGETS_clean[0]]]

######################################
# Train the model
nixt_token = os.environ.get("NIXTLA_API_KEY")

nixtla_client = NixtlaClient(
    api_key = nixt_token
)

nixtla_client.validate_api_key()

input_values = {
    'df': y,
    'h': 64,
    'time_col': 'Date',
    'target_col': TARGETS_clean[0],
    'add_history': True,
    'level': [80]
}


timegpt_fcst_df = nixtla_client.forecast(df=input_values['df'],
                                         h=input_values['h'],
                                         time_col=input_values['time_col'],
                                         target_col=input_values['target_col'],
                                         add_history=input_values.get('add_history', False),
                                         level = input_values.get('level', None)
)



# Save the forecast
location = '/home/lsmo/Desktop/aeml_project/aeml/DataDynamo/Output/TimeGPT/'
now = datetime.now()
date_string = now.strftime("%d%m%Y-%H%M%S")
output_file = f'timegpt_{date_string}_Skip{skip}.pkl'
save_to_pickle(input_values, timegpt_fcst_df, output_file=output_file, location=location)

# Load the forecast
# location = '/home/lsmo/Desktop/aeml_project/aeml/DataDynamo/Output/TimeGPT/'
# input_file = f'timegpt_29042024-170635_Skip{skip}.pkl'
# input_values, timegpt_fcst_df = load_from_pickle(input_file=input_file, location=location)
# print(input_values)

# '''Scale back'''
timegpt_fcst_df['Date'] = pd.to_datetime(timegpt_fcst_df['Date'])
timegpt_fcst_df = timegpt_fcst_df.set_index('Date')
temp = TimeSeries.from_series(
    pd.concat([timegpt_fcst_df['TimeGPT'], timegpt_fcst_df['TimeGPT']], axis=1, keys=[TARGETS_clean[0], 'dummy'])
)

y_forecast = y_transformer.inverse_transform(temp)[TARGETS_clean[0]]

if 'TimeGPT-lo-80' and 'TimeGPT-hi-80' in timegpt_fcst_df.columns:
    temp = TimeSeries.from_series(
        pd.concat([timegpt_fcst_df['TimeGPT-lo-80'], timegpt_fcst_df['TimeGPT-lo-80']], axis=1, keys=[TARGETS_clean[0], 'dummy'])
    )
    lower_percentile = y_transformer.inverse_transform(temp)[TARGETS_clean[0]]


    temp = TimeSeries.from_series(
        pd.concat([timegpt_fcst_df['TimeGPT-hi-80'], timegpt_fcst_df['TimeGPT-hi-80']], axis=1, keys=[TARGETS_clean[0], 'dummy'])
    )
    higher_percentile = y_transformer.inverse_transform(temp)[TARGETS_clean[0]]
else:
    lower_percentile = None
    higher_percentile = None

y = y.set_index('Date')
temp = TimeSeries.from_series(
    pd.concat([y[TARGETS_clean[0]], y[TARGETS_clean[0]]], axis=1, keys=[TARGETS_clean[0], 'dummy'])
)

y_actual = y_transformer.inverse_transform(temp)[TARGETS_clean[0]]

######################################
'''Calculating metrics'''
try:
    metrics = get_metrics(actual = y_actual,
                          predicted = y_forecast)
    print(metrics)
except Exception as e:
    metrics = None
    print(f'An error occured during metrics calcilations: {e}')

'''Plotting'''
plot_historical_forecast(df = y_actual.pd_dataframe(),
                        forecast = y_forecast.pd_dataframe(),
                        lower_percentile = lower_percentile.pd_dataframe().values.ravel(),
                        higher_percentile = higher_percentile.pd_dataframe().values.ravel(),
                        target_col = TARGETS_clean[0],
                        time_col = 'Date',
                        title = None,
)