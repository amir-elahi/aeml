import pandas as pd 
import numpy as np
from darts import TimeSeries
from aeml.models.gbdt.run import run_ci_model, run_model
from aeml.models.gbdt.settings import *

from aeml.preprocessing.resample import resample_regular
from darts.dataprocessing.transformers import Scaler
import joblib

import sys
sys.path.insert(0, '/home/lsmo/Desktop/aeml_project/aeml/DataDynamo/Utils2/')
from Plot import plot_historical_forecast
from metrics import get_metrics

"""
# =============================================================================
Script Name: ModelTrain_lgbm.py
Author(s) <Abrevation>: Amir Elahi <AE>
Date: 17 April 2022

TL;DR:
    This script is used to train the LightGBM model on the data.

Description:
    < >.

Usage:
    To Generate the historical forecast of the LightGBM model.
    To plot the histrical forecast of the LightGBM model. 

Dependencies:
    Refer to the *.yml file based on your operating system.

Notes:
    < >.

#TODO:
    < >.

Version History:
    <Date>, <Author>, <Description of Changes>
    14  April 2022 AE Add header to the script. Make TI-1213 and minor fixes.
    23  April 2022 AE Add the more efficient way of averaging. This should also happem in other scripts.
# =============================================================================
"""

np.random.seed(42)

df = pd.read_pickle('./DataDynamo/RawData/New_campaigns/202403 SCOPE data set dynamic campaign.pkl')

df = df.dropna()

df['TI-1213'] = np.where(df['valve position'] == 1, df['TI-13'], df['TI-12'])

TARGETS_clean = ['AMP-4', 'PZ-4'] 


MEAS_COLUMNS = [

# 'Date',
# 'PI-2',
# 'TI-2',
# 'F-3',
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
# 'ACA',
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
# 'TI-40',
# 'F-40',
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
'valve position',
# 'T-15',
# 'dp-32',
# 'dp-33',
# 'dp-34',
# 'dp-35',
# 'dp-36',
# 'Level Adsorber',
# 'TI-071',
# 'TI-072',
# 'TI-070',
# 'dp-071',
# 'dp-072',
# 'dp-073',
# 'flow process water',
# 'temperature processwater inlet acid wash',
# 'pH process water',
# 'demin water flow',
# 'H2SO4 flow',
# 'level column',

]

MEAS_COLUMNS = [ 'T-19', 'TI-3', 'F-19','F-11', 'TI-1213','TI-35']

startPoint = 0
endPoint = len(df)
skip = 1

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

y = ts_y

temp_x = average_timeseries(df_x, skip)
temp_x.columns = x.pd_dataframe().columns
ts_x = TimeSeries.from_dataframe(temp_x)
df_x = temp_x

x = ts_x

ds = int(92241 / skip)

# Break the dataset so that we don't consider the zero values in the end. The time index is roughly 2024-03-01 16:13:20
y = y[:ds]
x = x[:ds]

# Break the dataset into train and validation
train_percentage = 0.1
train_length = int(train_percentage * len(y))
y_train, y_val = y[:train_length] , y[train_length:]
x_train, x_val = x[:train_length] , x[train_length:]

######################################

# gbdt_all_data_0 = run_ci_model(xWESPoff, yWESPoff[TARGETS_clean[0]], **ci_6_0,
#                                 output_chunk_length=100, num_features= 7 )
# gbdt_all_data_1 = run_model(xWESPoff, yWESPoff[TARGETS_clean[1]], **settings_1_1, output_chunk_length=1)



# https://wandb.ai/amir_elahi/aeml_amir/sweeps/l3g3deg6?workspace=user-amirelahi-9877
# https://wandb.ai/amir_elahi/aeml_amir/runs/dxytxpiu/overview?workspace=user-amirelahi-9877
# Step 1
# settingsAmir = {
#     "bagging_fraction": 0.9515160572745732,
#     "bagging_freq": 8,
#     "extra_trees": False,
#     "lag_1": -39,
#     "lag_2": -48,
#     "lag_3": -5,
#     "lag_4": -4,
#     "lag_5": -57,
#     "lag_6": -92,
#     "lags": 40,
#     "max_depth": 40,
#     "n_estimators": 902,
#     "num_leaves": 110,
# }

# Sweep: https://wandb.ai/amir_elahi/aeml_amir/sweeps/lesd9cts/overview
# Best Run: https://wandb.ai/amir_elahi/aeml_amir/runs/c48z7bqh/overview?workspace=user-amirelahi-9877
# Step 30
settingsAmir = {
    "bagging_fraction": 0.13321852266317588,
    "bagging_freq": 0,
    "extra_trees": True,
    "lag_1": -33,
    "lag_2": -92,
    "lag_3": -15,
    "lag_4": -46,
    "lag_5": -12,
    "lag_6": -92,
    "lags": 112,
    "max_depth": 160,
    "n_estimators": 476,
    "num_leaves": 84,
}


# https://wandb.ai/amir_elahi/aeml_amir/sweeps/kf9tptza?workspace=user-amirelahi-9877
# https://wandb.ai/amir_elahi/aeml_amir/runs/ucefzkis/overview?workspace=user-amirelahi-9877
# Step 60
# settingsAmir = {
#     "bagging_fraction": 0.08045087267568593,
#     "bagging_freq": 4,
#     "extra_trees": True,
#     "lag_1": -19,
#     "lag_2": -130,
#     "lag_3": -29,
#     "lag_4": -36,
#     "lag_5": -51,
#     "lag_6": -133,
#     "lags": 86,
#     "max_depth": 320,
#     "n_estimators": 924,
#     "num_leaves": 66,
# }

# Check the step and setting be relavant
step = 64

my_dict = {
'quantiles' : (0.1, 0.5, 0.9)
}

gbdt_all_data_0 = run_model( x_train, y_train[TARGETS_clean[0]], **settingsAmir, 
                            output_chunk_length = step, quantiles= my_dict.get('quantiles') )

# forecast_0 = gbdt_all_data_0.forecast(n=len(yWESPon[TARGETS_clean[0]]),series = yWESPoff[TARGETS_clean[0]])
# forecast_0 = gbdt_all_data_0.forecast(n=1 ,series = train_yWESPoff[TARGETS_clean[0]])

historical_forceasts_0 = gbdt_all_data_0.historical_forecasts(
    series=y[TARGETS_clean[0]],  past_covariates=x, start=train_length , retrain=False, forecast_horizon=step, show_warnings=False
)

# gbdt_all_data_0 = RegressionModel(lags=10, model=BayesianRidge(), lags_past_covariates=5)
# gbdt_all_data_0.fit(yWESPoff[TARGETS_clean[0]], past_covariates= x)

# forecast_0 = gbdt_all_data_0.predict(n = 30 ,series=yWESPoff[TARGETS_clean[0]], past_covariates= x)

#######################################


# gbdt_all_data_0 = LightGBMModel(
#     lags=100,
#     lags_past_covariates=100,
#     output_chunk_length=10,
#     verbose=-1
# )

# gbdt_all_data_0.fit(yWESPoff[TARGETS_clean[0]], past_covariates=x)

# forecast_0=gbdt_all_data_0.predict(series=yWESPoff[TARGETS_clean[0]], n = 300, past_covariates=x)


'''Scale back'''
ts1 = historical_forceasts_0[1]
temp = TimeSeries.from_series(
    pd.concat([ts1.pd_series(), ts1.pd_series()], axis=1, keys=[TARGETS_clean[0], 'dummy'])
)
y_forecast = y_transformer.inverse_transform(temp)[TARGETS_clean[0]]

ts1 = historical_forceasts_0[0]
temp = TimeSeries.from_series(
    pd.concat([ts1.pd_series(), ts1.pd_series()], axis=1, keys=[TARGETS_clean[0], 'dummy'])
)
lower_percentile = y_transformer.inverse_transform(temp)[TARGETS_clean[0]].pd_dataframe()

ts1 = historical_forceasts_0[2]
temp = TimeSeries.from_series(
    pd.concat([ts1.pd_series(), ts1.pd_series()], axis=1, keys=[TARGETS_clean[0], 'dummy'])
)
higher_percentile = y_transformer.inverse_transform(temp)[TARGETS_clean[0]].pd_dataframe()

y_actual = y_transformer.inverse_transform(y)[TARGETS_clean[0]]

try:
    metrics = get_metrics(actual = y_actual,
                          predicted = y_forecast)
    print(metrics)
except Exception as e:
    metrics = None
    print(f'An error occured during metrics calcilations: {e}')

plot_historical_forecast(df = y_actual.pd_dataframe(),
                        forecast = y_forecast.pd_dataframe(),
                        lower_percentile = lower_percentile.values.ravel(),
                        higher_percentile = higher_percentile.values.ravel(),
                        target_col = TARGETS_clean[0],
                        time_col = 'Date',
                        title = None,
)                        