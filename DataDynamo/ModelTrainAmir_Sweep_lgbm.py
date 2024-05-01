import pandas as pd 
import numpy as np
from darts import TimeSeries
from aeml.models.gbdt.run import run_model
import wandb

from darts.dataprocessing.transformers import Scaler

import sys
sys.path.insert(0, '/home/lsmo/Desktop/aeml_project/aeml/DataDynamo/Utils2/')
from sweep import start_sweep
from metrics import get_metrics

''' Sweep Configuration for wandb'''

import time
timestr = time.strftime("%Y%m%d-%H%M%S")

############################################################################################################

np.random.seed(42)

df = pd.read_pickle('./DataDynamo/RawData/New_campaigns/202403 SCOPE data set dynamic campaign.pkl')

df = df.dropna()

df['TI-1213'] = np.where(df['valve position'] == 1, df['TI-13'], df['TI-12'])

TARGETS_clean = ['AMP-4', 'PZ-4'] 

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
train_percentage = 0.2
validation_percentage = 0.1
validation_length = int(validation_percentage * len(y))
train_length = int(train_percentage * len(y))
y_train, y_val = y[:train_length] , y[train_length:validation_length]
x_train, x_val = x[:train_length] , x[train_length:validation_length]


sweep_config = {
    "metric": {"goal": "minimize", "name": "score"},
    "name": f"Sweep_Time_{timestr}_skip_{skip}",
    "method": "bayes",
    "parameters": {
        "lags":  {"min":  1,   "max": 190, "distribution": "int_uniform"},
        "lag_1": {"min": -170, "max": -1,  "distribution": "int_uniform"},
        "lag_2": {"min": -170, "max": -1,  "distribution": "int_uniform"},
        "lag_3": {"min": -170, "max": -1,  "distribution": "int_uniform"},
        "lag_4": {"min": -170, "max": -1,  "distribution": "int_uniform"},
        "lag_5": {"min": -170, "max": -1,  "distribution": "int_uniform"},
        "lag_6": {"min": -170, "max": -1,  "distribution": "int_uniform"},
        "n_estimators": {"min": 50, "max": 1000},
        "bagging_freq": {"min": 0, "max": 10, "distribution": "int_uniform"},
        "bagging_fraction": {"min": 0.001, "max": 1.0},
        "num_leaves": {"min": 1, "max": 200, "distribution": "int_uniform"},
        "extra_trees": {"values": [True, False]},
        "max_depth": {"values": [-1, 10, 20, 40, 80, 160, 320]},
    },
}


wandb.login()
sweep_id = wandb.sweep(sweep_config, project='aeml_amir')

def objective(config,
              y_train = y_train,
              x_train = x_train,
              y_val = y_val,
              x_val = x_val,
              Target = TARGETS_clean[0]):

    settingsAmir = {
        "bagging_fraction": config.bagging_fraction,
        "bagging_freq": config.bagging_freq,
        "extra_trees": config.extra_trees,
        "lag_1": config.lag_1,
        "lag_2": config.lag_2,
        "lag_3": config.lag_3,
        "lag_4": config.lag_4,
        "lag_5": config.lag_5,
        "lag_6": config.lag_6,
        "lags": config.lags,
        "max_depth": config.max_depth,
        "n_estimators": config.n_estimators,
        "num_leaves": config.num_leaves,
    }


    gbdt_all_data_0 = run_model(x_train, y_train[Target], **settingsAmir, 
                                output_chunk_length=64,
                                quantiles=(0.1, 0.5, 0.9))

    forecast_0 = gbdt_all_data_0.forecast(n=64 ,series = y_val[Target], past_covariates = x_val)

    ''' Calculating metrics'''

    try:
        metrics = get_metrics(actual = y_val[Target] ,
                          predicted = forecast_0[1])
        print(metrics)
    except Exception as e:
        metrics = None
        print(f'An error occured during metrics calcilations: {e}')

    score = metrics['mae']

    return score

def main():
    wandb.init(project='aeml_amir')
    score = objective(wandb.config)
    wandb.log({
        "score": score,
        "skip": skip
    })

wandb.agent(sweep_id, function=main, count = 100)