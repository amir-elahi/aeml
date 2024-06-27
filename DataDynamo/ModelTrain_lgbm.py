import pandas as pd 
import numpy as np
from darts import TimeSeries
from aeml.models.gbdt.run import run_model

from darts.dataprocessing.transformers import Scaler
from aeml.utils.Plot import *
from aeml.utils.metrics import get_metrics

#! Script to plot the historical forecast and the error plot for the lgbm model (Figure 5b and Figure 6b)

"""
# =============================================================================
TL;DR:
    This script is used to train the LightGBM model on the data.

Usage:
    To Generate the historical forecast of the LightGBM model.
    To plot the histrical forecast of the LightGBM model. 
    To plot the AE vs Time plot of the LightGBM model.

Dependencies:
    Refer to the *.yml file based on your operating system.

Version History:
    <Date>, <Author>, <Description of Changes>
    14  April 2022 AE Add header to the script. Make TI-1213 and minor fixes.
    23  April 2022 AE Add the more efficient way of averaging. This should also happem in other scripts.
    27  June  2022 AE Ready for public release.
# =============================================================================
"""

np.random.seed(42)

df = pd.read_pickle('./DataDynamo/RawData/New_campaigns/202403 SCOPE data set dynamic campaign.pkl')

df = df.dropna()

df['TI-1213'] = np.where(df['valve position'] == 1, df['TI-13'], df['TI-12'])

TARGETS_clean = ['AMP-4', 'PZ-4'] 


MEAS_COLUMNS = [ 'T-19', 'TI-3', 'F-19','F-11', 'TI-1213','TI-35']

startPoint = 0
endPoint = len(df)
skip = 48
Error_plot = False # Set to True to plot the error plot and false to plot the historical forecast

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
train_percentage = 0.4
train_length = int(train_percentage * len(y))
y_train, y_val = y[:train_length] , y[train_length:]
x_train, x_val = x[:train_length] , x[train_length:]

######################################

# Sweep: https://wandb.ai/amir_elahi/aeml_amir/sweeps/lesd9cts/overview
# Best Run: https://wandb.ai/amir_elahi/aeml_amir/runs/hfy7bpnj/overview
# Step 64
# Skip 48
settingsAmir = {
    "bagging_fraction": 0.8846522786364932,
    "bagging_freq": 9,
    "extra_trees": True,
    "lag_1": -45,
    "lag_2": -39,
    "lag_3": -39,
    "lag_4": -51,
    "lag_5": -26,
    "lag_6": -38,
    "lags": 40,
    "max_depth": 10,
    "n_estimators": 919,
    "num_leaves": 176,
}


# Check the step and setting be relavant
step = 64

my_dict = {
'quantiles' : (0.1, 0.5, 0.9)
}

gbdt_all_data_0 = run_model( x_train, y_train[TARGETS_clean[0]], **settingsAmir, 
                            output_chunk_length = step, quantiles= my_dict.get('quantiles') )



if not Error_plot:

    historical_forceasts_0 = gbdt_all_data_0.historical_forecasts(
        series=y[TARGETS_clean[0]],  past_covariates=x, start=train_length , retrain=False, forecast_horizon=step, show_warnings=False,
        last_points_only=True
        )
    
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
                              predicted = y_forecast,
                              train_actual = y_train[TARGETS_clean[0]])
        print(metrics)
    except Exception as e:
        metrics = None
        print(f'An error occured during metrics calculations: {e}')

    plot_historical_forecast(df = y_actual.pd_dataframe(),
                            forecast = y_forecast.pd_dataframe(),
                            lower_percentile = lower_percentile.values.ravel(),
                            higher_percentile = higher_percentile.values.ravel(),
                            target_col = TARGETS_clean[0],
                            # output_Name=f'LGBM_{skip}',
                            title = None,
    )                        

else:

    historical_forceasts_0 = gbdt_all_data_0.historical_forecasts(
        series=y[TARGETS_clean[0]],  past_covariates=x, start=train_length , retrain=False, forecast_horizon=step, show_warnings=False,
        last_points_only=False
        )

    for i in range(0, len(historical_forceasts_0[1])):
        ts1 = historical_forceasts_0[1][i]
        temp = TimeSeries.from_series(
            pd.concat([ts1.pd_series(), ts1.pd_series()], axis=1, keys=[TARGETS_clean[0], 'dummy'])
        )
        historical_forceasts_0[1][i] = y_transformer.inverse_transform(temp)[TARGETS_clean[0]]

    y_forecast_list = []
    time_horizon = []
    for i in range(0,len(historical_forceasts_0[1][0])):
        ts_values = [ts.values()[i] for ts in historical_forceasts_0[1]]
        ts_times = [ts.time_index[i] for ts in historical_forceasts_0[1]]
        ts = TimeSeries.from_times_and_values(pd.DatetimeIndex(ts_times), ts_values)
        time_horizon.append(f"{((i + 1) * skip * 10 / 60 / 60):.2f}") # In hours
        y_forecast_list.append(ts)

    y_actual_list = [y_transformer.inverse_transform(y)[TARGETS_clean[0]]] * len(y_forecast_list)


    make_ae_error_plot(y_actual_list[::5] + [y_actual_list[-1]], 
                    y_forecast_list[::5] + [y_forecast_list[-1]], 
                    time_horizon[::5] + [time_horizon[-1]],
                    Violin=False,
                    Box=True,
                    # output_Name=f'LGBM_{skip}_Error'
                    )