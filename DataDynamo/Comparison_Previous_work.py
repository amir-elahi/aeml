import pandas as pd
import numpy as np
import joblib
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler


from aeml.utils.Plot import plot_historical_forecast
from aeml.utils.metrics import get_metrics

#! Script to plot the historical forecast with the model trained from the work of Jablonka et al. (2020) (Figure 7)

np.random.seed(42)

df = pd.read_pickle('./DataDynamo/RawData/New_campaigns/202403 SCOPE data set dynamic campaign.pkl')

df = df.dropna()

df['TI-1213'] = np.where(df['valve position'] == 1, df['TI-13'], df['TI-12'])

TARGETS_clean = ['AMP-4', 'PZ-4'] 

MEAS_COLUMNS = [ 'T-19', 'TI-3', 'F-19','F-11', 'TI-1213','TI-35']

startPoint = 0
endPoint = len(df)
skip = 12

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


file = './DataDynamo/JablonkaModel/20240208-095601_model_all_data_0_step_60'
gbdt_all_data_0 = joblib.load(file)

historical_forceasts_0 = gbdt_all_data_0.historical_forecasts(
    series=y[TARGETS_clean[0]],  past_covariates=x, start=0 , retrain=False, forecast_horizon=60, show_warnings=False
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
                        title = None,
                        labels = None)