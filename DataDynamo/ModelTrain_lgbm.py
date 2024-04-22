import pandas as pd 
import numpy as np
from darts import TimeSeries
from darts.models import RegressionModel, LightGBMModel
from darts.metrics import mape, mae, ope
from sklearn.linear_model import BayesianRidge
from aeml.models.gbdt.gbmquantile import LightGBMQuantileRegressor
from aeml.models.gbdt.run import run_ci_model, run_model
from aeml.models.gbdt.settings import *
from aeml.models.gbdt.plot import make_forecast_plot


from aeml.preprocessing.resample import resample_regular
from darts.dataprocessing.transformers import Scaler
import joblib

import matplotlib.pyplot as plt
plt.style.reload_library()
plt.style.use('science')
from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'

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

# =============================================================================
"""

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
skip = 48

y = TimeSeries.from_dataframe(df, value_cols=TARGETS_clean, time_col='Date')
x = TimeSeries.from_dataframe(df, value_cols=MEAS_COLUMNS, time_col='Date')


transformer = Scaler()
x = transformer.fit_transform(x)

y_transformer = Scaler()
y = y_transformer.fit_transform(y)

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

y = Ts3

y_train, y_val = y[:-64] , y[-64:]
x_train, x_val = x[:-64] , x[-64:]


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

#gbdt_all_data_0 = run_model(train_x, train_y[TARGETS_clean[0]], **settingsAmir, output_chunk_length=step)

gbdt_all_data_0 = run_model( x_train, y_train[TARGETS_clean[0]], **settingsAmir, 
                            output_chunk_length = step)

# forecast_0 = gbdt_all_data_0.forecast(n=len(yWESPon[TARGETS_clean[0]]),series = yWESPoff[TARGETS_clean[0]])
# forecast_0 = gbdt_all_data_0.forecast(n=1 ,series = train_yWESPoff[TARGETS_clean[0]])

historical_forceasts_0 = gbdt_all_data_0.historical_forecasts(
    series=y[TARGETS_clean[0]],  past_covariates=x, start=512 , retrain=False, forecast_horizon=step, show_warnings=False
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

'''Calculating metrics'''

def get_metrics(actual, predicted): 
    try:
        actual = TimeSeries.from_series(actual)
        predicted = TimeSeries.from_series(predicted)
    except:
        None

    mae_score = mae(actual, predicted, intersect=True)
    try:
        mape_score = mape(actual, predicted, intersect=True)
    except:
        mape_score = None

    ope_score = ope(actual, predicted, intersect=True)
    return {
        'mae': mae_score,
        'mape': mape_score,
        'ope': ope_score
    }

'''Scale back'''
ts1 = historical_forceasts_0[1]
ts2 = historical_forceasts_0[0]

temp = TimeSeries.from_series(
    pd.concat([ts1.pd_series(), ts2.pd_series()], axis=1, keys=['ts1', 'ts2'])
)


temp = y_transformer.inverse_transform(temp)

metrics = get_metrics(y_transformer.inverse_transform(y)[TARGETS_clean[0]],
                      temp['ts1']
)

print(metrics)


'''Plotting'''

y_transformer.inverse_transform(y_train)[TARGETS_clean[0]].plot(label='Train Values')
y_transformer.inverse_transform(y_val)[TARGETS_clean[0]].plot(label='True Values')
temp['ts1'].plot(label='Prediction by ML')

'''Plot Information and decoration'''
plt.ylabel(r'Emissions [$\mathrm{mg/nm^3}$]', fontsize=14)
plt.xlabel('Date', fontsize=14)
adsorbent_in_plot = r'2-Amino-2-methylpropanol $\mathrm{C_4H_{11}NO}$'
plt.title( f'{adsorbent_in_plot} with step time {10*step*skip/60} min\n {metrics}'
          , fontsize=18, fontweight='extra bold')

# Add a frame around the plot area
plt.gca().spines['top'].set_visible(True)
plt.gca().spines['right'].set_visible(True)
plt.gca().spines['bottom'].set_visible(True)
plt.gca().spines['left'].set_visible(True)

# Adjust font size for tick labels
plt.xticks(fontsize=12)
plt.yticks(fontsize=16)

plt.show()