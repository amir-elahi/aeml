import pandas as pd 
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


df = pd.read_excel('./RawData/Previous_campaigns/Vapour Dry Bed 18-21 NOV 2022.xlsx', sheet_name='PI-Daten', header=4)

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
     'TI-12',
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

MEAS_COLUMNS = [ 'T-19', 'TI-3', 'F-19','F-11', 'TI-12','TI-35']


y = TimeSeries.from_dataframe(df, value_cols=TARGETS_clean, time_col='Date')
x = TimeSeries.from_dataframe(df, value_cols=MEAS_COLUMNS, time_col='Date')

x_transformer = Scaler()
x = x_transformer.fit_transform(x)

y_transformer = Scaler()
y = y_transformer.fit_transform(y)

scal = y_transformer.transform(y)

x_train, x_test = x.split_before(pd.Timestamp("20-Nov-22 15:00:00"))
y_train, y_test = y.split_before(pd.Timestamp("20-Nov-22 15:00:00"))


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
step = 30

gbdt_all_data_0 = run_model(x_train, y_train[TARGETS_clean[0]], **settingsAmir, 
                            output_chunk_length=step)

# forecast_0 = gbdt_all_data_0.forecast(n=len(yWESPon[TARGETS_clean[0]]),series = yWESPoff[TARGETS_clean[0]])

# forecast_0 = gbdt_all_data_0.forecast(n=30 ,series = train_yWESPoff[TARGETS_clean[0]], past_covariates = xWESPoff)

historical_forceasts_0 = gbdt_all_data_0.historical_forecasts(
    series=y[TARGETS_clean[0]],  past_covariates=x, start=0, retrain=False, forecast_horizon=step,
        show_warnings = False 
)



'''Calculating metrics'''

def get_metrics(actual, predicted): 
    try:
        actual = TimeSeries.from_series(actual)
        predicted = TimeSeries.from_series(predicted)
    except:
        None

    mae_score = mae(actual, predicted)
    try:
        mape_score = mape(actual, predicted)
    except:
        mape_score = 'Series was not strictly positive'
    ope_score = ope(actual, predicted)
    return {
        'mae': mae_score,
        'mape': mape_score,
        'ope': ope_score
    }

#metrics = get_metrics(val_yWESPoff[TARGETS_clean[0]], forecast_0[1])
metrics = get_metrics(y[TARGETS_clean[0]], historical_forceasts_0[1])


print(metrics)


'''Scale back'''

y_train = y_transformer.inverse_transform(y_train)
y_test = y_transformer.inverse_transform(y_test)

ts1 = historical_forceasts_0[1]
ts2 = historical_forceasts_0[0]

temp = TimeSeries.from_series(
    pd.concat([ts1.pd_series(), ts2.pd_series()], axis=1, keys=['ts1', 'ts2'])
)


temp = y_transformer.inverse_transform(temp)


'''Plotting Events'''

DryBedOff_start = pd.to_datetime('18-Nov-22 00:00:00')
DryBedOff_end = pd.to_datetime('18-Nov-22 13:58:00')
DryBedOn_start = pd.to_datetime('19-Nov-22 03:00:00')
DryBedOn_end = pd.to_datetime('20-Nov-22 02:00:00')
DryBedOff2_start = pd.to_datetime('20-Nov-22 13:00:00')
DryBedOff2_end = pd.to_datetime('21-Nov-22 00:00:00')


plt.axvspan(DryBedOff_start, DryBedOff_end, color='green', alpha=0.3, label = 'Dry Bed Off, Single Water Wash, WESP Off')
plt.axvspan(DryBedOn_start, DryBedOn_end, color='red', alpha=0.3, label = 'Dry Bed On, Acid Wash, WESP Off ')
plt.axvspan(DryBedOff2_start, DryBedOff2_end, color='green', alpha=0.3)

'''Plotting'''

# y[TARGETS_clean[0]].plot(label='True Values')
y_train[TARGETS_clean[0]].plot(label='Train Values')
y_test[TARGETS_clean[0]].plot(label='Validation Values')
# historical_forceasts_0[0].plot(label='Prediction by ML (quantile:0.05)')
# historical_forceasts_0[1].plot(label='Prediction by ML (quantile:0.5)')
# historical_forceasts_0[2].plot(label='Prediction by ML (quantile:0.95)')
# forecast_0[0].plot(label='Prediction by ML (quantile:0.05)')
# forecast_0[1].plot(label='Prediction by ML (quantile:0.5)')
# forecast_0[2].plot(label='Prediction by ML (quantile:0.95)')
temp['ts1'].plot(label='Prediction by ML (quantile:0.5)')

'''Plot Information and decoration'''
# plt.legend(frameon=True)
plt.ylabel(r'Emissions [$\mathrm{mg/nm^3}$]', fontsize=14)
plt.xlabel('Date', fontsize=14)
adsorbent_in_plot = r'2-Amino-2-methylpropanol $\mathrm{C_4H_{11}NO}$'
plt.title( f'{adsorbent_in_plot} with step time {2*step} min\n {metrics}'
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