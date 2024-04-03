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

df = pd.read_excel('./RawData/Previous_campaigns/WESP on and off only water wash_04-05OCT2022.xlsx', sheet_name='PI-Daten', header=5)

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
    'TI-12/TI-13',	
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

train_x, val_x = x.split_before(pd.Timestamp("05-Oct-22 07:00:00"))
train_y, val_y = y.split_before(pd.Timestamp("05-Oct-22 07:00:00"))

xWESPoff, xWESPon = x.split_before(pd.Timestamp("04-Oct-22 22:00:00"))
yWESPoff, yWESPon = y.split_before(pd.Timestamp("04-Oct-22 22:00:00"))

train_xWESPoff, val_xWESPoff = xWESPoff.split_before(pd.Timestamp("04-Oct-22 21:00:00"))
train_yWESPoff, val_yWESPoff = yWESPoff.split_before(pd.Timestamp("04-Oct-22 21:00:00"))

# print(len(yWESPoff))


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
step = 30

#gbdt_all_data_0 = run_model(train_x, train_y[TARGETS_clean[0]], **settingsAmir, output_chunk_length=step)

gbdt_all_data_0 = run_model(train_xWESPoff, train_yWESPoff[TARGETS_clean[0]], **settingsAmir, 
                            output_chunk_length=step)

# forecast_0 = gbdt_all_data_0.forecast(n=len(yWESPon[TARGETS_clean[0]]),series = yWESPoff[TARGETS_clean[0]])
# forecast_0 = gbdt_all_data_0.forecast(n=1 ,series = train_yWESPoff[TARGETS_clean[0]])

historical_forceasts_0 = gbdt_all_data_0.historical_forecasts(
    series=y[TARGETS_clean[0]],  past_covariates=x, start=0 , retrain=False, forecast_horizon=step,
        show_warnings = False 
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

    mae_score = mae(actual, predicted)
    mape_score = mape(actual, predicted)
    ope_score = ope(actual, predicted)
    return {
        'mae': mae_score,
        'mape': mape_score,
        'ope': ope_score
    }

metrics = get_metrics(y[TARGETS_clean[0]]+1e-20, historical_forceasts_0[1])

print(metrics)


'''Plotting'''

WESPonDate = pd.to_datetime('04-Oct-22 22:00:00')
WESPoffDate = pd.to_datetime('05-Oct-22 02:02:00')

plt.axvline(x=WESPonDate, color='red', linestyle='--', label='WESP on')
plt.axvline(x=WESPoffDate, color='blue', linestyle='--', label='WESP off')

True_values = val_yWESPoff.append(yWESPon)
train_yWESPoff[TARGETS_clean[0]].plot(label='Train Values')
True_values[TARGETS_clean[0]].plot(label='True Values')
# train_yWESPoff[TARGETS_clean[0]].plot(label='Train Values')
# val_yWESPoff[TARGETS_clean[0]].plot(label='Validation Values')
historical_forceasts_0[1].plot(label='Prediction by ML')
# forecast_0[0].plot(label='Base value by ML 0.05')
# forecast_0[1].plot(label='Base value by ML 0.5')
# forecast_0[2].plot(label='Base value by ML 0.95')

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