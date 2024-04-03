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

import wandb

wandb.login()

''' Sweep Configuration for wandb'''

step = 60
import time
timestr = time.strftime("%Y%m%d-%H%M%S")

sweep_config = {
    "metric": {"goal": "minimize", "name": "mae_value"},
    "name": f"Sweep_Time_{timestr}_step_{step}",
    "method": "random",
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

sweep_id = wandb.sweep(sweep_config, project="aeml_amir")


def objective(config):

    ''' Reading the data from the excel file'''

    df = pd.read_excel('./RawData/Previous_campaigns/WESP on and off only water wash_04-05OCT2022.xlsx', sheet_name='PI-Daten', header=5)

    df = df.dropna()

    TARGETS_clean = ['AMP-4', 'PZ-4'] 


    MEAS_COLUMNS = [ 'T-19', 'TI-3', 'F-19','F-11', 'TI-12','TI-35']

    y = TimeSeries.from_dataframe(df, value_cols=TARGETS_clean, time_col='Date')
    x = TimeSeries.from_dataframe(df, value_cols=MEAS_COLUMNS, time_col='Date')

    # y = TimeSeries.from_dataframe(df, value_cols=TARGETS_clean)
    # x = TimeSeries.from_dataframe(df, value_cols=MEAS_COLUMNS)

    transformer = Scaler()
    x = transformer.fit_transform(x)

    y_transformer = Scaler()
    y = y_transformer.fit_transform(y)

    scal = y_transformer.transform(y)

    xWESPoff, xWESPon = x.split_before(pd.Timestamp("04-Oct-22 22:00:00"))
    yWESPoff, yWESPon = y.split_before(pd.Timestamp("04-Oct-22 22:00:00"))

    train_xWESPoff, val_xWESPoff = xWESPoff.split_before(pd.Timestamp("04-Oct-22 21:00:00"))
    train_yWESPoff, val_yWESPoff = yWESPoff.split_before(pd.Timestamp("04-Oct-22 21:00:00"))
    # print(len(yWESPoff))


    ######################################

    # gbdt_all_data_0 = run_ci_model(xWESPoff, yWESPoff[TARGETS_clean[0]], **ci_6_0,
    #                                 output_chunk_length=100, num_features= 7 )
    # gbdt_all_data_1 = run_model(xWESPoff, yWESPoff[TARGETS_clean[1]], **settings_1_1, output_chunk_length=1)


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


    gbdt_all_data_0 = run_model(train_xWESPoff, train_yWESPoff[TARGETS_clean[0]], **settingsAmir, 
                                output_chunk_length=step)

    # forecast_0 = gbdt_all_data_0.forecast(n=len(yWESPon[TARGETS_clean[0]]),series = yWESPoff[TARGETS_clean[0]])
    forecast_0 = gbdt_all_data_0.forecast(n=step ,series = train_yWESPoff[TARGETS_clean[0]], past_covariates = xWESPoff)

    # historical_forceasts_0 = gbdt_all_data_0.historical_forecasts(
    #     series=yWESPoff[TARGETS_clean[0]],  past_covariates=xWESPoff,start=0.7375, retrain=False, forecast_horizon=30,
    #         stride=1,
    # )

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

    ''' Calculating metrics'''

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

    metrics = get_metrics(val_yWESPoff[TARGETS_clean[0]], forecast_0[1])

    mae_value = metrics['mae']

    return mae_value

def main():
    wandb.init(project="aeml_amir")
    mae_value = objective(wandb.config)
    wandb.log({
        "mae_value": mae_value,
        "step": step
    })

wandb.agent(sweep_id, function=main, count=1000)