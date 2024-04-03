import pandas as pd
import joblib
from darts import TimeSeries
from darts.metrics import mape, mae, ope
import re

import matplotlib.pyplot as plt
plt.style.reload_library()
plt.style.use('science')
from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'

'''
The things inside the folder models i paper directory in aeml are not models they are forecasts
In the folder data2 in paper directory there are data and transformers but it doesn't work
There are y_connected.pkl and y_scaled are both scaled emissions of both amp and pip. 
Also the pickles in the folder data2 are not so undrestandble because they are already scaled and I cannot scale them back
'''


'''Work with pickles that kevin originaly had in the folder'''
# df_kevin = joblib.load('20210508_df_for_causalimpact.pkl')
# x_transformer_kevin = joblib.load('20210812_x_scaler_reduced_feature_set')
# y_transformer_kevin = joblib.load('20210812_y_transformer_co2_ammonia_reduced_feature_set')

# MEAS_COLUMNS_kevin = [ 'TI-19', 'TI-3', 'FI-19','FI-11', 'TI-1213','TI-35',
#                            "FI-23",
#                            "FI-20",
#                           "FI-20/FI-23",
#                            "TI-22",
#                            "delta_t"]

# TARGETS_clean_kevin = ['2-Amino-2-methylpropanol C4H11NO', 'Piperazine C4H10N2'] 
# TARGETS_clean_kevin = ['Carbon dioxide CO2' , 'Ammonia NH3'] 

# # print(df_kevin.head(1))

# x_kevin = TimeSeries.from_dataframe(df_kevin, value_cols=MEAS_COLUMNS_kevin)
# y_kevin = TimeSeries.from_dataframe(df_kevin, value_cols=TARGETS_clean_kevin)
# original_x_kevin = x_transformer_kevin.inverse_transform(x_kevin)
# original_y_kevin = y_transformer_kevin.inverse_transform(y_kevin)

# print(original_y_kevin[TARGETS_clean_kevin[0]])

# forecasttt = joblib.load('20220311-081020_6_filtered-1-step-target-0-forecasts_quantiles_0.1_0.9.pkl')
# (quantile0, quantile1, quantile2) = forecasttt

'''Working with the model that the predict_w_gbdt made'''
step = 1
# gbdt_all_data_0 = joblib.load('20240222-113644_6_filtered-30-step-target-0-model_quantiles_0.1_0.9.pkl')
gbdt_all_data_0 = joblib.load('20240222-114546_6_filtered-1-step-target-0-model_quantiles_0.1_0.9.pkl')


'''Working with pickle file that kevin jupyter notebook produced'''

# Use the correct step size WRT the model you are choosing
# step = 30

# Models with step=1
# gbdt_all_data_0 = joblib.load('20240207-164125_model_all_data_0_step_1')
# gbdt_all_data_1 = joblib.load('20240207-164125_model_all_data_1_step_1')
# Models with step=30
# gbdt_all_data_0 = joblib.load('20240207-165943_model_all_data_0_step_30')
# gbdt_all_data_1 = joblib.load('20240207-165943_model_all_data_1_step_30')
# Models with step=60
#gbdt_all_data_0 = joblib.load('20240208-095601_model_all_data_0_step_60')
gbdt_all_data_1 = joblib.load('20240208-095601_model_all_data_1_step_60')

x_transformer = joblib.load('20240207-164125_x_transformer')
y_transformer = joblib.load('20240207-164125_y_transformer')


df = pd.read_excel('./WESP on and off only water wash_04-05OCT2022.xlsx', sheet_name='PI-Daten', header=5)

df = df.dropna()

TARGETS_clean = ['AMP-4', 'PZ-4'] 

MEAS_COLUMNS = [ 'T-19', 'TI-3', 'F-19','F-11', 'TI-12','TI-35']

y = TimeSeries.from_dataframe(df, value_cols=TARGETS_clean, time_col='Date')
x = TimeSeries.from_dataframe(df, value_cols=MEAS_COLUMNS, time_col='Date')

x = x_transformer.transform(x)
y = y_transformer.transform(y)

xWESPoff, xWESPon = x.split_before(pd.Timestamp("04-Oct-22 22:00:00"))
yWESPoff, yWESPon = y.split_before(pd.Timestamp("04-Oct-22 22:00:00"))

historical_forceasts_0 = gbdt_all_data_0.historical_forecasts(
    series=y[TARGETS_clean[0]], past_covariates=x, start=0.0, retrain=False, forecast_horizon=step,
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
    mape_score = mape(actual, predicted)
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

ts1 = historical_forceasts_0[1]
ts2 = historical_forceasts_0[2]

temp = TimeSeries.from_series(
    pd.concat([ts1.pd_series(), ts2.pd_series()], axis=1, keys=['ts1', 'ts2'])
)


temp = y_transformer.inverse_transform(temp)


y = y_transformer.inverse_transform(y)

'''Plotting Events'''

Conv_FGD_start = pd.to_datetime('04-Oct-22 14:00:00')
Conv_FGD_end = pd.to_datetime('04-Oct-22 17:56:00')
HP_FGD_WESPOff_start = pd.to_datetime('04-Oct-22 18:00:00')
HP_FGD_WESPOff_end = pd.to_datetime('04-Oct-22 22:00:00')
HP_FGD_WESPON_start = pd.to_datetime('04-Oct-22 22:02:00')
HP_FGD_WESPON_end = pd.to_datetime('05-Oct-22 01:58:00')
HP_FGD_WESPOff2_start = pd.to_datetime('05-Oct-22 02:02:00')
HP_FGD_WESPOff2_end = pd.to_datetime('05-Oct-22 06:00:00')
Conv_FGD2_start = pd.to_datetime('05-Oct-22 06:02:00')
Conv_FGD2_end = pd.to_datetime('05-Oct-22 06:30:00')


plt.axvspan(Conv_FGD_start, Conv_FGD_end, color='blue', alpha=0.3, label = 'conventional FGD, WEPS off')
plt.axvspan(HP_FGD_WESPOff_start, HP_FGD_WESPOff_end, color='red', alpha=0.3, label = 'High perfomance FGD, WESP off')
plt.axvspan(HP_FGD_WESPON_start, HP_FGD_WESPON_end, color='green', alpha=0.3, label = 'High perfomance FGD, WESP on')
plt.axvspan(HP_FGD_WESPOff2_start, HP_FGD_WESPOff2_end, color='red', alpha=0.3)
plt.axvspan(Conv_FGD2_start, Conv_FGD2_end, color='blue', alpha=0.3)

'''Plotting'''

# historical_forceasts_0[1].plot(label = 'Model')
temp['ts1'].plot(label = 'Model')
y[TARGETS_clean[0]].plot(label = 'True values')

'''Plot Information and decoration'''

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

