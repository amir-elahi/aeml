# import matplotlib.pyplot as plt
# from nixtlats import NixtlaClient
# import pandas as pd

# import sys
# sys.path.insert(0, '/home/lsmo/Desktop/aeml_project/aeml/DataDynamo/Utils2/')
# from Save_and_Load import save_to_pickle, load_from_pickle


# df = pd.read_csv('https://raw.githubusercontent.com/Nixtla/transfer-learning-time-series/main/datasets/electricity-short-with-ex-vars.csv')
# df = df[df['unique_id'] == 'BE'][0:30]
# print(df.head())
# print(df.shape)

# df = df[['ds', 'y']]

# temp = pd.DataFrame()
# temp['ds'] = pd.to_datetime(df['ds'])
# temp = temp.set_index('ds')
# freq = pd.infer_freq(temp.index)
# print(freq)

# # my_api_key = 'nixt-gGOq087YvsJojMkOp6rAW1Yf9BbdcOQzmODvaoRz8nk6hIEAkF9aBDHcANkE5q64suLtzhBVn73EvxOM'

# # nixtla_client = NixtlaClient(
# #     # defaults to os.environ.get("NIXTLA_API_KEY")
# #     api_key = my_api_key
# # )


# # # nixtla_client.validate_api_key()

# # timegpt_fcst_df = nixtla_client.forecast(df=df, h=12, freq=str(freq), time_col='ds', target_col='y')

# # print(timegpt_fcst_df.head())

# location = '/home/lsmo/Desktop/aeml_project/aeml/DataDynamo/Output/TimeGPT/'
# output_file = 'test.pkl'

# # save_to_pickle(timegpt_fcst_df, output_file=output_file, location=location)
# # timegpt_fcst_df = load_from_pickle(input_file=output_file, location=location)
# # timegpt_fcst_df = timegpt_fcst_df[0]
# # print(input_values)

# df['ds'] = pd.to_datetime(df['ds'])
# timegpt_fcst_df['ds'] = pd.to_datetime(timegpt_fcst_df['ds'])
# # # Plot the data
# ax = df.plot(x='ds', y='y', figsize=(12, 6))
# timegpt_fcst_df.plot(x='ds', y='TimeGPT', label='Forecast', ax=ax)
# plt.show()


# # Test the save_to_pickle and load_from_pickle functions
# # import sys
# # sys.path.insert(0, '/home/lsmo/Desktop/aeml_project/aeml/DataDynamo/')
# # from Utils2.Save_and_Load import save_to_pickle, load_from_pickle

# # a = 1234
# # b = {
# #     "c": 123333333
# # }
# # d = [1234,3456, 1234]

# # save_to_pickle(a, b, d, output_file='test.pkl')

# # ass, bs, ds = load_from_pickle('test.pkl')


# from darts.metrics import ae
# from darts import TimeSeries
# import numpy as np

# # Define your time series data
# time_series_1 = TimeSeries.from_values(np.arange(1, 5), np.random.rand(4))
# time_series_2 = TimeSeries.from_values(np.arange(2, 6), np.random.rand(4))

# ae = ae(time_series_1,time_series_2)


import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib as mpl
from matplotlib.font_manager import FontProperties

plt.style.reload_library()
plt.style.use('grid')
plt.rcParams['font.family'] = 'sans-serif'



