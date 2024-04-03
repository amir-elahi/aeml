import pandas as pd
import numpy as np
from statistics_1 import check_stationarity, calculate_and_plot_ACF
import warnings

# warnings.filterwarnings("ignore")

df = pd.read_excel('./WESP on and off only water wash_04-05OCT2022.xlsx', sheet_name='PI-Daten', header=5)

df = df.dropna()

# df.reset_index(inplace=True)

# df = df.iloc[:, :20]
# print(df.info())
# print(df['index'])
calculate_and_plot_ACF(df)
# print(df.shape[1])
