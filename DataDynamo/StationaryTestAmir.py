import pandas as pd
import numpy as np
from statistics_1 import check_stationarity
import warnings

warnings.filterwarnings("ignore")

df = pd.read_excel('./WESP on and off only water wash_04-05OCT2022.xlsx', sheet_name='PI-Daten', header=5)

df = df.dropna()

# stationary_test_result = check_stationarity(series=df['TI-3'])
# print(stationary_test_result['adf']['stationary'] == False)

for column in df.columns:
    stationary_test_result = check_stationarity(series=df[column])

    # print(df.iloc[0,0:4])
    # print(df.info())
    if (stationary_test_result['adf']['stationary'] == True or stationary_test_result['kpss']['stationary'] == True):
        print(column, end='\n')
        print(stationary_test_result)
        print('-' * 50)