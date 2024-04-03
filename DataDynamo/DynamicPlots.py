import matplotlib.pyplot as plt
import pandas as pd
from aeml.models import plotting
from darts import TimeSeries
import glob

# dfSc1     = pd.read_excel('./DataDynamo/RawData/New_campaigns/20240301 Scenario 1.xlsx', sheet_name='PI-Daten', header=4).dropna()
# dfSc1AW   = pd.read_excel('./DataDynamo/RawData/New_campaigns/20240302 Scenario 1 with acid wash.xlsx', sheet_name='PI-Daten', header=4).dropna()
# dfSc2AW   = pd.read_excel('./DataDynamo/RawData/New_campaigns/20240304 Scenario 2 with acid wash.xlsx', sheet_name='PI-Daten', header=4).dropna()
# dfSc2Sh   = pd.read_excel('./DataDynamo/RawData/New_campaigns/20240306 Scenario 2 short.xlsx', sheet_name='PI-Daten', header=4).dropna()
# dfSc2AWSh = pd.read_excel('./DataDynamo/RawData/New_campaigns/20240306 Scenario 2 with acid wash short.xlsx', sheet_name='PI-Daten', header=4).dropna()
# dfSc12DB  = pd.read_excel('./DataDynamo/RawData/New_campaigns/20240310 Scenario 2 with dry bed.xlsx', sheet_name='PI-Daten', header=4).dropna()
# dfSc3     = pd.read_excel('./DataDynamo/RawData/New_campaigns/20240307 Scenario 3.xlsx', sheet_name='PI-Daten', header=4).dropna()
# dfSc3DBAW = pd.read_excel('./DataDynamo/RawData/New_campaigns/20240311 Scenario 3 with dry bed and acid wash.xlsx', sheet_name='PI-Daten', header=4).dropna()
# dfSc3DB   = pd.read_excel('./DataDynamo/RawData/New_campaigns/20240311 Scenario 3 with dry bed.xlsx', sheet_name='PI-Daten', header=4).dropna()

# df1  = pd.read_excel('./RawData/New_campaigns/20240311 Scenario 3 with dry bed_v2.xlsx', sheet_name='PI-Daten', header=4).dropna()
# df2  = pd.read_excel('./RawData/New_campaigns/20240301 Scenario 1_v2.xlsx', sheet_name='PI-Daten', header=4).dropna()
# df3  = pd.read_excel('./RawData/New_campaigns/20240313 shutdown and startup with dry bed and double WW_v2.xlsx', sheet_name='PI-Daten', header=4).dropna()
# df4  = pd.read_excel('./RawData/New_campaigns/20240304 Scenario 2 with acid wash_v2.xlsx', sheet_name='PI-Daten', header=4).dropna()
# df5  = pd.read_excel('./RawData/New_campaigns/20240306 Start up_v2.xlsx', sheet_name='PI-Daten', header=4).dropna()
# df6  = pd.read_excel('./RawData/New_campaigns/20240311 Scenario 3 with dry bed and acid wash_v2.xlsx', sheet_name='PI-Daten', header=4).dropna()
# df7  = pd.read_excel('./RawData/New_campaigns/20240306 Scenario 2 with acid wash short_v2.xlsx', sheet_name='PI-Daten', header=4).dropna()
# df8  = pd.read_excel('./RawData/New_campaigns/20240318 Scenario 2 double WW_v2.xlsx', sheet_name='PI-Daten', header=4).dropna()
# df9  = pd.read_excel('./RawData/New_campaigns/20240315 shutdown and startup with dry bed and acid wash_v2.xlsx', sheet_name='PI-Daten', header=4).dropna()
# df10 = pd.read_excel('./RawData/New_campaigns/20240317 shutdown and ramp startup with dry bed_v2.xlsx', sheet_name='PI-Daten', header=4).dropna()
# df11 = pd.read_excel('./RawData/New_campaigns/20240302 Scenario 1 with acid wash_v2.xlsx', sheet_name='PI-Daten', header=4).dropna()
# df12 = pd.read_excel('./RawData/New_campaigns/20240306 Scenario 2 short_v2 .xlsx', sheet_name='PI-Daten', header=4).dropna()
# df13 = pd.read_excel('./RawData/New_campaigns/20240307 Scenario 3 with acid wash_v2.xlsx', sheet_name='PI-Daten', header=4).dropna()
# df14 = pd.read_excel('./RawData/New_campaigns/20240307 Scenario 3_v2.xlsx', sheet_name='PI-Daten', header=4).dropna()
# df15 = pd.read_excel('./RawData/New_campaigns/20240316 shutdown and ramp startup with dry bed and acid wash_v2.xlsx', sheet_name='PI-Daten', header=4).dropna()
# df16 = pd.read_excel('./RawData/New_campaigns/20240314 shutdown and startup with dry bed_v2.xlsx', sheet_name='PI-Daten', header=4).dropna()
# df17 = pd.read_excel('./RawData/New_campaigns/20240318 Scenario 3 double WW_v2.xlsx', sheet_name='PI-Daten', header=4).dropna()
# df18 = pd.read_excel('./RawData/New_campaigns/20240310 Scenario 2 with dry bed_v2.xlsx', sheet_name='PI-Daten', header=4).dropna()
# df19 = pd.read_excel('./RawData/New_campaigns/20240313 Scenario 2 with dry bed and double WW_v2.xlsx', sheet_name='PI-Daten', header=4).dropna()
# df20 = pd.read_excel('./RawData/New_campaigns/20240312 Scenario 3 with dry bed and double WW_v2.xlsx', sheet_name='PI-Daten', header=4).dropna()
# df21 = pd.read_excel('./RawData/New_campaigns/20240311 Scenario 2 with dry bed and acid wash_v2.xlsx', sheet_name='PI-Daten', header=4).dropna()
# df22 = pd.read_excel('./RawData/New_campaigns/20240305 Scenario 2_v2.xlsx', sheet_name='PI-Daten', header=4).dropna()

files = glob.glob('./DataDynamo/RawData/New_campaigns/*.xlsx')
dfs = [pd.read_excel(file, sheet_name='PI-Daten', header=4).dropna() for file in files]

# Concatenate all dataframes
df = pd.concat(dfs)

# Convert the 'date' column to datetime
df['date'] = pd.to_datetime(df['Date'])

# Sort by 'date'
df = df.sort_values('date')

# Reset the index
df = df.reset_index(drop=True)




data = pd.read_excel('./DataDynamo/RawData/New_campaigns/20240301 Scenario 1_v2.xlsx', sheet_name='FIGURES', header=5).dropna()
print(data.columns)
print(data['Unnamed: 1'].values)
asdsadasf
'''
F-3 is the inlet FG flow to the adsorber 
F-11 is the Lean solvent flow rate after the heat exchanger
F-36 is the reboiler condensate flow to the electrical boiler
F-23 is the water from the water wash section to the adsorber
'''
stepped_values = [ 'F-3', 'F-11', 'F-36', 'F-23']


x = TimeSeries.from_dataframe(dfSc1AW, value_cols=stepped_values, time_col='Date')


'''Plotting'''

fig, host = plt.subplots(1,1, figsize=[21,9])
plt.xticks(fontsize=12)
plt.yticks(fontsize=16)

x[stepped_values[0]].plot(label='Inlet FG flow to the adsorber ', color = 'red')
x[stepped_values[1]].plot(label='Lean solvent flow rate after the heat exchanger', color = 'blue')
x[stepped_values[2]].plot(label='Reboiler condensate flow to the electrical boiler', color = 'green')
x[stepped_values[3]].plot(label='Water from the water wash section to the adsorber', color = 'black')


'''Plot Information and decoration'''
# plt.legend(frameon=True)
plt.xlabel('Date', fontsize=14)

# Add a frame around the plot area
plt.gca().spines['top'].set_visible(True)
plt.gca().spines['right'].set_visible(True)
plt.gca().spines['bottom'].set_visible(True)
plt.gca().spines['left'].set_visible(True)


plt.show()

# y1 = [ 1, 2]
# y2 = [10 , 100]
# y3 = [2, 5]

# # y = [y1 , y2, y3]
# plotting.make_dynamic_plot(y[MEAS_COLUMNS[0:2]])
# plt.show()
