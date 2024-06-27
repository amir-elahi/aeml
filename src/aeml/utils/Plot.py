import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib as mpl
from matplotlib.font_manager import FontProperties

plt.style.reload_library()
plt.style.use('grid')
plt.rcParams['font.family'] = 'sans-serif'

from aeml.utils.metrics import AE
import numpy as np
import pandas as pd
from darts import TimeSeries
from typing import List
import os

#! Script for scenario plot (Figure A1) and  event plot (Figure A2)


def plot_historical_forecast(df, 
                            forecast,
                            target_col,
                            title = None,
                            lower_percentile = None,
                            higher_percentile = None,
                            output_Path = './DataDynamo/Plots',
                            output_Name = None,
                            labels = ['Actual', 'Forecast'],
                            ylabel = r'Emissions $[\mathrm{mg/nm^3}]$',
                            ShowEvent = False,
) -> None:
    """
    Plot the historical forecast generated by ModelTrain_Chronos.py.
    
    Parameters
    ----------
    df : pd.DataFrame
        The historical data that is the ground truth.
    forecast : pd.DataFrame
        The forecasted data.
    target_col : str
        The target column in the data.
    time_col : str
        The time column in the data.
    title : str, optional
        The title of the plot, by default None
    lower_percentile : pd.DataFrame, optional
        The lower percentile of the forecast, by default None
    higher_percentile : pd.DataFrame, optional
        The higher percentile of the forecast, by default None
    output_Path : str, optional
        The path where the output will be saved, by default './DataDynamo/Plots'
    output_Name : str, optional
        The name of the output file, by default None. If None, the plot will be shown and not saved.
        
    Returns
    -------
    None
    """

    if not os.path.exists(output_Path):
        os.makedirs(output_Path)

    plt.figure(figsize=(4.9, 2.1))

    if ShowEvent:
            
        event_list = [
            [(pd.Timestamp('2024-03-05 07:00:00'), pd.Timestamp('2024-03-05 11:00:00'))],
            [(pd.Timestamp('2024-03-06 09:00:00'), pd.Timestamp('2024-03-06 10:00:00'))],
            [(pd.Timestamp('2024-03-07 09:20:00'), pd.Timestamp('2024-03-07 10:30:00'))],
    ]
        
        for event in event_list:
            start, end = event[0]
            plt.axvspan(start, end, color='red', alpha=0.3)
            plt.axvline(start, color='red', linestyle='--', lw=0.3)
            plt.axvline(end, color='red', linestyle='--', lw=0.3)

    plt.plot(df[target_col].index, df[target_col].values, label=labels[0] if labels is not None else None, color='black', lw=0.8)
    
    if forecast is not None:
        plt.plot(forecast[target_col].index, forecast[target_col].values, label=labels[1] if labels is not None else None, color='blue', lw=0.7)

    if lower_percentile is not None and higher_percentile is not None:
        plt.fill_between(forecast[target_col].index,
                        lower_percentile, 
                        higher_percentile,
                        color='blue', 
                        alpha=0.2)
    
    '''Plot Information and decoration'''
    # =============================================================================
    # Setting the font properties
    fpLegend = './DataDynamo/Fonts/calibri-regular.ttf'
    fpLegendtitle = './DataDynamo/Fonts/coolvetica rg.otf'
    fpTitle = './DataDynamo/Fonts/coolvetica rg.otf'
    fpLabel = './DataDynamo/Fonts/Philosopher-Bold.ttf'
    fpTicks = './DataDynamo/Fonts/Philosopher-Regular.ttf'

    fLegend = FontProperties(fname=fpLegend)
    fLegendtitle = FontProperties(fname=fpLegendtitle)
    fTitle = FontProperties(fname=fpTitle)
    fLabel = FontProperties(fname=fpLabel)
    fTicks = FontProperties( fname=fpTicks)

    # =============================================================================
    # Add labels and title and ticks
    plt.ylabel(ylabel, fontproperties = fLabel)
    plt.xlabel('Time', fontproperties = fLabel)

    if title is not None:
        plt.title(title, fontproperties = fTitle)

    for label in (plt.gca().get_xticklabels() + plt.gca().get_yticklabels()):
        label.set_fontproperties(fTicks)

    # =============================================================================
    # Add a frame around the plot area
    plt.gca().spines['top'].set_visible(True)
    plt.gca().spines['right'].set_visible(True)
    plt.gca().spines['bottom'].set_visible(True)
    plt.gca().spines['left'].set_visible(True)

    # =============================================================================
    # Add a legend
    legend = plt.legend(prop = fLegend)

    # =============================================================================
    # Set the date format
    date_format = mdates.DateFormatter('%b-%d')
    plt.gca().xaxis.set_major_formatter(date_format)

    # =============================================================================
    # Adjust font size for tick labels
    plt.xticks(rotation='vertical', fontproperties = fTicks)
    plt.yticks(fontproperties = fTicks)
    # plt.ylim(0,200)
    # =============================================================================
    plt.legend(frameon=False) if labels is None else None
    plt.tight_layout()

    if output_Name is not None:
        plt.savefig(output_Path + '/' + output_Name + '.pdf', bbox_inches='tight')

    plt.show()
    
    return None


def make_MAE_error_plot(time,
                        error,
                        title=None,
                        output_Path='./DataDynamo/Plots',
                        output_Name=None,
                        error_type='Mean Absolute Error [MAE]',
                        log_scale=False
) -> None:
    """
    Plot the Mean Absolute Error (MAE) over time.
    
    Parameters
    ----------
    time : list or array-like
        The time values.
    error : list or array-like
        The MAE values.
    title : str, optional
        The title of the plot, by default None. When None, no title is added.
    output_Path : str, optional
        The path where the output will be saved, by default './DataDynamo/Plots'
    output_Name : str, optional
        The name of the output file, by default None. If None, the plot will be shown and not saved.
    error_type : str, optional
        The type of error, by default 'Mean Absolute Error [MAE]'.
    log_scale : bool, optional
        Whether to plot the MAE in log scale, by default False.
        
    Returns
    -------
    None
    """

    if not os.path.exists(output_Path):
        os.makedirs(output_Path)
        
    plt.figure(figsize=(7, 3))
    plt.plot(time, error, label=error_type, color='red', lw=0.8)
    
    if log_scale:
        ax = plt.gca()
        ax.set_xscale('log')
    
    '''Plot Information and decoration'''
    plt.rcParams['font.family'] = 'sans-serif'

    fpLegend = './DataDynamo/Fonts/calibri-regular.ttf'
    fpLegendtitle = './DataDynamo/Fonts/coolvetica rg.otf'
    fpTitle = './DataDynamo/Fonts/coolvetica rg.otf'
    fpLabel = './DataDynamo/Fonts/Philosopher-Bold.ttf'
    fpTicks = './DataDynamo/Fonts/Philosopher-Regular.ttf'

    fLegend = FontProperties(fname=fpLegend)
    fLegendtitle = FontProperties(fname=fpLegendtitle)
    fTitle = FontProperties(fname=fpTitle)
    fLabel = FontProperties(fname=fpLabel)
    fTicks = FontProperties( fname=fpTicks)
    # Add labels and title and ticks
    plt.ylabel('Mean Absolute Error [MAE]', fontproperties=fLabel)
    plt.xlabel('Time', fontproperties=fLabel)

    if title is not None:
        plt.title(title, fontproperties=fTitle)


    for label in (plt.gca().get_xticklabels() + plt.gca().get_yticklabels()):
        label.set_fontproperties(fTicks)
    
    # Add a frame around the plot area
    plt.gca().spines['top'].set_visible(True)
    plt.gca().spines['right'].set_visible(True)
    plt.gca().spines['bottom'].set_visible(True)
    plt.gca().spines['left'].set_visible(True)
    
    # Add a legend
    legend = plt.legend(prop=fLegend)
    legend.set_title('Legend', prop=fLegendtitle)
    
    # Adjust font size for tick labels
    plt.xticks(rotation='vertical', fontproperties=fTicks)
    plt.yticks(fontproperties=fTicks)
    
    plt.tight_layout()
    
    if output_Name is not None:
        plt.savefig(output_Path + '/' + output_Name + '.pdf', bbox_inches='tight')
    
    plt.show()
    
    return None

def make_ae_error_plot(actual: List[TimeSeries], 
                       predicted: List[TimeSeries], 
                       time_horizon: List[str],
                       title: str = None,
                       output_Path: str ='./DataDynamo/Plots', 
                       output_Name: str = None, 
                       error_type: str ='Absolute Error [AE]', 
                       log_scale: bool = False,
                       Violin: bool = True,
                       Box: bool = False,
                       showfliers: bool = False,
                       y_lim: tuple = (0, 110),
                       x_label: str = 'Forecasting time horizon [hours]'
                       ):
    
    if not os.path.exists(output_Path):
        os.makedirs(output_Path)
    
    if Box and Violin or not Box and not Violin:
        raise ValueError('Please select either Box or Violin plot')
    
    if not isinstance(actual, list) or not isinstance(predicted, list):
        try:
            actual = list(actual)
            predicted = list(predicted)
        except Exception as e:
            raise ValueError(f'Could not convert the input to a list: {e}')

    
    error_ts_list = [AE(actual_ts, predicted_ts) for actual_ts, predicted_ts in zip(actual, predicted)]
    error_values_list = [error_ts.values() for error_ts in error_ts_list]
    # error_values_array = np.array(error_values_list)
    # error_values_array = error_values_array.reshape(error_values_array.shape[:-1]).T


    # Create subplots for each time horizon
    plt.figure(figsize=(4.9, 2.1*2))

    if Violin:
    # Use list comprehension to plot each sublist separately
        [plt.violinplot(sublist, showmeans=True, showmedians=True, positions=[i]) for i, sublist in enumerate(error_values_list)]
    elif Box:
    # Use list comprehension to plot each sublist separately
        [plt.boxplot(sublist, widths = 0.5, showfliers=showfliers, showmeans=True, meanline=True, positions=[i]) for i, sublist in enumerate(error_values_list)]

    '''Plot Information and decoration'''
    plt.rcParams['font.family'] = 'sans-serif'

    fpLegend = './DataDynamo/Fonts/calibri-regular.ttf'
    fpLegendtitle = './DataDynamo/Fonts/coolvetica rg.otf'
    fpTitle = './DataDynamo/Fonts/coolvetica rg.otf'
    fpLabel = './DataDynamo/Fonts/Philosopher-Bold.ttf'
    fpTicks = './DataDynamo/Fonts/Philosopher-Regular.ttf'

    fLegend = FontProperties(fname=fpLegend)
    fLegendtitle = FontProperties(fname=fpLegendtitle)
    fTitle = FontProperties(fname=fpTitle)
    fLabel = FontProperties(fname=fpLabel)
    fTicks = FontProperties( fname=fpTicks)
    # Add labels and title and ticks
    plt.ylabel('Absolute Error [AE]', fontproperties=fLabel)
    plt.xlabel(x_label, fontproperties=fLabel)

    if title is not None:
        plt.title(title, fontproperties=fTitle)


    for label in (plt.gca().get_xticklabels() + plt.gca().get_yticklabels()):
        label.set_fontproperties(fTicks)
    
    # Add a frame around the plot area
    plt.gca().spines['top'].set_visible(True)
    plt.gca().spines['right'].set_visible(True)
    plt.gca().spines['bottom'].set_visible(True)
    plt.gca().spines['left'].set_visible(True)



    # Create custom legend entries
    if Box:
        median_color = mpl.rcParams['boxplot.medianprops.color']
        median_linestyle = mpl.rcParams['boxplot.medianprops.linestyle']

        mean_color = mpl.rcParams['boxplot.meanprops.color']
        mean_linestyle = mpl.rcParams['boxplot.meanprops.linestyle']

        median_line = mlines.Line2D([], [], color=median_color, linestyle=median_linestyle, label='Median')
        mean_line = mlines.Line2D([], [], color=mean_color, linestyle=mean_linestyle, label='Mean')

        # Add the legend to the plot
        plt.legend(handles=[median_line, mean_line], prop = fLegend)
    
    # Adjust font size for tick labels
    plt.xticks(range(0, len(time_horizon)), time_horizon)  # Adjusted xticks range
    plt.xticks(fontproperties=fTicks)
    plt.yticks(fontproperties=fTicks)
    plt.ylim(y_lim)
    plt.grid(axis='x')

    plt.tight_layout()
    
    if output_Name is not None:
        plt.savefig(output_Path + '/' + output_Name + '.pdf', bbox_inches='tight')
    
    plt.show()

    return None


def make_scenario_plot():
    
    # Your base data

    # Your scenarios
    data_corrected = {
        "Scenario 1": list(range(100, 70, -2)) + [70] * 120 + list(range(70, 101, 2)),
        "Scenario 2": list(range(100, 50, -5)) + [50] * 120 + list(range(50, 101, 5)),
        "Scenario 3": list(range(100, 50, -5)) + [50] * 15 + list(range(50, 101, 5)),
        "Scenario 4": list(range(100, 50, -5)) + list(range(50, 80, 5)) + list(range(80, 50, -2)) + list(range(50, 101, 5))

}

    # Time data
    time_data = {
        "Scenario 1" : list(range(len(data_corrected["Scenario 1"]))),
        "Scenario 2" : list(range(len(data_corrected["Scenario 2"]))),
        "Scenario 3" : list(range(len(data_corrected["Scenario 3"]))),
        "Scenario 4" : list(range(len(data_corrected["Scenario 4"]))),
    }

    # List to hold DataFrames
    dataframes_list = []

    # Iterate through each scenario to create a DataFrame and append it to the list
    for scenario in data_corrected:
        df = pd.DataFrame({
            "Time [Minute]": time_data[scenario],
            scenario: data_corrected[scenario]
        })
        dataframes_list.append(df)


    # Plotting the data in three subplots horizontally
    fig, axs = plt.subplots(2, 2, figsize=(4.9*1.3, 2.1*2), sharey=True)

    # Plot for Scenario 1
    df_corrected = dataframes_list[0]
    axs[0,0].plot(df_corrected["Time [Minute]"], df_corrected["Scenario 1"], label="Scenario 1", color='orange')
    axs[0,0].grid(True)

    # Plot for Scenario 2
    df_corrected = dataframes_list[1]
    axs[0,1].plot(df_corrected["Time [Minute]"], df_corrected["Scenario 2"], label="Scenario 2", color='blue')
    axs[0,1].grid(True)

    # Plot for Scenario 3
    df_corrected = dataframes_list[2]
    axs[1,0].plot(df_corrected["Time [Minute]"], df_corrected["Scenario 3"], label="Scenario 3", color='red')
    axs[1,0].grid(True)

    # Plot for Scenario 4
    df_corrected = dataframes_list[3]
    axs[1,1].plot(df_corrected["Time [Minute]"], df_corrected["Scenario 4"], label="Scenario 4", color='green')
    axs[1,1].grid(True)

    '''Plot Information and decoration'''
    plt.rcParams['font.family'] = 'sans-serif'

    fpTitle = './DataDynamo/Fonts/coolvetica rg.otf'
    fpLabel = './DataDynamo/Fonts/Philosopher-Bold.ttf'

    fTitle = FontProperties(fname=fpTitle)
    fLabel = FontProperties(fname=fpLabel)

    # Add labels and title and ticks
    axs[0,0].set_ylabel('Load [%]', fontproperties=fLabel)
    axs[1,0].set_ylabel('Load [%]', fontproperties=fLabel)
    for i in range(2):
        for j in range(2):
            ax = axs[i, j]
            ax.set_xlabel('Time [Minute]', fontproperties=fLabel)
            # Add a frame around the plot area
            ax.spines['top'].set_visible(True)
            ax.spines['right'].set_visible(True)
            ax.spines['bottom'].set_visible(True)
            ax.spines['left'].set_visible(True)

    axs[0,0].set_title('Dynamic Scenario 1 (2%/Minute)', fontproperties=fTitle)
    axs[0,1].set_title('Dynamic Scenario 2 (5%/Minute)', fontproperties=fTitle)
    axs[1,0].set_title('Dynamic Scenario 3 (5%/Minute)', fontproperties=fTitle)
    axs[1,1].set_title('Dynamic Scenario 4 (Double Load Change)', fontproperties=fTitle)
    
    # Display the plot
    plt.tight_layout()
    plt.show()

def make_event_plot(df, event_list, target_col):

    plt.figure(figsize=(4.9*1.5, 2.1*1.3))

    # Ensure 'Date' is datetime and target_col is numeric
    df['Date'] = pd.to_datetime(df['Date'])
    df[target_col] = pd.to_numeric(df[target_col], errors='coerce')

    plt.plot(df['Date'], df[target_col], color='black', lw=0.8)

    for event in event_list:
        start, end = event[0]
        plt.axvspan(start, end, color='red', alpha=0.3)
        plt.axvline(start, color='red', linestyle='--', lw=0.3)
        plt.axvline(end, color='red', linestyle='--', lw=0.3)

    '''Plot Information and decoration'''
    plt.rcParams['font.family'] = 'sans-serif'

    fpTitle = './DataDynamo/Fonts/coolvetica rg.otf'
    fpLabel = './DataDynamo/Fonts/Philosopher-Bold.ttf'
    fpTicks = './DataDynamo/Fonts/Philosopher-Regular.ttf'

    fLabel = FontProperties(fname=fpLabel)
    fTicks = FontProperties( fname=fpTicks)
    # Add labels and title and ticks
    plt.ylabel(r'Emissions $[\mathrm{mg/nm^3}]$', fontproperties=fLabel)
    plt.xlabel('Time', fontproperties=fLabel)

        # =============================================================================
    # Set the date format
    date_format = mdates.DateFormatter('%b-%d')
    plt.gca().xaxis.set_major_formatter(date_format)


    for label in (plt.gca().get_xticklabels() + plt.gca().get_yticklabels()):
        label.set_fontproperties(fTicks)
    
    # Add a frame around the plot area
    plt.gca().spines['top'].set_visible(True)
    plt.gca().spines['right'].set_visible(True)
    plt.gca().spines['bottom'].set_visible(True)
    plt.gca().spines['left'].set_visible(True)
    
    
    # Adjust font size for tick labels
    plt.xticks(rotation='vertical', fontproperties=fTicks)
    plt.yticks(fontproperties=fTicks)
    
    plt.tight_layout()
    
    plt.show()




if __name__ == '__main__':
    
    # True for scenario plot (Figure A1) and False for event plot (Figure A2)
    scenario_plot = False

    ##############################################################
    if scenario_plot:
        make_scenario_plot()

    else: 
        ##############################################################
        df = pd.read_pickle('./DataDynamo/RawData/New_campaigns/202403 SCOPE data set dynamic campaign.pkl')

        df = df.dropna()

        df['TI-1213'] = np.where(df['valve position'] == 1, df['TI-13'], df['TI-12'])

        TARGETS_clean = ['AMP-4', 'PZ-4'] 

        MEAS_COLUMNS = [ 'T-19', 'TI-3', 'F-19','F-11', 'TI-1213','TI-35']

        event_list = [
            [(pd.Timestamp('2024-03-01 09:30:00'), pd.Timestamp('2024-03-01 12:00:00'))],
            [(pd.Timestamp('2024-03-05 07:00:00'), pd.Timestamp('2024-03-05 11:00:00'))],
            [(pd.Timestamp('2024-03-06 09:00:00'), pd.Timestamp('2024-03-06 10:00:00'))],
            [(pd.Timestamp('2024-03-07 09:20:00'), pd.Timestamp('2024-03-07 10:30:00'))],
            [(pd.Timestamp('2024-03-18 15:00:00'), pd.Timestamp('2024-03-18 17:30:00'))],
            [(pd.Timestamp('2024-03-18 21:45:00'), pd.Timestamp('2024-03-18 22:45:00'))],
            [(pd.Timestamp('2024-03-02 08:50:00'), pd.Timestamp('2024-03-02 11:30:00'))],
            [(pd.Timestamp('2024-03-04 09:00:00'), pd.Timestamp('2024-03-04 11:30:00'))],
            [(pd.Timestamp('2024-03-07 17:00:00'), pd.Timestamp('2024-03-07 17:50:00'))],
            [(pd.Timestamp('2024-03-10 18:40:00'), pd.Timestamp('2024-03-10 21:10:00'))],
            [(pd.Timestamp('2024-03-11 01:10:00'), pd.Timestamp('2024-03-11 02:00:00'))],
            [(pd.Timestamp('2024-03-13 08:45:00'), pd.Timestamp('2024-03-13 11:11:00'))],
            [(pd.Timestamp('2024-03-12 00:12:00'), pd.Timestamp('2024-03-12 01:00:00'))],
            [(pd.Timestamp('2024-03-11 16:00:00'), pd.Timestamp('2024-03-11 18:25:00'))],
            [(pd.Timestamp('2024-03-11 22:45:00'), pd.Timestamp('2024-03-11 23:35:00'))],

        ]

        make_event_plot(df, event_list, TARGETS_clean[0])