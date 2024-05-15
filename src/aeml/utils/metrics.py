# export LD_LIBRARY_PATH=/home/amir/miniconda/envs/pyprocessta/lib:$LD_LIBRARY_PATH

from darts import TimeSeries
from darts.metrics import mae, mape, ope, mase
import numpy as np
import pandas as pd

def get_metrics(
        actual,
        predicted,
        train_actual=None,
        intersect=True
) -> dict:
    
    '''
    Evaluate the performance of the model using the following metrics:
    - Mean Absolute Error (MAE)
    - Mean Absolute Percentage Error (MAPE)
    - Overall Percentage Error (OPE)
    - Mean Absolute Scaled Error (MASE)

    Parameters
    ----------
    actual : TimeSeries
        The actual values.
    predicted : TimeSeries
        The predicted values.
    train_actual : TimeSeries, optional
        The actual values of the training data, by default None.
    intersect : bool, optional
        Whether to intersect the actual and predicted values, by default True.

    Returns
    -------
    dict
        A dictionary containing the metrics.
    '''


    try:
        actual = TimeSeries.from_series(actual)
        predicted = TimeSeries.from_series(predicted)
    except:
        None

    mae_score = mae(actual, predicted, intersect)
    
    try:
        mape_score = mape(actual, predicted, intersect)
    except Exception as e:
        mape_score = e

    try:
        ope_score = ope(actual, predicted, intersect)
    except Exception as e:
        ope_score = e

    if train_actual is not None:
        try:
            mase_score = mase( actual_series = actual, 
                              pred_series = predicted, 
                              insample = train_actual, 
                              intersect = intersect)
        except Exception as e:
            mase_score = e
    else:
        mase_score = 'Did not provide the training actual values.'

    return {
        'mae': mae_score,
        'mape': mape_score,
        'ope': ope_score,
        'mase': mase_score
    }

def AE(actual, predicted):

    try:
        actual = TimeSeries.from_series(actual)
        predicted = TimeSeries.from_series(predicted)
    except:
        None
    
    actual_intersect = actual.slice_intersect(predicted)
    predicted_intersect = predicted.slice_intersect(actual)

    AE_values = np.abs(actual_intersect.values() - predicted_intersect.values())

    return TimeSeries.from_times_and_values(actual_intersect.time_index, AE_values)


if __name__ == '__main__':

    date_index = pd.date_range(start='1/1/2020', periods=4, freq='D')
    date_index1 = pd.date_range(start='1/3/2020', periods=4, freq='D')
    actual = TimeSeries.from_times_and_values(date_index,[1, 2, 1, 4])
    predicted = TimeSeries.from_times_and_values(date_index1,[3, 2, 4, 5])

    print(get_metrics(actual, predicted))
    print(AE(actual, predicted))
    print('Done')