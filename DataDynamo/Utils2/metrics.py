# export LD_LIBRARY_PATH=/home/amir/miniconda/envs/pyprocessta/lib:$LD_LIBRARY_PATH

from darts import TimeSeries
from darts.metrics import mae, mape, ope, mase

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
