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
    except:
        mape_score = 'Could not calculate'

    ope_score = ope(actual, predicted, intersect)

    if train_actual is not None:
        mase_score = mase(actual, predicted, train_actual, intersect)
    else:
        mase_score = 'Could not calculate'

    return {
        'mae': mae_score,
        'mape': mape_score,
        'ope': ope_score,
        'mase': mase_score
    }
