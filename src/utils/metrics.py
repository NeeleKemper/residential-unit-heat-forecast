import numpy as np
from math import sqrt
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """

    :param y_true:
    :param y_pred:
    :return:
    """
    error = sqrt(mean_squared_error(y_true, y_pred))
    return np.round(error, 2)


def mase(y_train: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, seasonality=1) -> np.ndarray:
    """

    :param y_train:
    :param y_true:
    :param y_pred:
    :param seasonality:
    :return:
    """
    e_t = y_true - y_pred
    scale = mean_absolute_error(y_train[seasonality:], y_train[:-seasonality])
    value = np.mean(np.abs(e_t / scale))
    return np.round(value, 2)


def mase_non_timeseries(y_train: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Mean Absolute Scaled Error for non timeseries (independent observations)
    https://stats.stackexchange.com/questions/108734/alternative-to-mape-when-the-data-is-not-a-time-series/108963#108963
    :param y_train:
    :param y_true:
    :param y_pred:
    :return:
    """
    y_ = np.mean(y_train)
    n = len(y_train)
    e_t = y_true - y_pred
    scale =(np.sum(np.abs(y_train - y_)))/n
    value = np.mean(np.abs(e_t / scale))
    return np.round(value, 2)


def mad(predictions: list) -> np.ndarray:
    """

    :param predictions:
    :return:
    """
    predictions = np.vstack(predictions)
    mad_values = stats.median_abs_deviation(predictions)
    mad_mean = np.mean(mad_values)
    return np.round(mad_mean, 2)
