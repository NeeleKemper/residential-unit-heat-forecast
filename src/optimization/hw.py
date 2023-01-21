import mlflow
import time
import pandas as pd
import numpy as np
from warnings import catch_warnings
from warnings import filterwarnings
from typing import Tuple, Optional
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from src.utils.utils import parse_season
from src.pipeline.DataPipelineTS import DataPipelineTS
from src.utils.metrics import rmse


def hw_forecast(series_train: pd.Series, series_test: pd.Series, config: list) -> list:
    """

    :param series_train:
    :param series_test:
    :param config:
    :return:
    """
    trend, seasonal = config
    # define model
    model = ExponentialSmoothing(series_train, seasonal_periods=24, trend=trend, seasonal=seasonal)
    # fit model
    model_fit = model.fit()
    # make one step forecasts
    y_hat = model_fit.forecast(steps=int(len(series_test)))
    return y_hat


def validation(series_train: pd.Series, series_test: pd.Series, cfg: list) -> np.ndarray:
    """
    validation for univariate data
    :param series_train:
    :param series_test:
    :param cfg:
    :return:
    """
    predictions = hw_forecast(series_train, series_test, cfg)
    # estimate prediction error
    error = rmse(y_true=np.array(series_test), y_pred=np.array(predictions))
    return error


def score_model(series_train: pd.Series, series_test: pd.Series, cfg: list, debug: bool = False) -> \
        Tuple[str, Optional[float]]:
    """
    score a model, return None on failure
    :param series_train:
    :param series_test:
    :param cfg:
    :param debug:
    :return:
    """
    run_name = int(time.time())
    with mlflow.start_run(run_name=str(run_name)):
        result = None
        # convert config to a key
        key = str(cfg)
        # show all warnings and fail on exception if debugging
        if debug:
            result = validation(series_train, series_test, cfg)
        else:
            # one failure during model validation suggests an unstable config
            try:
                # never show warnings when grid searching, too noisy
                with catch_warnings():
                    filterwarnings("ignore")
                    result = validation(series_train, series_test, cfg)
            except(Exception,):
                print("error occured")
        # check for an interesting result
        if result is not None:
            print(" > Model[%s] %.3f" % (key, float(result)))
            trend, seasonal = cfg
            params = {"trend": trend, "seasonal": seasonal}
            mlflow.log_params(params)
            mlflow.log_metric("rmse", float(result))
    return key, result


def grid_search(series_train: pd.Series, series_test: pd.Series, cfg_list: list) -> list:
    """
    grid search configs
    :param series_train:
    :param series_test:
    :param cfg_list:
    :return:
    """
    scores = [score_model(series_train, series_test, cfg) for cfg in cfg_list]
    # remove empty results
    scores = [r for r in scores if r[1] is not None]
    # sort configs by error, asc
    scores.sort(key=lambda tup: tup[1])
    return scores


def holt_winters_configs() -> list:
    """

    :return:
    """
    models = list()
    trend = ["additive", "multiplicative", None]
    seasonal = ["additive", "multiplicative", None]
    for t in trend:
        for s in seasonal:
            cfg = [t, s]
            models.append(cfg)
    return models


def main():
    season = parse_season()
    # load dataset
    dp = DataPipelineTS(season=season)
    series_train, series_test = dp()
    # model configs
    cfg_list = holt_winters_configs()

    client = mlflow.tracking.MlflowClient()
    try:
        # Creates a new experiment
        experiment_id = client.create_experiment(f"HW_{season}")
    except(Exception,):
        # Retrieves the experiment id from the already created project
        experiment_id = client.get_experiment_by_name(f"HW_{season}").experiment_id
    # view results under mlflow ui
    mlflow.set_experiment(experiment_id=experiment_id)
    # grid search
    scores = grid_search(series_train, series_test, cfg_list)
    # list top 3 configs
    for cfg, error in scores[:3]:
        print(cfg, error)


if __name__ == "__main__":
    main()
