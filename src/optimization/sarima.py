import mlflow
import time
import pandas as pd
import numpy as np
from typing import Tuple, Optional
from multiprocessing import cpu_count
from joblib import Parallel
from joblib import delayed
from warnings import catch_warnings
from warnings import filterwarnings
from statsmodels.tsa.statespace.sarimax import SARIMAX

from src.pipeline.DataPipelineTS import DataPipelineTS
from src.utils.utils import parse_season
from src.utils.metrics import rmse


def sarima_forecast(series_train: pd.Series, series_test: pd.Series, config: list) -> list:
    """

    :param series_train:
    :param series_test:
    :param config:
    :return:
    """
    order, seasonal_order, trend = config
    # define model
    model = SARIMAX(series_train, order=order, seasonal_order=seasonal_order, trend=trend)
    # fit model
    model_fit = model.fit(disp=False)
    # make forecast
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
    predictions = sarima_forecast(series_train, series_test, cfg)
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
            order, seasonal_order, trend = cfg
            params = {"p_order": order[0], "d_order": order[1], "q_order": order[2],
                      "P_seasonal_order": seasonal_order[0], "D_seasonal_order": seasonal_order[1],
                      "Q_seasonal_order": seasonal_order[2], "trend": trend}
            mlflow.log_params(params)
            mlflow.log_metric("rmse", float(result))
    return key, result


def grid_search(series_train: pd.Series, series_test: pd.Series, cfg_list: list, parallel: bool = False) -> list:
    """
    grid search configs
    :param series_train:
    :param series_test:
    :param cfg_list:
    :param parallel:
    :return:
    """
    if parallel:
        # execute configs in parallel
        executor = Parallel(n_jobs=cpu_count(), backend="multiprocessing")
        tasks = (delayed(score_model)(series_train, series_test, cfg) for cfg in cfg_list)
        scores = executor(tasks)
    else:
        scores = [score_model(series_train, series_test, cfg) for cfg in cfg_list]
    # remove empty results
    scores = [r for r in scores if r[1] is not None]
    # sort configs by error, asc
    scores.sort(key=lambda tup: tup[1])
    return scores


def sarima_configs() -> list:
    """
    create a set of sarima configs to try
    :return: configuration list
    """
    models = list()
    # define config lists
    p_params = [0, 1, 2]
    d_params = [0, 1]
    q_params = [0, 1, 2]
    t_params = ["n", "c", "t", "ct"]
    P_params = [0, 1, 2]
    D_params = [0, 1]
    Q_params = [0, 1, 2]
    m = 24
    # create config instances
    for p in p_params:
        for d in d_params:
            for q in q_params:
                for t in t_params:
                    for P in P_params:
                        for D in D_params:
                            for Q in Q_params:
                                cfg = [(p, d, q), (P, D, Q, m), t]
                                models.append(cfg)
    return models


def main():
    season = parse_season()
    # load dataset
    dp = DataPipelineTS(season=season)
    series_train, series_test = dp()
    # model configs
    cfg_list = sarima_configs()

    client = mlflow.tracking.MlflowClient()
    try:
        # Creates a new experiment
        experiment_id = client.create_experiment(f"SARIMA_{season}")
    except(Exception,):
        # Retrieves the experiment id from the already created project
        experiment_id = client.get_experiment_by_name(f"SARIMA_{season}").experiment_id

    # view results under mlflow ui
    mlflow.set_experiment(experiment_id=experiment_id)
    # grid search
    scores = grid_search(series_train, series_test, cfg_list)
    # list top 3 configs
    for cfg, error in scores[:3]:
        print(cfg, error)


if __name__ == "__main__":
    main()
