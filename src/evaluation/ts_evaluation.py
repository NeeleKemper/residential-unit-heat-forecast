import mlflow
from typing import Any
from warnings import catch_warnings
from warnings import filterwarnings

from src.pipeline.DataPipelineTS import DataPipelineTS
from src.utils.utils import watt_into_kilowatt
from src.utils.metrics import rmse, mase


def time_series_experiment(model, experiment_name: str, season: str) -> None:
    """

    :param model:
    :param experiment_name:
    :param season:
    :return:
    """
    client = mlflow.tracking.MlflowClient()
    try:
        # Creates a new experiment
        experiment_id = client.create_experiment(experiment_name)
    except(Exception,):
        # Retrieves the experiment id from the already created project
        experiment_id = client.get_experiment_by_name(experiment_name).experiment_id

    # view results under mlflow ui
    mlflow.set_experiment(experiment_id=experiment_id)

    with catch_warnings():
        filterwarnings("ignore")
        model_fit = model.fit()

    rmse, mase = time_series_evaluation(model_fit, season)

    if season == "all":
        rmse_value, rmse_value_summer, rmse_value_winter = rmse
        mase_value, mase_value_summer, mase_value_winter = mase
        mlflow.log_metric("rmse_summer", float(rmse_value_summer))
        mlflow.log_metric("mase_summer", float(mase_value_summer))
        mlflow.log_metric("rmse_winter", float(rmse_value_winter))
        mlflow.log_metric("mase_winter", float(mase_value_winter))
    else:
        rmse_value, mase_value = rmse, mase
    mlflow.log_metric("rmse", float(rmse_value))
    mlflow.log_metric("mase", float(mase_value))

    print(f"{experiment_name}\nRMSE: {rmse_value}\nMASE: {mase_value}")


def time_series_evaluation(model_fit: Any, season: str):
    """

    :param model_fit:
    :param season:
    :return:
    """
    dp_season = DataPipelineTS(season=season)
    series_train, series_test = dp_season()
    series_train = watt_into_kilowatt(series_train)
    series_test = watt_into_kilowatt(series_test)

    y_pred = model_fit.forecast(steps=int(len(series_test)))
    y_pred = watt_into_kilowatt(y_pred)
    rmse_value = rmse(y_true=series_test, y_pred=y_pred)
    mase_value = mase(y_train=series_train, y_true=series_test, y_pred=y_pred, seasonality=1)

    if season == "all":
        dp_summer = DataPipelineTS(season="summer")
        dp_summer()
        index_summer_test = [x for x in dp_season.test_index if x in dp_summer.test_index]
        index_summer_train = [x for x in dp_season.train_index if x in dp_summer.train_index]

        dp_winter = DataPipelineTS(season="winter")
        dp_winter()
        index_winter_test = [x for x in dp_season.test_index if x in dp_winter.test_index]
        index_winter_train = [x for x in dp_season.train_index if x in dp_winter.train_index]

        summer_len_test = len(index_summer_test)
        winter_len_test = len(index_winter_test)
        summer_len_train = len(index_summer_train)
        winter_len_train = len(index_winter_train)

        rmse_value_summer = rmse(y_true=series_test[:summer_len_test], y_pred=y_pred[:summer_len_test])
        mase_value_summer = mase(y_train=series_train[:summer_len_train], y_true=series_test[:summer_len_test],
                                 y_pred=y_pred[:summer_len_test], seasonality=1)

        rmse_value_winter = rmse(y_true=series_test[-winter_len_test:], y_pred=y_pred[-winter_len_test:])
        mase_value_winter = mase(y_train=series_train[-winter_len_train:], y_true=series_test[-winter_len_test:],
                                 y_pred=y_pred[-winter_len_test:], seasonality=1)

        return (rmse_value, rmse_value_summer, rmse_value_winter), (mase_value, mase_value_summer, mase_value_winter)
    else:
        return rmse_value, mase_value
