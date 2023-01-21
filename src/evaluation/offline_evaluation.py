import mlflow
import sklearn.base
import numpy as np
import tensorflow as tf
from typing import Any, Tuple

from src.data_processing.anomaly_detection import estimate_anomalies
from src.pipeline.DataPipelineOffline import DataPipelineOffline
from sklearn.model_selection import train_test_split
from src.evaluation.utils import get_history_metric
from src.utils.utils import watt_into_kilowatt
from src.utils.metrics import rmse, mase_non_timeseries, mad


def offline_experiment(model: Any, model_type: str, model_name: str, season: str):
    """

    :param model:
    :param model_type:
    :param model_name:
    :param season:
    :return:
    """
    experiment = mlflow.get_experiment_by_name('dnn_all')
    assert model_type in ["sklearn",
                          "keras"], "The model_type parameter is not defined. It must be 'sklearn' or 'keras'"
    variables = dict()

    # load dataset
    min_seed = 1
    max_seed = 30
    seeds = np.arange(min_seed, (max_seed + 1), 1)

    predictions = list()
    predictions_summer = list()
    predictions_winter = list()

    experiment_name = f"{model_name}_{season}"
    mlflow_client = mlflow.tracking.MlflowClient()
    try:
        # Creates a new experiment
        experiment_id = mlflow_client.create_experiment(experiment_name)
    except(Exception,):
        # Retrieves the experiment id from the already created project
        experiment_id = mlflow_client.get_experiment_by_name(experiment_name).experiment_id

    # view results under mlflow ui
    mlflow.set_experiment(experiment_id=experiment_id)
    print(f"Season: {season}")
    for seed in seeds:
        print(f"\nSeed {seed}")
        with mlflow.start_run(run_name=str(seed)) as variables[f"run_offline_{seed}"]:
            if model_type == "sklearn":
                model = sklearn.base.clone(model)
            else:
                model.clone()
            metrics, metrics_summer, metrics_winter = offline_evaluation(model=model, season=season, seed=seed)

            metrics_list = ["rmse", "mase", "anomaly"]
            print(f"RMSE: {metrics[0]}")
            for i, metric in enumerate(metrics_list):
                mlflow.log_metric(metric, float(metrics[i]))
            predictions.append(metrics[3])
            if season == "all":

                for i, metric in enumerate(metrics_list):
                    mlflow.log_metric(f"{metric}_summer", float(metrics_summer[i]))
                predictions_summer.append(metrics_summer[3])

                for i, metric in enumerate(metrics_list):
                    mlflow.log_metric(f"{metric}_winter", float(metrics_winter[i]))
                predictions_winter.append(metrics_winter[3])

    with mlflow.start_run(run_name=str("summary")):
        offline_summary(mlflow_client, variables, seeds, season="", predictions=predictions)

        if season == "all":
            offline_summary(mlflow_client, variables, seeds, season="summer", predictions=predictions_summer)
            offline_summary(mlflow_client, variables, seeds, season="winter", predictions=predictions_winter)


def offline_summary(mlflow_client: mlflow.tracking.MlflowClient, variables: dict, seeds: np.ndarray, season: str,
                    predictions: list):
    """

    :param mlflow_client:
    :param variables:
    :param seeds:
    :param season:
    :param predictions:
    :return:
    """
    if season != "":
        season = f"_{season}"
    metrics = ["rmse", "mase", "anomaly"]
    for metric in metrics:
        metric_list = get_history_metric(mlflow_client, f"{metric}{season}", variables, seeds, run_type="offline")
        value_mean = np.round(np.mean(metric_list), 2)
        mlflow.log_metric(f"{metric}{season}", float(value_mean))
    mad_value = mad(predictions=predictions)
    mlflow.log_metric(f"mad{season}", float(mad_value))


def offline_evaluation(model: Any, season: str, seed: int) -> Tuple[Tuple, Tuple, Tuple]:
    """

    :param model:
    :param season:
    :param seed:
    :return:
    """
    np.random.seed(seed)
    tf.random.set_seed(seed)
    dp = DataPipelineOffline(season=season)
    X, y = dp()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)


    model.fit(X_train, y_train)

    rmse_value, mase_value, prediction = offline_metric_calculation(model=model, X_test=X_test, X=X, y_train=y_train,
                                                                    y_test=y_test)
    anomaly = offline_anomaly_evaluation(model=model, X_season=X, X_test=X_test, y_test=y_test, season=season)
    if season != "all":
        return (rmse_value, mase_value, anomaly, prediction), (), ()
    else:
        rmse_value_summer, mase_value_summer, prediction_summer = \
            offline_seasonal_evaluation(model=model, X_train=X_train, X_test=X_test, X_season=dp.X_summer,
                                        y_train=y_train, y_test=y_test)

        anomaly_summer = offline_anomaly_evaluation(model=model, X_season=dp.X_summer, X_test=X_test, y_test=y_test,
                                                    season="summer")
        rmse_value_winter, mase_value_winter, prediction_winter = \
            offline_seasonal_evaluation(model=model, X_train=X_train, X_test=X_test, X_season=dp.X_winter,
                                        y_train=y_train, y_test=y_test)
        anomaly_winter = offline_anomaly_evaluation(model=model, X_season=dp.X_winter, X_test=X_test, y_test=y_test,
                                                    season="winter")

    return (rmse_value, mase_value, anomaly, prediction), \
           (rmse_value_summer, mase_value_summer, anomaly_summer, prediction_summer), \
           (rmse_value_winter, mase_value_winter, anomaly_winter, prediction_winter)


def offline_anomaly_evaluation(model: Any, X_season: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, season: str):
    """

    :param model:
    :param X_season:
    :param X_test:
    :param y_test:
    :param season:
    :return:
    """
    df, df_summer, df_winter = estimate_anomalies()
    anomalies, anomalies_summer, anomalies_winter = df["anomalies"], df_summer["anomalies"], df_winter["anomalies"]
    if season == "summer":
        anomalies = anomalies_summer
    elif season == "winter":
        anomalies = anomalies_winter

    # get index of anomalies
    idx_anomalies = [i for i, anomaly in enumerate(anomalies) if anomaly < 0]

    # get index of test data
    idx_test = list()
    for i, x_test in enumerate(X_test):
        # check if the test data is within the selected time period.
        if np.isin(x_test, X_season).all():
            idx_test.append(i)

    # get index of data points that are test data and anomalies
    idx = list(set(idx_anomalies).intersection(idx_test))

    X_test_anomalies = X_test[idx]
    y_test_anomalies = y_test[idx]

    y_pred = model.predict(X_test_anomalies)

    y_pred = watt_into_kilowatt(y_pred)
    y_test_anomalies = watt_into_kilowatt(y_test_anomalies)

    rmse_value = rmse(y_true=y_test_anomalies, y_pred=y_pred)
    return rmse_value


def offline_seasonal_evaluation(model: Any, X_train: np.ndarray, X_test: np.ndarray, X_season: np.ndarray,
                                y_train: np.ndarray, y_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """

    :param model:
    :param X_train:
    :param X_test:
    :param X_season:
    :param y_train:
    :param y_test:
    :return:
    """
    idx_train_season = list()
    idx_test_season = list()

    for i, x_train in enumerate(X_train):
        # check if the train data is within the selected time period.
        if np.isin(x_train, X_season).all():
            idx_train_season.append(i)

    for i, x_test in enumerate(X_test):
        # check if the test data is within the selected time period.
        if np.isin(x_test, X_season).all():
            idx_test_season.append(i)

    y_train_season = y_train[idx_train_season]
    X_test_season = X_test[idx_test_season]
    y_test_season = y_test[idx_test_season]

    return offline_metric_calculation(model, X_test_season, X_season, y_train_season, y_test_season)


def offline_metric_calculation(model: Any, X_test: np.ndarray, X: np.ndarray, y_train: np.ndarray, y_test: np.ndarray) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """

    :param model:
    :param X_test:
    :param X:
    :param y_train:
    :param y_test:
    :return:
    """
    y_pred = model.predict(X_test)

    y_pred = watt_into_kilowatt(y_pred)
    y_test = watt_into_kilowatt(y_test)
    y_train = watt_into_kilowatt(y_train)

    rmse_value = rmse(y_true=y_test, y_pred=y_pred)

    mase_value = mase_non_timeseries(y_train=y_train, y_true=y_test, y_pred=y_pred)

    prediction = model.predict(X)
    prediction = watt_into_kilowatt(prediction)
    return rmse_value, mase_value, prediction
