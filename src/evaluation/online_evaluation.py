import mlflow
import sklearn.base
import numpy as np
import pandas as pd
import tensorflow as tf
from typing import Any, Tuple

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

from src.data_processing.anomaly_detection import estimate_anomalies
from src.pipeline.DataPipelineOnline import DataPipelineOnline
from src.evaluation.utils import get_history_metric
from src.utils.utils import watt_into_kilowatt
from src.utils.contant import SUMMER_BEGIN, SUMMER_END, WINTER_BEGIN, WINTER_END
from src.utils.metrics import rmse, mase, mase_non_timeseries, mad
from src.online.ODLER import ODLER

sns.set()

rmse_over_time = np.array([])
mase_over_time = np.array([])
mad_over_time = np.array([])
prediction_over_time = np.array([])
y_over_time = np.array([])


def online_experiment(model: Any, model_type: str, model_name: str, season: str):
    """

    :param model:
    :param model_type:
    :param model_name:
    :param season:
    :return:
    """
    assert model_type in ["sklearn", "filter", "keras"], \
        "The model_type parameter is not defined. It must be 'filter', 'sklearn' or 'keras'"
    assert model_name in ["hbp", "odl", "odl_er", "rls", "sgd"], \
        "The model_name parameter is not defined. It must be 'hbp', 'odl', 'odl_er', 'rls' or 'sgd'."

    run_variables = dict()
    min_seed = 1
    max_seed = 30

    if model_type == "filter":
        max_seed = min_seed
    seeds = np.arange(min_seed, (max_seed + 1), 1)

    columns = create_columns("pred")
    df_pred = pd.DataFrame(columns=columns, index=seeds)

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
        tf.random.set_seed(seed)
        np.random.seed(seed)

        if model_type == "sklearn":
            model = sklearn.base.clone(model)
        if model_type == "keras":
            model.clone()
            path = f"checkpoints/{model_name}_{season}"
            model.save_weights(path)
        print(f"\nSeed {seed}")
        with mlflow.start_run(run_name=str(seed)) as run_variables[f"run_online_{seed}"]:
            df = online_evaluation(model=model, model_name=model_name, season=season, seed=seed)
            for col in df.columns:
                if "pred" in col:
                    df_pred.loc[seed, col] = df[col].values
                else:
                    if col == "rmse_24" or "rmse_48":
                        print(f"{col}: {float(df[col].values)}")
                    mlflow.log_metric(col, float(df[col].values))

    df_pred = df_pred.dropna(axis=0, how="all")
    df_pred = df_pred.dropna(axis=1, how="all")

    title = f"{model_name.upper()}-{season[0].upper()}{season[1:]}"
    title = title.replace("_", "-")
    save_as = f"{model_name}_{season}"
    summary_over_time(season=season, title=title, save_as=save_as)

    with mlflow.start_run(run_name=str("summary")):
        for col in df.columns:
            if "pred" in col:
                pred = [row.tolist() for row in df_pred[col]]
                value = mad(predictions=pred)
                metric_name = col.replace("pred", "mad")
                mlflow.log_metric(metric_name, float(value))
            else:
                metric_list = get_history_metric(mlflow_client, col, run_variables, seeds, run_type="online")
                value = np.round(np.mean(metric_list), 2)
                mlflow.log_metric(col, float(value))



def create_columns(metrics: str) -> list:
    """

    :param metrics:
    :return:
    """
    periods = ["_24", "_48"]
    season = ["", "_summer", "_winter"]
    year = ["", "_one", "_two"]

    columns = list()
    for p in periods:
        for s in season:
            for y in year:
                col = f"{metrics}{s}{y}{p}"
                columns.append(col)
    return columns


def online_evaluation(model: Any, model_name: str, season: str, seed: int) -> pd.DataFrame:
    """

    :param model:
    :param model_name:
    :param season:
    :param seed:
    :return:
    """
    df_anomaly, _, _ = estimate_anomalies()

    np.random.seed(seed)
    dp = DataPipelineOnline(season=season)

    columns = create_columns("pred")
    columns.extend(create_columns("rmse"))
    columns.extend(create_columns("mase"))
    columns.extend(create_columns("anomaly"))

    idx_len = int(len(df_anomaly) / 24)
    df_results = pd.DataFrame(columns=columns, index=range(idx_len))

    df_summary = pd.DataFrame(columns=columns, index=range(1))

    y_true = list()
    for i, (X_train, X_test, y_train, y_test, idx_test) in enumerate(dp()):
        anomaly = df_anomaly.loc[idx_test]["anomalies"].tolist()
        if model_name == "sgd":
            learning_iteration = int(1e4)
            for z in range(learning_iteration):
                model.partial_fit(X_train, y_train.ravel())
        elif model_name == "rls":
            model.run(y_train, X_train)
        elif model_name == "hbp" or model_name == "odl" or model_name == "odl_er":
            path = f"checkpoints/{model_name}_{season}"
            model.load_weights(path)
            model.fit(X_train=X_train, y_train=y_train)
            model.save_weights(path)
        metrics = online_metric_calculation(model, X_test, y_train, y_test, anomaly)

        y_true.append(y_test[:24].flatten())

        dt_summer_one = pd.date_range(start=SUMMER_BEGIN[0], end=SUMMER_END[0], freq="H")
        dt_summer_two = pd.date_range(start=SUMMER_BEGIN[1], end=SUMMER_END[1], freq="H")
        dt_winter_one = pd.date_range(start=WINTER_BEGIN[0], end=WINTER_END[0], freq="H")
        dt_winter_two = pd.date_range(start=WINTER_BEGIN[1], end=WINTER_END[1], freq="H")

        idx_start_24 = idx_test[0]
        idx_start_48 = idx_start_24
        if len(idx_test) > 24:
            idx_start_48 = idx_test[24]

        for key in metrics:
            df_results.loc[i, key] = metrics[key]

            metric, period = key.split("_")

            new_keys = list()
            if idx_start_24 in dt_summer_one:
                new_keys.append(f"{metric}_summer_one_{period}")
            if idx_start_48 in dt_summer_one:
                new_keys.append(f"{metric}_summer_one_{period}")
            if idx_start_24 in dt_summer_two:
                new_keys.append(f"{metric}_summer_two_{period}")
            if idx_start_48 in dt_summer_two:
                new_keys.append(f"{metric}_summer_two_{period}")

            if idx_start_24 in dt_winter_one:
                new_keys.append(f"{metric}_winter_one_{period}")
            if idx_start_48 in dt_winter_one:
                new_keys.append(f"{metric}_winter_one_{period}")
            if idx_start_24 in dt_winter_two:
                new_keys.append(f"{metric}_winter_two_{period}")
            if idx_start_48 in dt_winter_two:
                new_keys.append(f"{metric}_winter_two_{period}")

            if season == "all":
                if idx_start_24 in dt_summer_one or idx_start_24 in dt_summer_two:
                    new_keys.append(f"{metric}_summer_{period}")
                if idx_start_48 in dt_summer_one or idx_start_48 in dt_summer_two:
                    new_keys.append(f"{metric}_summer_{period}")
                if idx_start_24 in dt_winter_one or idx_start_24 in dt_winter_two:
                    new_keys.append(f"{metric}_winter_{period}")
                if idx_start_48 in dt_winter_one or idx_start_48 in dt_winter_two:
                    new_keys.append(f"{metric}_winter_{period}")
                if idx_start_24 in dt_summer_one or idx_start_24 in dt_winter_one:
                    new_keys.append(f"{metric}_one_{period}")
                if idx_start_48 in dt_summer_one or idx_start_48 in dt_winter_one:
                    new_keys.append(f"{metric}_one_{period}")
                if idx_start_24 in dt_summer_two or idx_start_24 in dt_winter_two:
                    new_keys.append(f"{metric}_two_{period}")
                if idx_start_48 in dt_summer_two or idx_start_48 in dt_winter_two:
                    new_keys.append(f"{metric}_two_{period}")

            if new_keys:
                for k in new_keys:
                    df_results.loc[i, k] = metrics[key]
    df_results = df_results.dropna(axis=0, how="all")
    df_results = df_results.dropna(axis=1, how="all")

    evaluation_over_time(df_results, y_true)

    for col in df_results.columns:
        df = df_results[col].dropna(axis=0)
        l = list()
        if "pred" in col:
            for row in df:
                l.extend(row)
            value = l
        elif "anomaly" in col:
            for row in df:
                l.extend(row)
            value = online_anomaly_error(l)
        else:
            value = np.round(np.mean(df), 2)
        df_summary.loc[0, col] = value
    df_summary = df_summary.dropna(axis=1, how="all")
    return df_summary


def online_metric_calculation(model: Any, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray,
                              anomalies: np.ndarray) -> dict:
    """

    :param model:
    :param X_test:
    :param y_train:
    :param y_test:
    :param anomalies:
    :return:
    """
    global  y_train_over_time
    y_train = watt_into_kilowatt(y_train)
    y_test = watt_into_kilowatt(y_test)

    pred_24 = model.predict(X_test[:24])
    y_test_24 = y_test[:24]
    y_test_48 = y_test[24:]

    anomaly_idx_24 = np.argwhere(np.array(anomalies[:24]) < 0).flatten().tolist()
    pred_24 = watt_into_kilowatt(pred_24)
    anomaly_24 = [[pred_24[i], y_test_24[i].tolist()[0]] for i in anomaly_idx_24]

    rmse_24 = rmse(y_test_24, pred_24)
    if type(model) is ODLER:
        mase_24 = mase_non_timeseries(y_train=y_train, y_true=y_test, y_pred=pred_24.flatten())
    else:
        y_train_over_time = y_train_over_time + y_train.flatten()
        mase_24 = mase(y_train=y_train_over_time, y_true=y_test_24.flatten(), y_pred=pred_24.flatten(), seasonality=1)

    if len(X_test) > 24:
        anomaly_idx_48 = np.argwhere(np.array(anomalies[24:]) < 0).flatten().tolist()
        # two day a ahead forecast
        pred_48 = model.predict(X_test[24:])
        pred_48 = watt_into_kilowatt(pred_48)
        anomaly_48 = [[pred_48[i], y_test_48[i].tolist()[0]] for i in anomaly_idx_48]
        rmse_48 = rmse(y_test_48, pred_48)
        if type(model) is ODLER:
            mase_48 = mase_non_timeseries(y_train=y_train, y_true=y_test, y_pred=pred_48.flatten())
        else:
            mase_48 = mase(y_train=y_train.flatten(), y_true=y_test_48.flatten(), y_pred=pred_48.flatten(), seasonality=1)
    else:
        pred_48 = pred_24
        rmse_48 = rmse_24
        mase_48 = mase_24
        anomaly_48 = anomaly_24

    return {
        "rmse_24": rmse_24, "mase_24": mase_24, "pred_24": pred_24, "anomaly_24": anomaly_24,
        "rmse_48": rmse_48, "mase_48": mase_48, "pred_48": pred_48, "anomaly_48": anomaly_48,
    }


def online_anomaly_error(anomaly_list: list) -> np.ndarray:
    """

    :param anomaly_list:
    :return:
    """
    pred = [i[0] for i in anomaly_list]
    y_true = [i[1] for i in anomaly_list]
    return rmse(np.array(pred), np.array(y_true))


def evaluation_over_time(df: pd.DataFrame, y) -> None:
    """

    :param df:
    :return:
    """
    global rmse_over_time, mase_over_time, mad_over_time, prediction_over_time, y_over_time
    rmse_val = df["rmse_24"].to_numpy()
    mase_val = df["mase_24"].to_numpy()
    pred = [row.tolist() for row in df["pred_24"]]
    y_list = [row.tolist() for row in y]
    mad_val = [mad(p) for p in pred]

    # On 2020-12-20 dates missing from 09:00 to 17:00.
    for p, y in zip(pred, y_list):
        if len(p) != 24:
            [p.insert(i, np.nan) for i in range(9, 18)]
            [y.insert(i, np.nan) for i in range(9, 18)]

    pred_val = [item for sublist in pred for item in sublist]
    y_val = [item for sublist in y_list for item in sublist]
    rmse_over_time = np.vstack([rmse_over_time, rmse_val]) if rmse_over_time.size else rmse_val
    mase_over_time = np.vstack([mase_over_time, mase_val]) if mase_over_time.size else mase_val
    mad_over_time = np.vstack([mad_over_time, np.array(mad_val)]) if mad_over_time.size else np.array(mad_val)
    prediction_over_time = np.vstack(
        [prediction_over_time, np.array(pred_val)]) if prediction_over_time.size else np.array(pred_val)
    y_over_time = np.vstack([y_over_time, np.array(y_val)]) if y_over_time.size else np.array(y_val)


def summary_over_time(season: str, title: str, save_as: str) -> None:
    """

    :param season:
    :param title:
    :param save_as:
    :return:
    """
    if "RLS" not in title:
        rmse_values = np.around(np.mean(rmse_over_time, axis=0).astype(np.float), 2)
        mase_values = np.around(np.mean(mase_over_time, axis=0).astype(np.float), 2)
        mad_values = np.around(np.mean(mad_over_time, axis=0).astype(np.float), 2)
        pred_values = np.around(np.mean(prediction_over_time, axis=0).astype(np.float), 2)
        y_values = np.around(np.mean(y_over_time, axis=0).astype(np.float), 2)
    else:
        rmse_values = rmse_over_time
        mase_values = mase_over_time
        mad_values = mad_over_time
        pred_values = prediction_over_time
        y_values = y_over_time

    dt, idx = create_datetime_index(season=season, freq="D")

    metrics = ["RMSE", "MASE", "MAD", "prediction"]

    y_series = pd.Series(data=y_values)
    for m in metrics:
        if m == "RMSE":
            series = pd.Series(data=rmse_values, index=dt)
        elif m == "MASE":
            series = pd.Series(data=mase_values, index=dt)
        elif m == "MAD":
            series = pd.Series(data=mad_values, index=dt)
        else:
            dt, idx = create_datetime_index(season=season, freq="H")
            series = pd.Series(data=pred_values, index=dt)
            y_series = pd.Series(data=y_values, index=dt)
            y_series = y_series.reindex(idx, fill_value=np.nan)

            # save prediction to csv
            series = series.reindex(idx, fill_value=np.nan)
            series_to_save = pd.DataFrame({"date":series.index, "pred": series.values, "heat": y_series.values * 0.001})
            series_to_save.set_index("date", inplace=True)
            series_to_save.to_csv(f"eval_over_time/{save_as}_prediction.csv", sep=";")

        series = series.reindex(idx, fill_value=np.nan)
        plot_over_time(x=series.index, y=series.values, y_heat=y_series.values, metric=m, title=title, save_as=save_as)


def create_datetime_index(season: str, freq: str) -> Tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    """

    :param season:
    :param freq:
    """
    if season == "summer":
        dt_summer_one = pd.date_range(start="2020-06-02 00:00:00", end=SUMMER_END[0], freq=freq)
        dt_summer_two = pd.date_range(start=SUMMER_BEGIN[1], end="2021-09-19 23:00:00", freq=freq)
        idx = pd.date_range(start="2020-06-02 00:00:00", end="2021-09-19 23:00:00", freq=freq)
        dt = dt_summer_one.append(dt_summer_two)
    elif season == "winter":
        dt_winter_one = pd.date_range(start="2020-09-22 00:00:00", end=WINTER_END[0], freq=freq)
        dt_winter_two = pd.date_range(start=WINTER_BEGIN[1], end="2022-02-27 23:00:00", freq=freq)
        idx = pd.date_range(start="2020-09-22 00:00:00", end="2022-02-27 23:00:00", freq=freq)
        dt = dt_winter_one.append(dt_winter_two)
    else:
        dt = pd.date_range(start="2020-06-02 00:00:00", end="2022-02-27 23:00:00", freq=freq)
        idx = dt
    return dt, idx


def plot_over_time(x: pd.Index, y: np.array, y_heat: np.array, metric: str, title: str, save_as: str) -> None:
    """

    :param x:
    :param y:
    :param y_heat:
    :param metric:
    :param title:
    :param save_as:
    :return:
    """
    x_label = "Time (Day)"
    color = "b"
    fig, ax = plt.subplots()

    if metric != "prediction":
        ax.set_xlabel(x_label)
        ax.set_ylabel(f"{metric} [kWh]")
        plt.title(f"Daily {metric} values of the {title}")
        ax.plot(x, y, linewidth=0.5, alpha=1, color=color)
    else:
        ax.set_xlabel("Time [h]")
        ax.set_ylabel("Heat [kW]")
        values = y_heat * 0.001
        ax.plot(x, values, linewidth=0.5, color="lightgrey", alpha=0.9, label="heat consumption")
        ax.plot(x, y, linewidth=0.5, color=color, alpha=1, label="predicted heat consumption")
        plt.title(f"{title} (24 hour prediction)")
        ax.legend(loc='upper left')
    # format the ticks
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m\n%Y"))

    ax.xaxis.set_minor_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_minor_formatter(mdates.DateFormatter("%m"))

    date_min = np.datetime64(x[0], "D")
    date_max = np.datetime64(x[-1], "D")

    ax.set_xlim(date_min, date_max)

    ax.grid(True, which="both")

    plt.tight_layout()

    plt.savefig(f"eval_over_time/{save_as}_{metric}.png")

    plt.show()
    plt.close()
