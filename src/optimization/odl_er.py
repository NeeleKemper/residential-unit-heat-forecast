import mlflow
import time
import numpy as np
from typing import Tuple

from src.utils.metrics import rmse
from src.online.ODLER import ODLER
from src.pipeline.DataPipelineOnline import DataPipelineOnline
from src.utils.utils import parse_season


def er_forecast(config: list, pipeline: DataPipelineOnline):
    """

    :param config:
    :param pipeline:
    :return:
    """
    error_list_24 = []
    error_list_48 = []
    season, cfg = config
    model = ODLER(season, cfg)
    path = f"checkpoints/er_{season}"
    model.save_weights(path)
    for X_train, X_test, y_train, y_test, _ in pipeline():
        model.load_weights(path)
        model.fit(X_train=X_train, y_train=y_train)
        model.save_weights(path)
        # one day a ahead forecast
        pred_24 = model.predict(X_test[:24])
        error_24 = rmse(y_true=y_test[:24], y_pred=pred_24)
        error_list_24.append(error_24)
        # two days ahead forecast
        if len(X_test) > 24:
            pred_48 = model.predict(X_test[24:])
            error_48 = rmse(y_true=y_test[24:], y_pred=pred_48)
        else:
            error_48 = error_24
        error_list_48.append(error_48)
    return error_list_24, error_list_48


def er_config(season: str):
    models = []
    batch_size = [4, 8, 16]
    capacity = [500, 1000, 5000, 10000, 20000]
    factor = [0.1, 0.2, 0.3, 0.4, 0.5]
    min_recollection = [0 * 24, 1 * 24, 2 * 24, 7 * 24, 14 * 24, 30 * 24]
    for b in batch_size:
        for c in capacity:
            for f in factor:
                for mr in min_recollection:
                    cfg = {"batch_size": b, "capacity": c, "factor": f, "min_recollection": mr}
                    models.append([season, cfg])
    return models


# grid search configs
def grid_search(cfg_list: list, pipeline: DataPipelineOnline) -> list:
    """

    :param cfg_list:
    :param pipeline:
    :return:
    """
    scores = [score_model(cfg, pipeline) for cfg in cfg_list]
    # remove empty results
    scores = [r for r in scores if r[1] is not None]
    return scores


def score_model(config: list, pipeline: DataPipelineOnline) -> Tuple[str, float, float]:
    """

    :param config:
    :param pipeline:
    :return:
    """

    run_name = int(time.time())
    with mlflow.start_run(run_name=str(run_name)):
        season, cfg = config
        key = f'{cfg["batch_size"]}, {cfg["capacity"]}, {cfg["factor"]},{cfg["min_recollection"]}'
        errors_24, errors_48 = er_forecast(config, pipeline)
        mean_error_24 = np.round(np.mean(errors_24), 4)
        mean_error_48 = np.round(np.mean(errors_48), 4)
        params = {"batch_size": cfg["batch_size"], "capacity": cfg["capacity"], "factor": cfg["factor"],
                  "min_recollection": cfg["min_recollection"]}
        mlflow.log_params(params)
        mlflow.log_metric("rmse_24", mean_error_24)
        mlflow.log_metric("rmse_48", mean_error_48)
        print(
            f"{cfg} 24 hour forecast {mean_error_24}, 48 hour forecast {mean_error_48}")
    return key, mean_error_24, mean_error_48


if __name__ == "__main__":
    season = parse_season()
    cfg = er_config(season)
    print(len(cfg))

    pipeline = DataPipelineOnline(season)

    client = mlflow.tracking.MlflowClient()
    try:
        # Creates a new experiment
        experiment_id = client.create_experiment(f"ER_{season}")
    except(Exception,):
        # Retrieves the experiment id from the already created project
        experiment_id = client.get_experiment_by_name(f"ER_{season}").experiment_id

    # view results under mlflow ui
    mlflow.set_experiment(experiment_id=experiment_id)

    scores = grid_search(cfg, pipeline)
    # sort configs by error for 24 hour forecast,asc
    scores.sort(key=lambda tup: tup[1])
    # list top 3 configs
    print("\nBest config for one day a ahead forecast:")
    for cfg, error_24, error_48 in scores[:3]:
        print(cfg, error_24, error_48)
    # sort configs by error for 48 hour forecast,asc
    scores.sort(key=lambda tup: tup[2])
    print("\nBest config two days ahead forecast:")
    for cfg, error_24, error_48 in scores[:3]:
        print(cfg, error_24, error_48)
