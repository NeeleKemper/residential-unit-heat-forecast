import mlflow
import time
import numpy as np
from typing import Tuple
from src.utils.metrics import rmse
from sklearn.linear_model import SGDRegressor


from src.pipeline.DataPipelineOnline import DataPipelineOnline
from src.utils.utils import parse_season


def sgd_forecast(config: list, pipeline: DataPipelineOnline):
    """

    :param config:
    :param pipeline:
    :return:
    """
    error_list_24 = []
    error_list_48 = []
    loss, penalty,learning_rate = config
    learning_iteration = int(1e3)
    model = SGDRegressor(loss=loss, penalty=penalty, tol=1e-3, learning_rate=learning_rate)

    for X_train, X_test, y_train, y_test, _ in pipeline():
        for z in range(learning_iteration):
            model.partial_fit(X_train, y_train.ravel())
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


def sgd_config():
    models = []
    loss = ['huber','squared_loss', 'epsilon_insensitive', 'squared_epsilon_insensitive']
    penalty = ['l2', 'l1', 'elasticnet']
    learning_rate = ['invscaling', 'optimal', 'constant', 'adaptive']
    for l in loss:
        for p in penalty:
            for lr in learning_rate:
                cfg = [l, p, lr]
                models.append(cfg)
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


def score_model(cfg: list, pipeline: DataPipelineOnline) -> Tuple[str, float, float]:
    """

    :param cfg:
    :param pipeline:
    :return:
    """

    run_name = int(time.time())
    with mlflow.start_run(run_name=str(run_name)):
        loss, penalty, learning_rate = cfg
        key = str(cfg)
        errors_24, errors_48 = sgd_forecast(cfg, pipeline)
        mean_error_24 = np.round(np.mean(errors_24), 4)
        mean_error_48 = np.round(np.mean(errors_48), 4)
        params = {"loss": loss, "penalty": penalty, "learning_rate": learning_rate}
        mlflow.log_params(params)
        mlflow.log_metric("rmse_24", mean_error_24)
        mlflow.log_metric("rmse_48", mean_error_48)
        print(
            f"{cfg} 24 hour forecast {mean_error_24}, 48 hour forecast {mean_error_48}")
    return key, mean_error_24, mean_error_48


if __name__ == "__main__":
    cfg = sgd_config()

    season = parse_season()
    pipeline = DataPipelineOnline(season)

    client = mlflow.tracking.MlflowClient()
    try:
        # Creates a new experiment
        experiment_id = client.create_experiment(f"SGD_{season}")
    except(Exception,):
        # Retrieves the experiment id from the already created project
        experiment_id = client.get_experiment_by_name(f"SGD_{season}").experiment_id

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