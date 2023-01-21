import mlflow
import numpy as np


def get_history_metric(client: mlflow.tracking.MlflowClient, key: str, variables: dict, seeds: np.ndarray, run_type: str) -> list:
    """

    :param client:
    :param key:
    :param variables:
    :param seeds:
    :return:
    """
    metric_list = [client.get_metric_history(variables[f"run_{run_type}_{i}"].info.run_id, key) for i in seeds]
    return [m[0].value for m in metric_list]
