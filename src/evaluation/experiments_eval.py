import mlflow
import numpy as np
from typing import Tuple
from mlflow.tracking import MlflowClient


def get_experiment(experiment_name: str) -> Tuple[MlflowClient, str]:
    """

    :param experiment_name:
    :return:
    """
    client = MlflowClient()
    exp = client.get_experiment_by_name(experiment_name)
    exp_id = exp.experiment_id
    return client, exp_id


def get_description(experiment_name: str) -> None:
    """

    :param experiment_name:
    :return:
    """
    client, exp_id = get_experiment(experiment_name)
    runs = mlflow.search_runs(exp_id)

    # remove summary
    runs = runs[1:].dropna(axis=1)
    print(f"Description of experiment {experiment_name}:\n{runs.describe()}")


def get_metric_list(experiment_name: str, metric_key: str) -> list:
    """

    :param experiment_name:
    :param metric_key:
    :return:
    """
    client, exp_id = get_experiment(experiment_name)
    runs_info = client.list_run_infos(exp_id)[1:]
    runs_ids = [run.run_id for run in runs_info]
    metric_list = [client.get_metric_history(id, metric_key) for id in runs_ids]
    return [m[0].value for m in metric_list]


def get_summary(experiment_name):
    """

    :param experiment_name:
    :return:
    """
    client, exp_id = get_experiment(experiment_name)
    run_info = client.list_run_infos(exp_id)[0]
    run_id = run_info.run_id
    data = mlflow.get_run(run_id).data.to_dictionary()
    metrics = data["metrics"]
    return metrics


def main():
    season = "all"
    model_name = "hbp"
    experiment_name = f"{model_name}_{season}"
    get_description(experiment_name)
    metrics = get_summary(experiment_name)
    print("\nSummary: ")
    for key in metrics:
        print(f"{key}: {metrics[key]}")


if __name__ == "__main__":
    main()
