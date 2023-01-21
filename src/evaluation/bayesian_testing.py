from itertools import combinations  # permutation
from experiments_eval import get_metric_list


def bayesian_testing(metric_a: list, metric_b: list) -> None:
    """

    :param metric_a: the average rmse values for model a
    :param metric_b: the average rmse values for model b
    :return:
    """
    # TODO: Do your stuff here
    pass


def main():
    # specify the model type: offline or online
    ml = "offline"  # online
    # specify the season type of the model: all, summer or winter
    season = "all"  # summer, winter

    offline_experiments = ["lr", "krr", "svr", "rfr", "dnn", "cnn", "lstm"]
    online_experiments = ["rls", "sgd", "odl", "hbp", "odl_er"]

    if ml == "offline":
        experiments = offline_experiments
        metric_key = "rmse"
    else:
        experiments = online_experiments
        metric_key = "rmse_24"

    # create all possible combination of models (tuple)
    comb = combinations(experiments, 2)
    comb_list = [list(item) for item in comb]

    for c in comb_list:
        # name of experiment = model_season
        experiment_a = f"{c[0]}_{season}"
        experiment_b = f"{c[1]}_{season}"

        # get the average rmse values (as a list) of the models to be compared.
        metrics_a = get_metric_list(experiment_name=experiment_a, metric_key=metric_key)
        metrics_b = get_metric_list(experiment_name=experiment_b, metric_key=metric_key)

        # adjust the number of rmse values for the RLS model so that there are 30 values for both models.
        if "rls" in experiment_a or "rls" in experiment_b:
            if len(metrics_a) == 1:
                metrics_a = metrics_a * 30
            else:
                metrics_b = metrics_b * 30
        assert len(metrics_a) == len(metrics_b), "sample sizes are not equal."

        bayesian_testing(metrics_a, metrics_b)

if __name__ == "__main__":
    main()
