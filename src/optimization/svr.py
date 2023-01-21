import tensorflow as tf
import keras_tuner as kt
import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from src.pipeline.DataPipelineOffline import DataPipelineOffline

from src.utils.utils import parse_season
from src.optimization.utils import sklearn_optimization

tf.random.set_seed(42)
np.random.seed(42)

parameters = {
    "kernel": ["rbf", "linear"],
    "gamma": ["auto", "scale"],
    "c_min": 10,
    "c_max": 10000,
    "c_step": 100,
    "eps_min": 0.1,
    "eps_max": 100.0,
    "eps_step": 0.1,
}


def build_model(hp: kt.HyperParameters) -> Pipeline:
    """

    :param hp:
    :return:
    """
    kernel = hp.Choice("kernel", parameters["kernel"])
    C = hp.Int("C", min_value=parameters["c_min"], max_value=parameters["c_max"], step=parameters["c_step"])
    gamma = hp.Choice("gamma", parameters["gamma"])
    epsilon = hp.Float("epsilon", min_value=parameters["eps_min"], max_value=parameters["eps_max"],
                       step=parameters["eps_step"])

    model = Pipeline([
        ('scale', StandardScaler()),
        ('clf', SVR(kernel=kernel, C=C, gamma=gamma, epsilon=epsilon))])
    return model


def main():
    global parameters
    season = parse_season()

    dp = DataPipelineOffline(season=season)
    X, y = dp()

    parameters["c_min"] = round(y.min(), -2)
    parameters["c_max"] = round(y.max(), -2)
    if season == "summer":
        parameters["c_step"] = 10
    else:
        parameters["c_step"] = 100

    sklearn_optimization(season=season, project_name=f"krr_{season}", build_model=build_model)


if __name__ == "__main__":
    main()
