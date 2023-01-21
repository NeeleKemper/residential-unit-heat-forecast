import tensorflow as tf
import keras_tuner as kt
import numpy as np
from sklearn.kernel_ridge import KernelRidge

from src.utils.utils import parse_season
from src.optimization.utils import sklearn_optimization

tf.random.set_seed(42)
np.random.seed(42)

parameters = {
    "kernel": ["rbf", "linear"],
    "a_max": 10,
    "a_min": 0.1,
    "a_step": 0.1,
    "g_min": 0.1,
    "g_max": 10,
    "g_step": 0.1,
}


def build_model(hp: kt.HyperParameters) -> KernelRidge:
    """

    :param hp:
    :return:
    """
    kernel = hp.Choice("kernel", parameters["kernel"])
    alpha = hp.Float("alpha", min_value=parameters["a_min"], max_value=parameters["a_max"],
                     step=parameters["a_step"])
    gamma = hp.Float("gamma", min_value=parameters["g_min"], max_value=parameters["g_max"],
                     step=parameters["g_step"])

    model = KernelRidge(kernel=kernel, alpha=alpha, gamma=gamma)
    return model


def main():
    season = parse_season()
    sklearn_optimization(season=season, project_name=f"krr_{season}", build_model=build_model)


if __name__ == "__main__":
    main()
