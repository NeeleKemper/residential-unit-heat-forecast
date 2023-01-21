import tensorflow as tf
import keras_tuner as kt
import numpy as np
from sklearn.linear_model import SGDRegressor

from src.utils.utils import parse_season
from src.optimization.utils import sklearn_optimization

tf.random.set_seed(42)
np.random.seed(42)

parameters = {
    "loss": ["huber", "squared_error", "epsilon_insensitive", "squared_epsilon_insensitive"],
    "penalty": ["l2", "l1", "elasticnet"],
    "lr": ["invscaling", "optimal", "constant", "adaptive"],
    "a_min": 0.1,
    "a_max": 10,
    "a_step": 0.1,
    "eps_min": 0.1,
    "eps_max": 10,
    "eps_step": 0.1,
    "eta_min": 0.1,
    "eta_max": 10,
    "eta_step": 0.1,
    "pow_min": 0.1,
    "pow_max": 0.1,
    "pow_step": 0.1
}


def build_model(hp: kt.HyperParameters) -> SGDRegressor:
    """

    :param hp:
    :return:
    """
    loss = hp.Choice("loss", parameters["loss"])
    penalty = hp.Choice("penalty", parameters["penalty"])
    learning_rate = hp.Choice("learning_rate", parameters["lr"])
    alpha = hp.Float("alpha", min_value=parameters["a_min"], max_value=parameters["a_max"],
                     step=parameters["a_step"])
    epsilon = hp.Float("epsilon", min_value=parameters["eps_min"], max_value=parameters["eps_max"],
                       step=parameters["eps_step"])
    eta0 = hp.Float("eta0", min_value=parameters["eta_min"], max_value=parameters["eta_max"],
                    step=parameters["eta_step"])
    power_t = hp.Float("power_t", min_value=parameters["pow_min"], max_value=parameters["pow_max"],
                       step=parameters["pow_step"])

    model = SGDRegressor(loss=loss, penalty=penalty, learning_rate=learning_rate, alpha=alpha, epsilon=epsilon,
                                 eta0=eta0, power_t=power_t, max_iter=20000, early_stopping=True, n_iter_no_change=10)

    return model


if __name__ == "__main__":
    season = parse_season()
    sklearn_optimization(season=season, project_name=f"sgd_{season}", build_model=build_model)

