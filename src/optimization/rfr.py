import tensorflow as tf
import keras_tuner as kt
import numpy as np
from sklearn.ensemble import RandomForestRegressor

from src.utils.utils import parse_season
from src.optimization.utils import sklearn_optimization

tf.random.set_seed(42)
np.random.seed(42)

parameters = {
    "n_est_max": 5000,
    "n_est_min": 100,
    "n_est_step": 10,
    "max_features": ["auto", "sqrt", "log2"],
    "max_depth": [None],
    "min_samples_split_min": 2,
    "min_samples_split_max": 10,
    "min_samples_split_step": 1,
    "min_weight_fraction_leaf_min": 0.0,
    "min_weight_fraction_leaf_max": 0.5,
    "min_weight_fraction_leaf_step": 0.1,
    "min_samples_leaf_min": 2,
    "min_samples_leaf_max": 10,
    "min_samples_leaf_step": 1,
    "bootstrap": [True, False],
}


def build_model(hp: kt.HyperParameters) -> RandomForestRegressor:
    """

    :param hp:
    :return:
    """
    n_estimators = hp.Int("n_estimators", min_value=parameters["n_est_min"], max_value=parameters["n_est_max"],
                          step=parameters["n_est_step"])
    max_features = hp.Choice("max_features", parameters["max_features"])
    min_samples_split = hp.Int("min_samples_split", min_value=parameters["min_samples_split_min"],
                               max_value=parameters["min_samples_split_max"], step=parameters["min_samples_split_step"])
    min_samples_leaf = hp.Int("min_samples_leaf", min_value=parameters["min_samples_leaf_min"],
                              max_value=parameters["min_samples_leaf_max"], step=parameters["min_samples_leaf_step"])

    min_weight_fraction_leaf = hp.Float("min_weight_fraction_leaf",
                                        min_value=parameters["min_weight_fraction_leaf_min"],
                                        max_value=parameters["min_weight_fraction_leaf_max"],
                                        step=parameters["min_weight_fraction_leaf_step"])

    bootstrap = hp.Choice("bootstrap", parameters["bootstrap"])

    model = RandomForestRegressor(n_estimators=n_estimators, max_features=max_features, max_depth=None,
                                  min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                                  bootstrap=bootstrap, min_weight_fraction_leaf=min_weight_fraction_leaf)

    return model


def main():
    season = parse_season()
    sklearn_optimization(season=season, project_name=f"rfr_{season}", build_model=build_model)


if __name__ == "__main__":
    main()

