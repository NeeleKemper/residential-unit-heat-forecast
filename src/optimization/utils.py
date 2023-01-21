import tensorflow as tf
import keras_tuner as kt
import numpy as np
from sklearn import metrics
from sklearn import model_selection
from typing import Callable

from src.utils.metrics import rmse
from src.pipeline.DataPipelineOffline import DataPipelineOffline


def init_sklearn_tuner(max_trials: int, project_name: str, hypermodel: Callable) -> kt.tuners.SklearnTuner:
    """

    :param max_trials:
    :param project_name:
    :param hypermodel:
    :return:
    """
    tuner = kt.tuners.SklearnTuner(
        oracle=kt.oracles.BayesianOptimizationOracle(
            objective=kt.Objective('score', 'min'),
            max_trials=max_trials),
        hypermodel=hypermodel,
        scoring=metrics.make_scorer(rmse),
        cv=model_selection.KFold(n_splits=10, random_state=42, shuffle=True),
        directory='.',
        project_name=project_name)
    return tuner


def init_nn_tuner(max_trials: int, project_name: str, hypermodel: Callable, objective: str="val_mse") -> kt.tuners.BayesianOptimization:
    """

    :param max_trials:
    :param project_name:
    :param hypermodel:
    :return:
    """
    tuner = kt.BayesianOptimization(
        hypermodel=hypermodel(),
        objective=kt.Objective(objective, "min"),
        max_trials=max_trials,
        seed=42,
        directory=".",
        project_name=project_name)
    return tuner


def return_best_hyperparameters(tuner: kt.tuners) -> None:
    """

    :param tuner:
    :return:
    """
    num_models = 3
    print(f"Top {num_models} models:")
    best_hyperparameters = tuner.get_best_hyperparameters(num_models)
    for best_hp in best_hyperparameters:
        print(f"Hyperparameters: {best_hp.values}")


def sklearn_optimization(season: str, project_name: str, build_model: Callable) -> None:
    """

    :param season:
    :param project_name:
    :param build_model:
    :return:
    """
    max_trials = 100

    dp = DataPipelineOffline(season=season)
    X, y = dp()

    tuner = init_sklearn_tuner(max_trials=max_trials, project_name=project_name, hypermodel=build_model)
    tuner.search_space_summary()
    tuner.search(X, y)

    return_best_hyperparameters(tuner)


def nn_optimization_offline(project_name: str, X: np.ndarray, y: np.ndarray, hypermodel: Callable):
    """

    :param project_name:
    :param X:
    :param y:
    :param hypermodel:
    :return:
    """
    max_trials = 100
    epochs = 512

    tuner = init_nn_tuner(max_trials=max_trials, project_name=project_name, hypermodel=hypermodel)

    print(tuner.search_space_summary())
    stop_early = tf.keras.callbacks.EarlyStopping("val_loss", patience=10, mode="auto", restore_best_weights=True)
    tuner.search(X, y, epochs=epochs, validation_split=0.2, callbacks=[stop_early], verbose=1)

    return_best_hyperparameters(tuner)


def nn_optimization_online(project_name: str, hypermodel: Callable, objective: str = "val_mse"):
    """

    :param project_name:
    :param hypermodel:
    :return:
    """
    max_trials = 100

    tuner = init_nn_tuner(max_trials=max_trials, project_name=project_name, hypermodel=hypermodel, objective=objective)

    print(tuner.search_space_summary())

    # dummy input
    dp = DataPipelineOffline(season="all")
    X, y = dp()

    tuner.search(X, y)

    return_best_hyperparameters(tuner)
