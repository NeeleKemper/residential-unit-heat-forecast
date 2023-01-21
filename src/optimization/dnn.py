import tensorflow as tf
import keras_tuner as kt
import numpy as np

from src.pipeline.DataPipelineOffline import DataPipelineOffline
from src.optimization.utils import nn_optimization_offline
from src.utils.utils import parse_season

tf.random.set_seed(42)
np.random.seed(42)

parameters = {
    "batch_size": [16, 32, 64, 128, 256],
    "units": [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096],
    "optimizer": ["sgd", "rmsprop", "adam"],
    "activation": ["linear", "relu"],
    "dropout_rate": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
    "kernel_initializer": ["lecun_uniform", "he_normal", "he_uniform", "glorot_normal", "glorot_uniform",
                           "normal", "zero", "ones", "uniform", "random_normal", "random_uniform"],
    "regularizer": ["l1", "l2", "l1_l2"],
    "constraint": ["unit_norm", "non_neg", "max_norm", "min_max_norm"],
}


class MyHyperModel(kt.HyperModel):

    def build(self, hp: kt.HyperParameters) -> tf.keras.Sequential:
        """

        :param hp:
        :return:
        """
        variables = dict()
        optimizer = hp.Choice("optimizer", values=parameters["optimizer"])
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Input(8))
        for i in range(1, 4):
            if i < 3:
                variables[f"units_{i}"] = hp.Choice(f"units_{i}", values=parameters["units"])
                variables[f"activation_{i}"] = hp.Choice(f"activation_{i}", values=parameters["activation"])
            else:
                variables[f"units_{i}"] = 1
                variables[f"activation_{i}"] = 'relu'

            variables[f"kernel_regularizer_{i}"] = hp.Choice(f"kernel_regularizer_{i}",
                                                             values=parameters["regularizer"])
            variables[f"activity_regularizer_{i}"] = hp.Choice(f"activity_regularizer_{i}",
                                                               values=parameters["regularizer"])
            variables[f"bias_regularizer_{i}"] = hp.Choice(f"bias_regularizer_{i}", values=parameters["regularizer"])
            variables[f"kernel_initializer_{i}"] = hp.Choice(f"kernel_initializer_{i}",
                                                             values=parameters["kernel_initializer"])
            variables[f"kernel_constraint_{i}"] = hp.Choice(f"kernel_constraint_{i}", values=parameters["constraint"])
            variables[f"bias_constraint_{i}"] = hp.Choice(f"bias_constraint_{i}", values=parameters["constraint"])
            variables[f"dropout_{i}"] = hp.Choice(f"dropout_{i}", values=parameters["dropout_rate"])

            model.add(tf.keras.layers.Dense(units=variables[f"units_{i}"],
                                            activation=variables[f"activation_{i}"],
                                            kernel_regularizer=variables[f"kernel_regularizer_{i}"],
                                            activity_regularizer=variables[f"activity_regularizer_{i}"],
                                            bias_regularizer=variables[f"bias_regularizer_{i}"],
                                            kernel_initializer=variables[f"kernel_initializer_{i}"],
                                            kernel_constraint=variables[f"kernel_constraint_{i}"],
                                            bias_constraint=variables[f"bias_constraint_{i}"],
                                            ))
            if i < 3:
                model.add(tf.keras.layers.Dropout(rate=variables[f"dropout_{i}"]))

        model.compile(loss="mse", optimizer=optimizer, metrics=["mse"])
        # model.summary()
        return model

    def fit(self, hp: kt.HyperParameters, model: tf.keras.Sequential, *args, **kwargs):
        return model.fit(
            *args,
            batch_size=hp.Choice("batch_size", values=parameters["batch_size"]),
            **kwargs,
        )


def main():
    season = parse_season()
    project_name = f"dnn_{season}"

    dp = DataPipelineOffline(season=season)
    X, y = dp()
    nn_optimization_offline(project_name=project_name, X=X, y=y, hypermodel=MyHyperModel)


if __name__ == "__main__":
    main()
