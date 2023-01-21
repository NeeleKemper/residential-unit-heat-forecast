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
    "units": [8, 16, 32, 64, 128, 256, 512, 1024, 2048],
    "optimizer": ["sgd", "rmsprop", "adam"],
    "activation": ["linear", "relu"],
    "filters": [4, 8, 16, 32, 64, 128],
    "kernel_size": [3, 5, 7, 9],
    "dropout_rate": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
    "kernel_initializer": ["lecun_uniform", "he_normal", "he_uniform", "glorot_normal", "glorot_uniform",
                           "normal", "zero", "ones", "uniform"],
    "recurrent_initializer": ["orthogonal", "lecun_uniform", "he_normal", "he_uniform", "glorot_normal",
                              "glorot_uniform", "normal"],
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
        model = tf.keras.Sequential()

        optimizer = hp.Choice("optimizer", values=parameters["optimizer"])

        model.add(tf.keras.layers.Input(shape=(8, 1)))

        for i in range(1, 4):
            variables[f"filters_cnn_{i}"] = hp.Choice(f"filter_cnn_{i}", values=parameters["filters"])
            variables[f"kernel_size_cnn_{i}"] = hp.Choice(f"kernel_size_cnn_{i}", values=parameters["kernel_size"])
            variables[f"activation_cnn_{i}"] = hp.Choice(f"activation_cnn_{i}", values=parameters["activation"])

            variables[f"kernel_regularizer_cnn_{i}"] = hp.Choice(f"kernel_regularizer_cnn_{i}",
                                                                 values=parameters["regularizer"])
            variables[f"activity_regularizer_cnn_{i}"] = hp.Choice(f"activity_regularizer_cnn_{i}",
                                                                   values=parameters["regularizer"])
            variables[f"bias_regularizer_cnn_{i}"] = hp.Choice(f"bias_regularizer_cnn_{i}",
                                                               values=parameters["regularizer"])
            variables[f"kernel_initializer_cnn_{i}"] = hp.Choice(f"kernel_initializer_cnn_{i}",
                                                                 values=parameters["kernel_initializer"])
            variables[f"kernel_constraint_cnn_{i}"] = hp.Choice(f"kernel_constraint_cnn_{i}",
                                                                values=parameters["constraint"])
            variables[f"bias_constraint_cnn_{i}"] = hp.Choice(f"bias_constraint_cnn_{i}",
                                                              values=parameters["constraint"])

            variables[f"dropout_{i}"] = hp.Choice(f"dropout_{i}", values=parameters["dropout_rate"])

            model.add(tf.keras.layers.Conv1D(
                filters=variables[f"filters_cnn_{i}"],
                kernel_size=variables[f"kernel_size_cnn_{i}"],
                activation=variables[f"activation_cnn_{i}"],
                use_bias=True,
                padding='same',
                kernel_initializer=variables[f"kernel_initializer_cnn_{i}"],
                kernel_regularizer=variables[f"kernel_regularizer_cnn_{i}"],
                bias_regularizer=variables[f"bias_regularizer_cnn_{i}"],
                activity_regularizer=variables[f"activity_regularizer_cnn_{i}"],
                kernel_constraint=variables[f"kernel_constraint_cnn_{i}"],
                bias_constraint=variables[f"bias_constraint_cnn_{i}"],
            ))
            model.add(tf.keras.layers.MaxPool1D(
                pool_size=2,
                strides=None,
                padding='same'
            ))
            model.add(tf.keras.layers.Dropout(rate=variables[f"dropout_{i}"]))

        model.add(tf.keras.layers.Flatten())

        for i in range(1, 3):
            if i < 2:
                variables[f"units_dense_{i}"] = hp.Choice(f"units_dense_{i}", values=parameters["units"])
                variables[f"activation_dense_{i}"] = hp.Choice(f"activation_dense_{i}", values=parameters["activation"])
            else:
                variables[f"units_dense_{i}"] = 1
                variables[f"activation_dense_{i}"] = "relu"

            variables[f"kernel_regularizer_dense_{i}"] = hp.Choice(f"kernel_regularizer_dense_{i}",
                                                                   values=parameters["regularizer"])
            variables[f"activity_regularizer_dense_{i}"] = hp.Choice(f"activity_regularizer_dense_{i}",
                                                                     values=parameters["regularizer"])
            variables[f"bias_regularizer_dense_{i}"] = hp.Choice(f"bias_regularizer_dense_{i}",
                                                                 values=parameters["regularizer"])
            variables[f"kernel_initializer_dense_{i}"] = hp.Choice(f"kernel_initializer_dense_{i}",
                                                                   values=parameters["kernel_initializer"])
            variables[f"kernel_constraint_dense_{i}"] = hp.Choice(f"kernel_constraint_dense_{i}",
                                                                  values=parameters["constraint"])
            variables[f"bias_constraint_dense_{i}"] = hp.Choice(f"bias_constraint_dense_{i}",
                                                                values=parameters["constraint"])
            variables[f"dropout_{i}"] = hp.Choice(f"dropout_{i}", values=parameters["dropout_rate"])

            model.add(tf.keras.layers.Dense(units=variables[f"units_dense_{i}"],
                                            activation=variables[f"activation_dense_{i}"],
                                            kernel_regularizer=variables[f"kernel_regularizer_dense_{i}"],
                                            activity_regularizer=variables[f"activity_regularizer_dense_{i}"],
                                            bias_regularizer=variables[f"bias_regularizer_dense_{i}"],
                                            kernel_initializer=variables[f"kernel_initializer_dense_{i}"],
                                            kernel_constraint=variables[f"kernel_constraint_dense_{i}"],

                                            bias_constraint=variables[f"bias_constraint_dense_{i}"],
                                            ))
            if i < 2:
                model.add(tf.keras.layers.Dropout(rate=variables[f"dropout_{i}"], ))

        model.compile(loss="mse", optimizer=optimizer, metrics=["mse"])
        return model



    def fit(self, hp: kt.HyperParameters, model: tf.keras.Sequential, *args, **kwargs):
        return model.fit(
            *args,
            batch_size=hp.Choice("batch_size", values=parameters["batch_size"]),
            **kwargs,
        )


def main():
    season = parse_season()
    project_name = f"cnn_{season}"

    dp = DataPipelineOffline(season=season)
    X, y = dp()
    X = np.expand_dims(X, axis=-1)
    nn_optimization_offline(project_name=project_name, X=X, y=y, hypermodel=MyHyperModel)


if __name__ == "__main__":
    main()
