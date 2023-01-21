import tensorflow as tf
import keras_tuner as kt
import numpy as np

from src.pipeline.DataPipelineOnline import DataPipelineOnline
from src.optimization.utils import nn_optimization_online
from src.utils.utils import parse_season

tf.random.set_seed(42)
np.random.seed(42)

season = "summer"
pipeline = DataPipelineOnline(season=season)
path = f"checkpoints/odl_{season}"

parameters = {
    "units": [8, 16, 32, 64, 128, 256, 512, 1024, 2048],
    "optimizer": ["sgd", "rmsprop", "adam"],
    "activation": ["linear", "relu"],
    "learning_rate": [1e-3, 1e-4, 1e-5, 1e-6],
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
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Input(shape=(8, 1)))

        optimizer = hp.Choice("optimizer", values=parameters["optimizer"])
        learning_rate = hp.Choice("learning_rate", values=parameters["learning_rate"])
        for i in range(1, 4):
            if i < 3:
                variables[f"units_{i}"] = hp.Choice(f"units_{i}", values=parameters["units"])
                variables[f"activation_{i}"] = hp.Choice(f"activation_{i}", values=parameters["activation"])
            else:
                variables[f"units_{i}"] = 1
                variables[f"activation_{i}"] = "relu"

            variables[f"kernel_regularizer_{i}"] = hp.Choice(f"kernel_regularizer_{i}",
                                                             values=parameters["regularizer"])
            variables[f"activity_regularizer_{i}"] = hp.Choice(f"activity_regularizer_{i}",
                                                               values=parameters["regularizer"])
            variables[f"bias_regularizer_{i}"] = hp.Choice(f"bias_regularizer_{i}", values=parameters["regularizer"])
            variables[f"kernel_initializer_{i}"] = hp.Choice(f"kernel_initializer_{i}",
                                                             values=parameters["kernel_initializer"])
            variables[f"kernel_constraint_{i}"] = hp.Choice(f"kernel_constraint_{i}", values=parameters["constraint"])
            variables[f"bias_constraint_{i}"] = hp.Choice(f"bias_constraint_{i}", values=parameters["constraint"])

            model.add(tf.keras.layers.Dense(units=variables[f"units_{i}"],
                                            activation=variables[f"activation_{i}"],
                                            kernel_regularizer=variables[f"kernel_regularizer_{i}"],
                                            activity_regularizer=variables[f"activity_regularizer_{i}"],
                                            bias_regularizer=variables[f"bias_regularizer_{i}"],
                                            kernel_initializer=variables[f"kernel_initializer_{i}"],
                                            kernel_constraint=variables[f"kernel_constraint_{i}"],
                                            bias_constraint=variables[f"bias_constraint_{i}"],
                                            ))

        if optimizer == "adam":
            opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer == "sgd":
            opt = tf.keras.optimizers.SGD(learning_rate=learning_rate)
        else:
            opt = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
        model.compile(loss="mse", optimizer=opt, metrics=["mse"])
        model.save_weights(path)
        return model

    def fit(self, hp: kt.HyperParameters, model: tf.keras.Sequential, *args, **kwargs):
        hist_list = list()
        for X_train, X_val, y_train, y_val, _ in pipeline():
            model.load_weights(path)
            history = model.fit(
                X_train,
                y_train,
                epochs=50,
                validation_data=(X_val, y_val),
                verbose=0
            )
            model.save_weights(path)
            hist_list.append(history)
        return hist_list


def main():
    global pipeline
    global path
    season = parse_season()
    pipeline = DataPipelineOnline(season=season)
    path = f"checkpoints/odl_{season}"

    project_name = f"odl_{season}"
    nn_optimization_online(project_name=project_name, hypermodel=MyHyperModel)


if __name__ == "__main__":
    main()
