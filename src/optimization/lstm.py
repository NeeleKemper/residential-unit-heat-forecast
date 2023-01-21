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
    "lstm_activation": ["linear", "relu", "tanh", "sigmoid"],
    "dropout_rate": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
    "kernel_initializer": ["lecun_uniform", "he_normal", "he_uniform", "glorot_normal", "glorot_uniform",
                           "normal", "zero", "ones", "uniform"],
    "recurrent_initializer": ["orthogonal", "lecun_uniform", "he_normal", "he_uniform", "glorot_normal",
                              "glorot_uniform", "normal"],
    "regularizer": ["l1", "l2", "l1_l2"],
    "constraint": ["unit_norm", "non_neg", "max_norm", "min_max_norm"],
    "use_bias": [False, True],
    "unit_forget_bias": [False, True],
    "go_backwards": [False, True],
    "time_major": [False, True]
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

        units_lstm = hp.Choice(f"units_lstm", values=parameters["units"])
        activation_lstm = hp.Choice(f"activation_lstm", values=parameters["lstm_activation"])
        recurrent_activation_lstm = hp.Choice(f"recurrent_activation_lstm", values=parameters["lstm_activation"])
        use_bias_lstm = hp.Choice(f"use_bias_lstm", values=parameters["use_bias"])
        dropout_lstm = hp.Choice(f"dropout_lstm", values=parameters["dropout_rate"])
        recurrent_dropout_lstm = hp.Choice(f"recurrent_dropout_lstm", values=parameters["dropout_rate"])
        kernel_initializer_lstm = hp.Choice("kernel_initializer_lstm", values=parameters["kernel_initializer"])
        recurrent_initializer_lstm = hp.Choice("recurrent_initializer_lstm", values=parameters["recurrent_initializer"])
        unit_forget_bias_lstm = hp.Choice("unit_forget_bias_lstm", values=parameters["unit_forget_bias"])

        recurrent_regularizer_lstm = hp.Choice("recurrent_regularizer_lstm", values=parameters["regularizer"])
        kernel_regularizer_lstm = hp.Choice("kernel_regularizer_lstm", values=parameters["regularizer"])
        bias_regularizer_lstm = hp.Choice("bias_regularizer_lstm", values=parameters["regularizer"])
        activity_regularizer_lstm = hp.Choice("activity_regularizer_lstm", values=parameters["regularizer"])

        kernel_constraint_lstm = hp.Choice("kernel_constraint_lstm", values=parameters["constraint"])
        bias_constraint_lstm = hp.Choice("bias_constraint_lstm", values=parameters["constraint"])
        recurrent_constraint_lstm = hp.Choice("recurrent_constraint_lstm", values=parameters["constraint"])

        go_backwards_lstm = hp.Choice("go_backwards_lstm", values=parameters["go_backwards"])
        time_major_lstm = hp.Choice("time_major_lstm", values=parameters["time_major"])
        model.add(tf.keras.layers.LSTM(
            input_shape=(1, 8),
            units=units_lstm,
            activation=activation_lstm,
            recurrent_activation=recurrent_activation_lstm,
            use_bias=use_bias_lstm,
            kernel_initializer=kernel_initializer_lstm,
            recurrent_initializer=recurrent_initializer_lstm,
            unit_forget_bias=unit_forget_bias_lstm,
            kernel_regularizer=kernel_regularizer_lstm,
            recurrent_regularizer=recurrent_regularizer_lstm,
            bias_regularizer=bias_regularizer_lstm,
            activity_regularizer=activity_regularizer_lstm,
            kernel_constraint=kernel_constraint_lstm,
            recurrent_constraint=recurrent_constraint_lstm,
            bias_constraint=bias_constraint_lstm,
            dropout=dropout_lstm,
            recurrent_dropout=recurrent_dropout_lstm,
            return_sequences=True,
            return_state=False,
            go_backwards=go_backwards_lstm,
            stateful=False,
            time_major=time_major_lstm,
            unroll=False,
        ))

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
                                            name=f"dense_{i}"))
            if i < 2:
                model.add(tf.keras.layers.Dropout(rate=variables[f"dropout_{i}"], name=f"dropout_{i}"))

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
    project_name = f"lstm_{season}"

    dp = DataPipelineOffline(season=season)
    X, y = dp()
    # reshape input (n_samples, 8) to (n_samples, 1, 8)
    X = X.reshape(X.shape[0], 1, X.shape[1])
    nn_optimization_offline(project_name=project_name, X=X, y=y, hypermodel=MyHyperModel)


if __name__ == "__main__":
    main()
