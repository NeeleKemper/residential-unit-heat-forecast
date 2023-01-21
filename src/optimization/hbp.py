import tensorflow as tf
import keras_tuner as kt
import numpy as np

from src.online.HBP import HedgeBackPropagation
from src.pipeline.DataPipelineOnline import DataPipelineOnline
from src.optimization.utils import nn_optimization_online
from src.utils.utils import parse_season

tf.random.set_seed(42)
np.random.seed(42)

season = "winter"
pipeline = DataPipelineOnline(season=season)
path = f"checkpoints/hbp_{season}"


parameters = {
    "n_layers": np.arange(2, 10, 1).tolist(),
    "units": [8, 16, 32, 64, 128, 256, 512, 1024, 2048],
    "learning_rate": [1e-3, 1e-4, 1e-5],
    "activation": ["linear", "relu"],
    "kernel_initializer": ["lecun_uniform", "he_normal", "he_uniform", "glorot_normal", "glorot_uniform",
                           "normal", "zero", "ones", "uniform", "random_normal", "random_uniform"],
    "regularizer": ["l1", "l2", "l1_l2"],
    "constraint": ["unit_norm", "non_neg", "max_norm", "min_max_norm"],
}


class MyHyperModel(kt.HyperModel):

    def build(self, hp: kt.HyperParameters) -> tf.keras.Model:
        """

        :param hp:
        :return:
        """
        n_layers = hp.Choice("n_layers", values=parameters["n_layers"])
        hidden_units = hp.Choice("hidden_units", values=parameters["units"])
        activation = hp.Choice("activation", values=parameters["activation"])
        learning_rate = hp.Choice("learning_rate", values=parameters["learning_rate"])
        kernel_initializer = hp.Choice("kernel_initializer", values=parameters["kernel_initializer"])
        kernel_regularizer = hp.Choice("kernel_regularizer", values=parameters["regularizer"])
        activity_regularizer = hp.Choice("activity_regularizer", values=parameters["regularizer"])
        bias_regularizer = hp.Choice("bias_regularizer", values=parameters["regularizer"])
        kernel_constraint = hp.Choice("kernel_constraint", values = parameters["constraint"])
        bias_constraint = hp.Choice("bias_constraint", values=parameters["constraint"])

        model = HedgeBackPropagation(n_layers=n_layers,
                                     hidden_units=hidden_units,
                                     activation=activation,
                                     out_activation="relu",

                                     kernel_initializer=kernel_initializer,
                                     kernel_regularizer= kernel_regularizer,
                                     activity_regularizer= activity_regularizer,
                                     bias_regularizer= bias_regularizer,
                                     kernel_constraint= kernel_constraint,
                                     bias_constraint=bias_constraint
                                     )

        model.compile(optimizer="adam", learning_rate=learning_rate,metrics=["mse"])
        model.save_weights(path)
        return model.return_model()

    def fit(self, hp: kt.HyperParameters, model: HedgeBackPropagation, *args, **kwargs):
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
            hist_list.append(history)
            model.save_weights(path)
        return hist_list


def main():
    global pipeline
    global path
    season = parse_season()
    pipeline = DataPipelineOnline(season=season)
    path = f"checkpoints/hbp_{season}"

    project_name = f"hbp_{season}"

    nn_optimization_online(project_name=project_name, hypermodel=MyHyperModel, objective="val_loss")


if __name__ == "__main__":
    main()
