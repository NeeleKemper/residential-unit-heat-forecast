# Hedge Back-Propagation
# https://towardsdatascience.com/online-deep-learning-odl-and-hedge-back-propagation-277f338a14b2
# https://arxiv.org/pdf/1711.03705.pdf
import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, Input
from keras.callbacks import Callback


class HedgeBackPropagation():
    def __init__(self, n_layers: int, hidden_units: int, activation: str, out_activation: str,
                 kernel_initializer: str, kernel_regularizer: str, activity_regularizer: str, bias_regularizer: str,
                 kernel_constraint: str, bias_constraint: str
                 ):

        outs, self.out_name, self.loss = self.create_configs(n_layers)
        out_name_loss = [s + "_loss" for s in self.out_name]
        if n_layers == 1:
            out_name_loss = ["loss"]
        self.loss_weights = [1.0 / n_layers] * n_layers
        N = n_layers
        inputs = Input((8,))

        for i in range(N):
            if i == 0:
                layer = Dense(units=hidden_units,
                              activation=activation,
                              kernel_initializer=kernel_initializer,
                              kernel_regularizer=kernel_regularizer,
                              activity_regularizer=activity_regularizer,
                              bias_regularizer=bias_regularizer,
                              kernel_constraint=kernel_constraint,
                              bias_constraint=bias_constraint
                              )(inputs)
                outs[i] = Dense(1,
                                activation=out_activation,
                                kernel_initializer=kernel_initializer,
                                kernel_regularizer=kernel_regularizer,
                                activity_regularizer=activity_regularizer,
                                bias_regularizer=bias_regularizer,
                                kernel_constraint=kernel_constraint,
                                bias_constraint=bias_constraint,
                                name=outs[i])(layer)
                continue
            layer = Dense(units=hidden_units,
                          activation=activation,
                          kernel_initializer=kernel_initializer,
                          kernel_regularizer=kernel_regularizer,
                          activity_regularizer=activity_regularizer,
                          bias_regularizer=bias_regularizer,
                          kernel_constraint=kernel_constraint,
                          bias_constraint=bias_constraint
                          )(layer)
            outs[i] = Dense(1,
                            activation=out_activation,
                            kernel_initializer=kernel_initializer,
                            kernel_regularizer=kernel_regularizer,
                            activity_regularizer=activity_regularizer,
                            bias_regularizer=bias_regularizer,
                            kernel_constraint=kernel_constraint,
                            bias_constraint=bias_constraint,
                            name=outs[i])(layer)

        self.model = Model(inputs, outs)
        # self.model.summary()
        self.my_callback = MyCallback(self.loss_weights, n_layers=n_layers, names=out_name_loss)
        self.n_layers = n_layers
        self.optimizer = None

    def return_model(self):
        return self.model

    def compile(self, optimizer: str, learning_rate: float, metrics: list):
        if optimizer=="adam":
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer == "sgd":
            self.optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
        else:
            self.optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
        self.model.compile(optimizer=self.optimizer, loss=self.loss, loss_weights=self.loss_weights, metrics=metrics)

    def fit(self, X_train, y_train, epochs=50, validation_data=None, callbacks=None, verbose=0):
        self.model.fit(X_train, y_train, epochs=epochs, callbacks=callbacks, validation_data=validation_data,
                       verbose=verbose)

    def clone(self):
        self.model = tf.keras.models.clone_model(self.model)
        self.model.compile(loss="mse", optimizer=self.optimizer, metrics=["mse"])

    def load_weights(self, path: str):
        self.model.load_weights(path)

    def save_weights(self, path: str):
        self.model.save_weights(path)

    def prediction_per_layer(self, X: np.ndarray):
        return self.model.predict(X)

    def predict(self, X: np.ndarray):
        pred_per_layer = self.prediction_per_layer(X)
        alpha = self.my_callback.weights
        weighted_average = np.average(pred_per_layer, weights=alpha, axis=0)
        return weighted_average.flatten()

    @staticmethod
    def create_configs(n_layers: int):
        outs = [""] * n_layers
        out_name = [""] * n_layers
        loss = [""] * n_layers
        for i in range(n_layers):
            outs[i] = "out" + str(i)
            out_name[i] = "out" + str(i)
            loss[i] = "mse"
        return outs, out_name, loss

    def build_data_dict(self, out_data):
        out_dict = dict((k, out_data) for k in self.out_name)
        if self.n_layers == 1:
            out_dict = {"out0": out_data}
        return out_dict


class MyCallback(Callback):
    def __init__(self, w, n_layers, beta=0.99, s=0.05, names=[]):
        self.weights = w
        self.beta = beta
        self.names = names
        self.l = []
        self.logs = dict()
        self.s = s
        self.n_layers = n_layers

    def on_train_begin(self, logs=None):
        if logs is None:
            logs = {}
        self.logs["weights"] = []

    def on_batch_end(self, batch, logs=None):
        if logs is None:
            logs = {}
        self.l.append(logs.get("loss"))
        losses = [logs[name] for name in self.names]

        M = sum(losses)
        losses = [loss / M for loss in losses]
        min_loss = np.amin(losses)
        max_loss = np.amax(losses)
        range_of_loss = max_loss - min_loss

        losses = [(loss - min_loss) / range_of_loss for loss in losses]

        # Update alpha
        alpha = [self.beta ** loss for loss in losses]
        alpha = [a * w for a, w in zip(alpha, self.weights)]

        # Smoothing alpha
        min_alpha = self.s / self.n_layers  # 0.01
        alpha = [max(min_alpha, a) for a in alpha]

        # Normalize alpha
        M = sum(alpha)
        alpha = [a / M for a in alpha]
        self.weights = alpha

    def on_batch_begin(self, epoch, logs=None):
        if logs is None:
            logs = {}
        self.model.holder = self.weights

    def on_train_end(self, epoch, logs={}):
        self.model.holder = self.weights
