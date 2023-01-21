import collections
import numpy as np
import tensorflow as tf


class ODLER:

    def __init__(self, season: str, cfg: dict):

        self.batch_size = cfg["batch_size"]
        capacity = cfg["capacity"]
        factor = cfg["factor"]
        min_recollection = cfg["min_recollection"]

        if season == "summer":
            parameters = {'optimizer': 'adam', 'learning_rate': 0.001, 'units_1': 512, 'activation_1': 'relu',
                          'kernel_regularizer_1': 'l2', 'activity_regularizer_1': 'l1', 'bias_regularizer_1': 'l1',
                          'kernel_initializer_1': 'random_uniform', 'kernel_constraint_1': 'non_neg',
                          'bias_constraint_1': 'min_max_norm', 'units_2': 512, 'activation_2': 'linear',
                          'kernel_regularizer_2': 'l2', 'activity_regularizer_2': 'l2', 'bias_regularizer_2': 'l2',
                          'kernel_initializer_2': 'normal', 'kernel_constraint_2': 'min_max_norm',
                          'bias_constraint_2': 'unit_norm', 'kernel_regularizer_3': 'l2',
                          'activity_regularizer_3': 'l1_l2', 'bias_regularizer_3': 'l1_l2',
                          'kernel_initializer_3': 'glorot_uniform', 'kernel_constraint_3': 'unit_norm',
                          'bias_constraint_3': 'max_norm'}
        elif season == "winter":
            parameters = {'optimizer': 'adam', 'learning_rate': 0.001, 'units_1': 512, 'activation_1': 'relu',
                          'kernel_regularizer_1': 'l2', 'activity_regularizer_1': 'l1', 'bias_regularizer_1': 'l1',
                          'kernel_initializer_1': 'random_uniform', 'kernel_constraint_1': 'non_neg',
                          'bias_constraint_1': 'min_max_norm', 'units_2': 512, 'activation_2': 'linear',
                          'kernel_regularizer_2': 'l2', 'activity_regularizer_2': 'l2', 'bias_regularizer_2': 'l2',
                          'kernel_initializer_2': 'normal', 'kernel_constraint_2': 'min_max_norm',
                          'bias_constraint_2': 'unit_norm', 'kernel_regularizer_3': 'l2',
                          'activity_regularizer_3': 'l1_l2', 'bias_regularizer_3': 'l1_l2',
                          'kernel_initializer_3': 'glorot_uniform', 'kernel_constraint_3': 'unit_norm',
                          'bias_constraint_3': 'max_norm'}
        else:
            parameters = {'optimizer': 'adam', 'learning_rate': 0.001, 'units_1': 512, 'activation_1': 'relu',
                          'kernel_regularizer_1': 'l2', 'activity_regularizer_1': 'l1', 'bias_regularizer_1': 'l1',
                          'kernel_initializer_1': 'random_uniform', 'kernel_constraint_1': 'non_neg',
                          'bias_constraint_1': 'min_max_norm', 'units_2': 512, 'activation_2': 'linear',
                          'kernel_regularizer_2': 'l2', 'activity_regularizer_2': 'l2', 'bias_regularizer_2': 'l2',
                          'kernel_initializer_2': 'normal', 'kernel_constraint_2': 'min_max_norm',
                          'bias_constraint_2': 'unit_norm', 'kernel_regularizer_3': 'l2',
                          'activity_regularizer_3': 'l1_l2', 'bias_regularizer_3': 'l1_l2',
                          'kernel_initializer_3': 'glorot_uniform', 'kernel_constraint_3': 'unit_norm',
                          'bias_constraint_3': 'max_norm'}

        self.optimizer = parameters["optimizer"]
        self.learning_rate = parameters["learning_rate"]

        self.model = tf.keras.Sequential([])
        self.model.add(tf.keras.layers.Dense(
            units=parameters["units_1"],
            activation=parameters["activation_1"],
            kernel_regularizer=parameters["kernel_regularizer_1"],
            activity_regularizer=parameters[f"activity_regularizer_1"],
            bias_regularizer=parameters["bias_regularizer_1"],
            kernel_initializer=parameters["kernel_initializer_1"],
            kernel_constraint=parameters[f"kernel_constraint_1"],
            bias_constraint=parameters[f"bias_constraint_1"],
            input_shape=(8,)
        ))
        self.model.add(tf.keras.layers.Dense(
            units=parameters["units_2"],
            activation=parameters["activation_2"],
            kernel_regularizer=parameters["kernel_regularizer_2"],
            activity_regularizer=parameters[f"activity_regularizer_2"],
            bias_regularizer=parameters["bias_regularizer_2"],
            kernel_initializer=parameters["kernel_initializer_2"],
            kernel_constraint=parameters[f"kernel_constraint_2"],
            bias_constraint=parameters[f"bias_constraint_2"],
        ))
        self.model.add(tf.keras.layers.Dense(
            units=1,
            activation="relu",
            kernel_regularizer=parameters["kernel_regularizer_3"],
            activity_regularizer=parameters[f"activity_regularizer_3"],
            bias_regularizer=parameters["bias_regularizer_3"],
            kernel_initializer=parameters["kernel_initializer_3"],
            kernel_constraint=parameters[f"kernel_constraint_3"],
            bias_constraint=parameters[f"bias_constraint_3"],
        ))
        self.experience_replay = ER(capacity, factor, min_recollection)
        self.compile(self.optimizer, self.learning_rate, ["mse"])

    def return_model(self):
        return self.model

    def compile(self, optimizer: str, learning_rate: float, metrics: list):
        if optimizer == "adam":
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer == "sgd":
            self.optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
        else:
            self.optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
        self.model.compile(loss="mse", optimizer=self.optimizer, metrics=metrics)

    def fit(self, X_train, y_train, epochs=1000, validation_split=0.2, callbacks=None, verbose=0):
        X_train, y_train = self.experience_replay.draw_memory(X_train, y_train)

        if callbacks is None:
            callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, mode='auto',
                                                          restore_best_weights=True)]
        self.model.fit(X_train, y_train, epochs=epochs, callbacks=callbacks, validation_split=validation_split,
                       verbose=verbose)

    def clone(self):
        self.model = tf.keras.models.clone_model(self.model)
        self.model.compile(loss="mse", optimizer=self.optimizer, metrics=["mse"])

    def load_weights(self, path: str):
        self.model.load_weights(path)

    def save_weights(self, path: str):
        self.model.save_weights(path)

    def predict(self, x):
        return self.model.predict(x).flatten()


class ER:
    def __init__(self, capacity: int, factor: float, min_recollection: int):
        self.buffer = collections.deque(maxlen=capacity)
        self.min_recollection = min_recollection
        self.factor = factor

    def __len__(self):
        """

        :return:
        """
        return len(self.buffer)

    def append(self, X_train: np.ndarray, y_train: np.ndarray):
        """

        :param X_train:
        :param y_train:
        :return:
        """
        [self.buffer.append([x, y]) for x, y in zip(X_train, y_train)]

    def draw_memory(self, X_train: np.ndarray, y_train: np.ndarray):
        X, y = list(), list()
        if len(self.buffer) > self.min_recollection:
            recollection = max(self.min_recollection, int(len(self.buffer) * self.factor))
            indices = np.random.choice(len(self.buffer), recollection, replace=False)
            train_data = np.array(self.buffer, dtype=object)[indices]
            for element in train_data:
                X.append(element[0])
                y.append(element[1])
        self.append(X_train, y_train)
        [X.append(x_t) for x_t in X_train]
        [y.append(y_t) for y_t in y_train]
        return np.array(X), np.array(y)
