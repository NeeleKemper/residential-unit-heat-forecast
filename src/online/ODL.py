import tensorflow as tf


class ODL:

    def __init__(self, parameters: dict):
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

    def predict(self, x):
        return self.model.predict(x).flatten()
