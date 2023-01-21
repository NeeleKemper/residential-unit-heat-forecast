import tensorflow as tf


class DNN:

    def __init__(self, parameters: dict):
        self.epoch = 1000
        self.batch_size = parameters["batch_size"]
        self.optimizer = parameters["optimizer"]

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
        self.model.add(tf.keras.layers.Dropout(parameters["dropout_1"]))
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
        self.model.add(tf.keras.layers.Dropout(parameters["dropout_2"]))
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

    def clone(self):
        self.model = tf.keras.models.clone_model(self.model)
        self.model.compile(loss="mse", optimizer=self.optimizer, metrics=["mse"])

    def fit(self, x_train, y_train):
        early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss",
                                                      patience=10,
                                                      mode="auto",
                                                      restore_best_weights=True)
        self.model.fit(x_train, y_train,
                       epochs=self.epoch,
                       batch_size=self.batch_size,
                       validation_split=0.2,
                       verbose=0,
                       callbacks=[early_stop])

    def predict(self, x):
        return self.model.predict(x).flatten()
