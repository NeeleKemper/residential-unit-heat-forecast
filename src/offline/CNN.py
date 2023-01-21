import tensorflow as tf


class CNN:

    def __init__(self, parameters):
        self.epoch = 1000
        self.batch_size = parameters["batch_size"]
        self.optimizer = parameters["optimizer"]

        self.model = tf.keras.Sequential([])
        self.model.add(tf.keras.layers.Conv1D(
            parameters["filter_cnn_1"],
            parameters["kernel_size_cnn_1"],
            activation=parameters["activation_cnn_1"],
            use_bias=True,
            padding="same",
            kernel_initializer=parameters["kernel_initializer_cnn_1"],
            kernel_regularizer=parameters["kernel_regularizer_cnn_1"],
            bias_regularizer=parameters["bias_regularizer_cnn_1"],
            activity_regularizer=parameters["activity_regularizer_cnn_1"],
            kernel_constraint=parameters["kernel_constraint_cnn_1"],
            bias_constraint=parameters["bias_constraint_cnn_1"],
            input_shape=(8, 1)
        ))
        self.model.add(tf.keras.layers.MaxPool1D(
            pool_size=2,
            strides=None,
            padding="same",
            data_format="channels_last"))
        self.model.add(tf.keras.layers.Dropout(parameters["dropout_1"]))
        self.model.add(tf.keras.layers.Conv1D(
            parameters["filter_cnn_2"],
            parameters["kernel_size_cnn_2"],
            activation=parameters["activation_cnn_2"],
            use_bias=True,
            padding="same",
            kernel_initializer=parameters["kernel_initializer_cnn_2"],
            kernel_regularizer=parameters["kernel_regularizer_cnn_2"],
            bias_regularizer=parameters["bias_regularizer_cnn_2"],
            activity_regularizer=parameters["activity_regularizer_cnn_2"],
            kernel_constraint=parameters["kernel_constraint_cnn_2"],
            bias_constraint=parameters["bias_constraint_cnn_2"],
        ))
        self.model.add(tf.keras.layers.Dropout(parameters["dropout_2"]))
        self.model.add(tf.keras.layers.Conv1D(
            parameters["filter_cnn_3"],
            parameters["kernel_size_cnn_3"],
            activation=parameters["activation_cnn_3"],
            use_bias=True,
            padding="same",
            kernel_initializer=parameters["kernel_initializer_cnn_3"],
            kernel_regularizer=parameters["kernel_regularizer_cnn_3"],
            bias_regularizer=parameters["bias_regularizer_cnn_3"],
            activity_regularizer=parameters["activity_regularizer_cnn_3"],
            kernel_constraint=parameters["kernel_constraint_cnn_3"],
            bias_constraint=parameters["bias_constraint_cnn_3"],
        ))
        self.model.add(tf.keras.layers.Dropout(parameters["dropout_3"]))
        self.model.add(tf.keras.layers.Flatten())

        self.model.add(tf.keras.layers.Dense(
            parameters["units_dense_1"],
            activation=parameters["activation_dense_1"],
            kernel_regularizer=parameters["kernel_regularizer_dense_1"],
            kernel_initializer=parameters["kernel_initializer_dense_1"],
            bias_regularizer=parameters["bias_regularizer_dense_1"],
            activity_regularizer=parameters["activity_regularizer_dense_1"],
            kernel_constraint=parameters["kernel_constraint_dense_1"],
            bias_constraint=parameters["bias_constraint_dense_1"],
        ))
        self.model.add(tf.keras.layers.Dense(1,
                                             activation="relu",
                                             kernel_regularizer=parameters["kernel_regularizer_dense_2"],
                                             kernel_initializer=parameters["kernel_initializer_dense_2"],
                                             bias_regularizer=parameters["bias_regularizer_dense_2"],
                                             activity_regularizer=parameters["activity_regularizer_dense_2"],
                                             kernel_constraint=parameters["kernel_constraint_dense_2"],
                                             bias_constraint=parameters["bias_constraint_dense_2"]))

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
