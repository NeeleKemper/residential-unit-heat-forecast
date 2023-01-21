import tensorflow as tf


class LSTM:

    def __init__(self, parameters):
        self.epoch = 1000
        self.batch_size = parameters["batch_size"]
        self.optimizer = parameters["optimizer"]

        self.model = tf.keras.Sequential([])
        self.model.add(tf.keras.layers.LSTM(
            input_shape=(1, 8),
            units=parameters["units_lstm"],
            activation=parameters["activation_lstm"],
            recurrent_activation=parameters["recurrent_activation_lstm"],
            use_bias=parameters["use_bias_lstm"],
            kernel_initializer=parameters["kernel_initializer_lstm"],
            recurrent_initializer=parameters["recurrent_initializer_lstm"],
            unit_forget_bias=parameters["unit_forget_bias_lstm"],
            kernel_regularizer=parameters["kernel_regularizer_lstm"],
            recurrent_regularizer=parameters["recurrent_regularizer_lstm"],
            bias_regularizer=parameters["bias_regularizer_lstm"],
            activity_regularizer=parameters["activity_regularizer_lstm"],
            kernel_constraint=parameters["kernel_constraint_lstm"],
            recurrent_constraint=parameters["recurrent_constraint_lstm"],
            bias_constraint=parameters["bias_constraint_lstm"],
            dropout=parameters["dropout_lstm"],
            recurrent_dropout=parameters["recurrent_dropout_lstm"],
            return_sequences=True,
            return_state=False,
            go_backwards=parameters["go_backwards_lstm"],
            stateful=False,
            time_major=parameters["time_major_lstm"],
            unroll=False,
        ))
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
        self.model.add(tf.keras.layers.Dropout(parameters["dropout_1"]))
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
        x_train = x_train.reshape(x_train.shape[0], 1, x_train.shape[1])
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
        x = x.reshape(x.shape[0], 1, x.shape[1])
        return self.model.predict(x).flatten()
