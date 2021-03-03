from functools import cached_property
from template import NetworkABC
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from erum_data_data.erum_data_data import Airshower


def resolution(y_true, y_pred):
    """ Metric to control for standart deviation """
    mean, var = tf.nn.moments((y_true - y_pred), axes=[0])
    return tf.sqrt(var)


def NormToLabel(stats, factor=1, name=None):
    return tf.keras.layers.Lambda(lambda x: factor * stats["std"] * x + stats["mean"], name=name)


def recurrent_block(input, nodes=10):
    # fmt: off
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.TimeDistributed(
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True))))(input)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.TimeDistributed(
        tf.keras.layers.LSTM(nodes, return_sequences=False)))(x)
    # fmt: on
    return tf.keras.layers.Reshape((x.shape[1], x.shape[2], nodes))(x)


def residual_unit(input, nfilter, f_size=3, bottleneck=False, batchnorm=True):
    if bottleneck:
        # bottleneck layer
        input = tf.keras.layers.Conv2D(nfilter, 1, padding="same", activation="relu")(input)
    x = tf.keras.layers.Conv2D(nfilter, f_size, padding="same")(input)
    if batchnorm:
        x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Conv2D(nfilter, f_size, padding="same")(x)
    if batchnorm:
        x = tf.keras.layers.BatchNormalization()(x)
    out = tf.keras.layers.Add()([x, input])
    return tf.keras.layers.Activation("relu")(out)


def ResNet(input):
    x = residual_unit(input, 64, bottleneck=True)
    x = residual_unit(x, 64)
    x = residual_unit(x, 64)
    x = residual_unit(x, 64)
    x = residual_unit(x, 64)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = residual_unit(x, 128, bottleneck=True)
    x = residual_unit(x, 128)
    x = residual_unit(x, 128)
    x = residual_unit(x, 128)
    x = residual_unit(x, 128)
    return x


def norm_time(time, std=None):
    u = np.isnan(time)
    time -= np.nanmean(time, axis=(1, 2, 3), keepdims=True)

    if std is None:
        std = np.nanstd(time)

    time /= std
    time[u] = 0
    return time, std


def norm_signal(signal):
    signal[np.less(signal, 0)] = 0
    signal = np.log10(signal + 1)
    signal[np.isnan(signal)] = 0
    return signal


class Network(NetworkABC):
    compatible_datasets = [Airshower]

    @property
    def callbacks(self):
        return [
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=4,
                verbose=1,
                mode="auto",
                min_delta=0,
                min_lr=1e-4,
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                min_delta=0.0001,
                patience=10,
                restore_best_weights=True,
            ),
        ]

    @property
    def metrics(self):
        return [resolution]

    @property
    def compile_args(self):
        return dict(
            super().compile_args,
            loss="mse",
            optimizer=tf.keras.optimizers.Adam(lr=3e-3, amsgrad=True),
        )

    @property
    def fit_args(self):
        return dict(
            super().fit_args,
            batch_size=32,
            epochs=50,
            verbose=1,
            validation_split=0.1,
            shuffle=True,
        )

    def preprocessing(self, in_data):
        signal, time, _ = in_data
        time, _ = norm_time(time, self.std)
        signal = norm_signal(signal)
        return [signal, time]

    @cached_property
    def stats(self):
        _, y_train = Airshower.load(split="train")
        return dict(
            std=np.std(y_train),
            mean=np.mean(y_train),
        )

    @cached_property
    def std(self):
        x_train, _ = Airshower.load(split="train")
        _, time, _ = x_train
        _, std = norm_time(time)
        return std

    def get_shapes(self, in_data):
        # assume (see: preprocessing):
        #   in_data[0] := signal
        #   in_data[1] := time
        return [in_data[0].shape[1:], in_data[1].shape[1:]]

    def model(self, ds, shapes):
        assert ds in self.compatible_datasets

        # Input Cube from time traces
        sig_in = tf.keras.layers.Input(shape=shapes[0], name="signal")
        # T0 input
        time_in = tf.keras.layers.Input(shape=shapes[1], name="time")
        TimeProcess = tf.keras.layers.Reshape(sig_in.shape.as_list()[1:] + [1])(sig_in)
        # Time trace characterization
        TimeProcess = recurrent_block(TimeProcess)
        x = tf.keras.layers.Concatenate()([TimeProcess, time_in])
        x = ResNet(x)
        x = tf.keras.layers.GlobalMaxPooling2D()(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        x = tf.keras.layers.Dense(1)(x)
        output = NormToLabel(self.stats)(x)
        return tf.keras.Model(inputs=[sig_in, time_in], outputs=output)

    def evaluation(self, **kwargs):
        history = kwargs.pop("history")
        dataset_name = kwargs.pop("dataset_name")
        plot_loss(history, dataset_name, True)


def plot_loss(history, ds, save=False):
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title(ds + " model loss [training]")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "val"], loc="upper left")
    if save:
        plt.savefig(f"{ds}_train_loss.png", dpi=96)
    # plt.show()
    plt.clf()