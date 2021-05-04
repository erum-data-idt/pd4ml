import tensorflow as tf
import numpy as np

from erum_data_data.erum_data_data import Spinodal
import sys
sys.path.append("..")
from template_model.template import NetworkABC

class Network(NetworkABC):
    def __init__(self):
        pass
    model_name = '_cnn_spinodal_'
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", min_delta=0.0001, patience=15, restore_best_weights=True
        ),
#        tf.keras.callbacks.ModelCheckpoint(
 #           "./cnn_checkpoint", monitor="val_loss", save_best_only=True, save_weights_only=True
  #      ),
    ]  ##list of callbacks to be used in model.
    def metrics(self, task): return [
        tf.keras.metrics.BinaryAccuracy(name="acc"),
        tf.keras.metrics.AUC(name="AUC"),
    ]  ##list of metrics to be used
    def compile_args(self, task): return {
        "optimizer": tf.keras.optimizers.Adam(0.001),
        "loss": tf.keras.losses.BinaryCrossentropy(from_logits=True),
        "metrics": self.metrics(task),
    }  ##dictionary of the arguments to be passed to the method compile()
    fit_args = {
        "shuffle": True,
        "validation_split": 0.2,
        "epochs": 200,
        "callbacks": callbacks,
        "batch_size": 100,
    }  ##dictionary of the arguments to be passed to the method fit()

    compatible_datasets = [
        Spinodal
    ]  ## we would also ask you to add a list of the datasets that would be compatible with your implementation

    def preprocessing(self, in_data):
        """
        Method should take as an input the list of datasets to be used as an iput for the model
        and after the application of all the preprocessing routin, it should return the modified data
        in the desired shapes
        """
        out_data = np.reshape(in_data[0], (len(in_data[0]), 20, 20, 1))

        return [out_data]

    def get_shapes(self, in_data):
        return [x.shape[1:] for x in in_data]

    def model(self, ds, shapes, save_model_png=False):
        assert ds in self.compatible_datasets

        model = tf.keras.Sequential()
        model.add(
            tf.keras.layers.Conv2D(
                15, (3, 3), activation="relu", input_shape=shapes[0], strides=(1, 1), padding="same"
            )
        )
        model.add(tf.keras.layers.MaxPooling2D((2, 2)))
        model.add(tf.keras.layers.Conv2D(15, (3, 3), activation="relu", padding="same"))
        model.add(tf.keras.layers.MaxPooling2D((2, 2)))
        model.add(tf.keras.layers.Conv2D(25, (3, 3), activation="relu", padding="same"))
        model.add(tf.keras.layers.Dropout(0.4))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(10, activation="relu"))
        model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

        if save_model_png:
            tf.keras.utils.plot_model(model, to_file="./{}_model.png".format(ds.__name__))

        return model
