##################################
###  FCN EXAMPLE IMPELEMTATION ###
from template import NetworkABC

import tensorflow as tf
import numpy as np

from erum_data_data.erum_data_data import TopTagging, Spinodal, EOSL, Airshower, Belle


class Network(NetworkABC):

    metrics = [tf.keras.metrics.BinaryAccuracy(name="acc"), tf.keras.metrics.AUC(name="AUC")]

    compile_args = {
        "optimizer": tf.keras.optimizers.Adam(0.001),
        "loss": tf.keras.losses.BinaryCrossentropy(),
        "metrics": metrics,
    }
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", min_delta=0.0001, patience=15, restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            "./", monitor="val_loss", save_best_only=True, save_weights_only=True
        ),
    ]
    fit_args = {
        "shuffle": True,
        "validation_split": 0.2,
        "epochs": 100,
        "callbacks": callbacks,
        "batch_size": 300,
    }

    compatible_datasets = [
                           TopTagging, 
                           Spinodal, 
                           EOSL,
                           Airshower,
                           Belle
                          ]

   


    def preprocessing(self, in_data):
        """in_data: list of arrays. Input to be preprocessed
        returns list of flattened array.
        """
        out_data = []
        for data in in_data:
            if len(data.shape[1:]) > 1:
                size = 1
                for i in range(1, len(data.shape)):
                    size *= data.shape[i]  
                out_data.append(np.reshape(data, (len(data), size)))#data.shape[1] * data.shape[2])))
            else:
                out_data.append(data)    
        if len(out_data) > 2:
            return out_data[:2]
        return out_data


    def model_all(self, ds, shapes, save_model_png=False):
        input_layers = {}
        dense_layers = []
        for i, shape in enumerate(shapes):
            if len(shape) > 0:
                input_layers[i] = tf.keras.Input(shape = shape) 
                dense_layers.append({})
                dense_layers[i][0] = tf.keras.layers.Dense(256, activation = 'relu')(input_layers[i])
                for j in range(1, 4):
                    dense_layers[i][j] = tf.keras.layers.Dense(256, activation = 'relu')(dense_layers[i][j-1])
        dense = {}
        if len(input_layers) > 1:
            merged = tf.keras.layers.Concatenate(axis=1)([dense_layers[i][3] for i in range(len(dense_layers))])
            dense[0] = tf.keras.layers.Dense(256, activation = 'relu')(merged)
        else: 
            dense[0] = tf.keras.layers.Dense(256, activation = 'relu')(input_layers[0])
        for i in range (1, 10):
            dense[i] = tf.keras.layers.Dense(256, activation = 'relu')(dense[i-1])
        output  = tf.keras.layers.Dense(1, activation = 'sigmoid')(dense[len(dense)-1])
        model = tf.keras.models.Model(inputs = [input_layers[i] for i in range(len(input_layers))], outputs = output)
        
        return model
        
        
        

    def model(self, ds, shapes, save_model_png=False):

        """
        Builds sequential model.
        ds: string. Name of the dataset that will be used as input for the model;
        shape: tuple. Shape of the dataset not counting number of events. This model can only take in datasets with one input dataset;
        save_model_png = bool. Save .png of the model in the execution dir.
        """

        assert ds in self.compatible_datasets

        model = tf.keras.Sequential()
        model.add(tf.keras.Input(shape=shapes[0]))
        tf.keras.layers.BatchNormalization()
        for _ in range(15):
            model.add(tf.keras.layers.Dense(256, activation="relu"))  # add hidden layers
        model.add(tf.keras.layers.Dense(1, activation="sigmoid"))  # add output layer

        if save_model_png:
            tf.keras.utils.plot_model(model, to_file="./{}_model.png".format(ds.name))

        return model
