##################################
###  FCN EXAMPLE IMPELEMTATION ###
from template import NetworkABC
from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np

from erum_data_data.erum_data_data import TopTagging, Spinodal, EOSL, Airshower, Belle


class Network(NetworkABC):
    
    model_name = '_fcn_'
    
    def metrics(self, task):
        if task == 'regression':
            return [tf.keras.metrics.MeanSquaredError()]
        else:
            return [tf.keras.metrics.BinaryAccuracy(name="acc"), tf.keras.metrics.AUC(name="AUC")]

    def loss(self, task):
        loss = tf.keras.losses.MeanSquaredError() if task == 'regression' else tf.keras.losses.BinaryCrossentropy() 
        return loss

    def compile_args(self, task): 
        return {
                "optimizer": tf.keras.optimizers.Adam(0.001),
                "loss": self.loss(task),
                "metrics": self.metrics(task)
               }


    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", min_delta=0.0001, patience=15, restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            "./fcn_checkpoint", monitor="val_loss", save_best_only=True, save_weights_only=True
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
            #               TopTagging, 
            #               Spinodal, 
            #               EOSL,
                           Airshower,
               #            Belle
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

    def get_shapes(self, in_data):
        return [x.shape[1:] for x in in_data]


    def model(self, ds, shapes, save_model_png=False):
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
        if ds.task == 'classification':
            output  = tf.keras.layers.Dense(1, activation = 'sigmoid')(dense[len(dense)-1])
        if ds.task == 'regression':
            output  = tf.keras.layers.Dense(1, activation = 'linear')(dense[len(dense)-1])
        model = tf.keras.models.Model(inputs = [input_layers[i] for i in range(len(input_layers))], outputs = output)
        
        return model

    def evaluation(self, **kwargs):
	
        dataset = kwargs.get("dataset")
        if dataset.task == "classification":
            super().evaluation( **kwargs)
        else:
            history = kwargs.pop("history")
            plot_loss(history, dataset.name, True)
            x_test = kwargs.pop("x_test")
            y_test = kwargs.pop("y_test")
            model  = kwargs.pop("model")
            y_pred = model.predict(x_test)
            test_predict(y_test, y_pred)
#def resolution(y_true, y_pred):
#    """ Metric to control for standart deviation """
#    mean, var = tf.nn.moments((y_true - y_pred), axes=[0])
#    return tf.sqrt(var)
def _mean_resolution(y_true, y_pred):
    """ Metric to control for standart deviation """
    y_true = tf.cast(y_true, tf.float32)
    mean, var = tf.nn.moments((y_true - y_pred), axes=[0])
    return tf.reduce_mean(tf.sqrt(var))

def test_predict(y_test, y_pred):
    from sklearn.metrics import mean_squared_error as MSE
    _str = "Test MSE score for Airshower dataset is: {} \n".format(MSE(y_test, y_pred))
    Str = "Test resolution for Airshower ds is: {} \n".format(_mean_resolution(y_test, y_pred))
    print(_str)
    with open('scores_fcn_Airshower.txt', 'a') as file:
        file.write(_str)
        file.write(Str)

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
