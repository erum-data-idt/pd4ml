##################################
###  FCN IMPELEMTATION ###
import os
import sys
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from pd4ml.pd4ml import TopTagging, Spinodal, EOSL, Airshower, Belle

sys.path.append('..')
from template_model.template import NetworkABC

class Network(NetworkABC):
    
    model_name = '_fcn_'
    build_graph = False 
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
            monitor="val_loss", min_delta=0.001, patience=15, restore_best_weights=True
        ),

        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.75,
            patience=8,
            verbose=1,
            mode="auto",
            min_delta=0,
            min_lr=1e-5,),
    ] 
    fit_args = {
        "shuffle": True,
        "validation_split": 0.2,
        "epochs": 300,
        "callbacks": callbacks,
        "batch_size": 256,
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
        for data in in_data['features']:
            if len(data.shape[1:]) > 1:
                size = 1
                for i in range(1, len(data.shape)):
                    size *= data.shape[i]  
                out_data.append(np.reshape(data, (len(data), size)))
            else:
                out_data.append(data)    
        if len(out_data) > 2:
            return out_data[:2]
        return out_data

    def get_shapes(self, in_data):
        return [x.shape[1:] for x in in_data]


    def model(self, ds, shapes, save_model_png=False):
        input_layer = tf.keras.Input(shape = shapes[0]) 
        x = tf.keras.layers.Dense(256, activation = 'relu')(input_layer)
        for i in range (11):
            x = tf.keras.layers.Dense(256, activation = 'relu')(x)
        if ds.task == 'classification':
            output  = tf.keras.layers.Dense(1, activation = 'sigmoid')(x)
        if ds.task == 'regression':
            output  = tf.keras.layers.Dense(1, activation = 'linear')(x)
        model = tf.keras.models.Model(inputs = input_layer, outputs = output)
        
        return model

    def evaluation(self, **kwargs):
	
        dataset = kwargs.get("dataset")
        if dataset.task == "classification":
            super().evaluation( **kwargs)
        else:
            history = kwargs.pop("history")
            path = kwargs.pop("path")
            if history != None:
                plot_loss(history, path, dataset.name, True)
            x_test = kwargs.pop("x_test")
            y_test = kwargs.pop("y_test")
            model  = kwargs.pop("model")
            y_pred = model.predict(x_test)
            y_pred = y_pred.squeeze()
            test_predict(y_test, y_pred, path)



def _mean_resolution(y_true, y_pred):
    """ Metric to control for standart deviation """
    y_true = tf.cast(y_true, tf.float32)
    mean, var = tf.nn.moments((y_true - y_pred), axes=[0])
    return tf.reduce_mean(tf.sqrt(var))

def test_predict(y_test, y_pred, path):
    from sklearn.metrics import mean_squared_error as MSE
    mse_score = MSE(y_test, y_pred)
    mean_res_score = _mean_resolution(y_test,y_pred)
    _str = "Test MSE score for Airshower dataset is: {} and Resolution score is: {} \n".format(mse_score, mean_res_score)
    print(_str)
    path = os.path.join(path, "Scores/")
    if not (os.path.isdir(path)):
        os.makedirs(path)
    with open(path + 'scores_fcn_Airshower.txt', 'a') as file:
        file.write(_str)

def plot_loss(history, path, ds, save=False):
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title(ds + " model loss [training]")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "val"], loc="upper left")
    if save:
        path = os.path.join(path, "Plots/")
        if not (os.path.isdir(path)):
            os.makedirs(path)
        plt.savefig(f"{path}{ds}_fcn_train_loss.png", dpi=96)
    plt.clf()
