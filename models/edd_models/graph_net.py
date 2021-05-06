import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

sys.path.append('..')
from erum_data_data.erum_data_data import TopTagging, Spinodal, EOSL, Belle, Airshower
from template_model.template import NetworkABC


#from ../models/utils import train_plots, roc_auc, test_accuracy, test_f1_score #test_predict_regression, plot_loss_regression


class Network(NetworkABC):

    model_name = '_graph_net_'
    build_graph = True
    def metrics(self, task):
        if task == 'regression':
            return [tf.keras.metrics.MeanSquaredError()]
        elif task == 'classification':
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


    #metrics   = [tf.keras.metrics.BinaryAccuracy(name = "acc")]  ##list of metrics to be used
    #compile_args = {'loss':'binary_crossentropy',#'categorical_crossentropy'
    #                'optimizer':tf.keras.optimizers.Adam(learning_rate=1e-3),
    #                'metrics': metrics
    #               }                      ##dictionary of the arguments to be passed to the method compile()

    callbacks = [#tf.keras.callbacks.ModelCheckpoint(filepath='./',
                 #            monitor='val_loss',
                 #            verbose=1,
                 #            save_best_only=True),
                 #tf.keras.callbacks.LearningRateScheduler(lr_schedule),
                 tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                  min_delta =0.0001,
                                                  patience=1,
                                                  verbose = 15,
                                                  restore_best_weights = True),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss",
                    factor=0.75,
                    patience=8,
                    verbose=1,
                    mode="auto",
                    min_delta=0,
                    min_lr=1e-5,
            ),
                ]                                              ##list of callbacks to be used in model.
    fit_args = {'batch_size': 256,
                'epochs': 300,
                'validation_split': 0.2,
                'shuffle': True,
                'callbacks': callbacks
               }                      ##dictionary of the arguments to be passed to the method fit()

    
    compatible_datasets = [TopTagging, Spinodal, EOSL, Belle, Airshower]          ## we would also ask you to add a list of the datasets that would be compatible with your implementation 
    
    def get_shapes(self, input_dataset):
        """
        Method should take as an input the list of datasets to be used as an iput for the model
        and after the application of all the preprocessing routine, it should return the modified data
        in the desired shapes
        """
        input_shapes = {k:input_dataset[k].shape[1:] for k in input_dataset}
        return input_shapes

    def model(self,ds,shapes):
        """
        ...
        Parameters
        ----------
        input_shapes : dict
            The shapes of each input (`points`, `features`, `mask`, `adj_matrix`).
        """
        units = 256#128
        feature_input = tf.keras.layers.Input(shape=shapes['features'], name="features")
        adjacency_input = tf.keras.layers.Input(shape=shapes['adj_matrix'], name="adj_matrix")

        adj_l = adjacency_input

        #symmetric
        #adj_l = adj_l + tf.linalg.matrix_transpose(adj_l)

        #have values either 0 or 1
        #(the adding above could produce higher values than 1)
        #adj_l = tf.cast(adj_l != 0, dtype=tf.float32)#--->  gives error, not needed

        #normalization
        #calculate outer product of degree vector and multiply with adjaceny matrix
        #deg_diag = tf.reduce_sum(adj_l, axis=2)
        #deg12_diag = tf.where(deg_diag > 0, deg_diag ** -0.5, 0)
        #adj_l = (
        #    tf.matmul(
        #        tf.expand_dims(deg12_diag, axis=2),
        #        tf.expand_dims(deg12_diag, axis=1),
        #    )
        #    * adj_l
        #)

        p = feature_input
        for i in range(3):
        #for i in range(1):
            p = tf.keras.layers.Dense(units, activation="relu")(p)
            #p = tf.keras.layers.Dropout(0.2)(p)
        
        #new to avoid overfitting
        #p = tf.keras.layers.AveragePooling1D(pool_size=2, data_format="channels_first")(p)

        #units = int(units/2)
        for i in range(3):
            p = SimpleGCN(units, activation="relu")([p, adj_l])

        x = tf.keras.layers.GlobalAveragePooling1D()(p)

        for i in range(3):
            x = tf.keras.layers.Dense(units, activation="relu")(x)

        if ds.task == 'classification':
            outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
        elif ds.task == 'regression':
            outputs = tf.keras.layers.Dense(1, activation="linear")(x)

        return tf.keras.Model(inputs=[feature_input,adjacency_input], outputs=outputs, name=ds.name)
    
    
    
    def evaluation(self, **kwargs):
        dataset = kwargs.get("dataset")
        if dataset.task == "classification":
            super().evaluation( **kwargs)
        else: # regression task
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
            
            
            
            
# evaluation methods for regression task (should probably go into utils as well)
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
    with open(path + 'scores_simple_graph_Airshower.txt', 'a') as file:
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
        plt.savefig(f"{path}{ds}_simple_graph_train_loss.png", dpi=96)
    plt.clf()



class SimpleGCN(tf.keras.layers.Layer):
    """
    Simple graph convolution. Should be equivalent to Kipf & Welling (https://arxiv.org/abs/1609.02907)
    when fed a normalized adjacency matrix.
    """

    def __init__(self, units, activation="relu"):
        super().__init__()
        self.dense = tf.keras.layers.Dense(units)
        self.activation = tf.keras.activations.get(activation)

    def call(self, inputs):
        feat, adjacency = inputs
        return self.activation(tf.matmul(adjacency, self.dense(feat)))
