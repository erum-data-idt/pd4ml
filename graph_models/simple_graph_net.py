import tensorflow as tf
import numpy as np

from template import NetworkABC
from erum_data_data.erum_data_data import TopTagging, Spinodal, EOSL



class Network(NetworkABC):

    model_name = '_simple_graph_'

    def __init__(self):
        pass


    metrics   = [tf.keras.metrics.BinaryAccuracy(name = "acc")]  ##list of metrics to be used
    compile_args = {'loss':'binary_crossentropy',#'categorical_crossentropy'
                    'optimizer':tf.keras.optimizers.Adam(learning_rate=1e-3),
                    'metrics': metrics
                   }                      ##dictionary of the arguments to be passed to the method compile()

    callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath='./',
                             monitor='val_acc',
                             verbose=1,
                             save_best_only=True),
                 #tf.keras.callbacks.LearningRateScheduler(lr_schedule),
                 tf.keras.callbacks.EarlyStopping(monitor='val_acc',
                                                  min_delta =0.0001,
                                                  patience=15,
                                                  restore_best_weights = True),
                ]                                              ##list of callbacks to be used in model.
    fit_args = {'batch_size': 1024,
                'epochs': 200,
                'validation_split': 0.2,
                'shuffle': True,
                'callbacks': callbacks
               }                      ##dictionary of the arguments to be passed to the method fit()

    compatible_datasets = [Spinodal]          ## we would also ask you to add a list of the datasets that would be compatible with your implementation 

    '''
    def preprocessing(self, X):
        """
        Method should take as an input the list of datasets to be used as an iput for the model
        and after the application of all the preprocessing routine, it should return the modified data
        in the desired shapes
        """
        X[0] = X[0][:,0:max_part,:]
        X = np.reshape(X[0], (len(X[0]), (X[0].shape[1]*X[0].shape[2])))
        v = convert(X)
        dataset = Dataset(v, data_format='channel_last')
        return dataset.X
    '''
    
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

        outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

        return tf.keras.Model(inputs=[feature_input,adjacency_input], outputs=outputs, name=ds.name)


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
