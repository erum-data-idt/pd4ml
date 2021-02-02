##################################
###  FCN EXAMPLE IMPELEMTATION ###


import tensorflow as tf
import numpy as np

class Fcn:

    metrics =   [
                    tf.keras.metrics.BinaryAccuracy(name = "acc"), 
                    tf.keras.metrics.AUC(name = "AUC")
                    ]

    compile_args = {'optimizer': tf.keras.optimizers.Adam(0.001),
                        'loss': tf.keras.losses.BinaryCrossentropy(),
                        'metrics': metrics,
                        }
    callbacks = [
                        tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta =0.0001, patience=15, restore_best_weights = True),
                        tf.keras.callbacks.ModelCheckpoint('./',  monitor='val_loss', save_best_only=True, save_weights_only=True)
                        ] 
    fit_args =  {
                    'shuffle': True,
                    'validation_split': 0.2,
                    'epochs': 100,
                    'callbacks': callbacks,
                    'batch_size': 300,
                    }

    compatible_datasets = ["top", "spinodal", "EOSL"]

    def preprocessing(self, in_data):
        ''' in_data: numpy array. Input to be preprocessed
            returns flattened array.
        '''
        if len(in_data.shape[1:]) > 1:
            out_data = np.reshape(in_data, (len(in_data), in_data.shape[1]*in_data.shape[2]))
            return out_data
        return in_data

    def model(self, ds, shape, save_model_png = False): 
        
        """
        Builds sequential model.
        ds: string. Name of the dataset that will be used as input for the model;
        shape: tuple. Shape of the dataset not counting number of events. This model can only take in datasets with one input dataset;
        save_model_png = bool. Save .png of the model in the execution dir.
        """

        if ds == 'airshower' or ds == 'belle':
            print("Dataset {} has too many input datasets.".format(self.ds))
            return

        model = tf.keras.Sequential()
        model.add(tf.keras.Input(shape = shape))
        tf.keras.layers.BatchNormalization()
        for l in range(15):
            model.add(tf.keras.layers.Dense(256, activation = 'relu'))  ##add hidden layers
        model.add(tf.keras.layers.Dense(1, activation = 'sigmoid'))         ##add output layer
    
        if save_model_png:
            tf.keras.utils.plot_model(model, to_file='./{}_model.png'.format(ds))

        return model

