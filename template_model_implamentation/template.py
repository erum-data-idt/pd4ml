### This is a template file meant to be a guideline to smooth out the implementation of our models in the same framework. 


import tensorflow as tf
import numpy as np


class ModelTemplate:

    def __init__(self, dataset):
        self.ds = dataset

    my_callbacks = []                                                   ##list of callbacks to be used in model.
    my_metrics   = []                                                   ##list of metrics to be used
    my_compile_args = {'metrics': my_metrics}                      ##dictionary of the arguments to be passed to the method compile()
    my_fit_args = {'callbacks': my_callbacks}                      ##dictionary of the arguments to be passed to the method fit()

    my_compatible_datasets = []         ## we would also ask you to add a list of the datasets that would be compatible with your implementation 

    def my_preprocessing(self, in_data):
        """
        Method should take as an input the list of datasets to be used as an iput for the model
        and after the application of all the preprocessing routin, it should return the modified data
        in the desired shapes
        """
        
        #   write your preprocessing routin here
        return out_data


    def my_model(self, shapes = None, save_model_png = False):
        '''
         model should take shapes of the input datasets (not counting the number of events)
         and return the desired model
        '''
        #   write your model here
        return model



##################################
###  FCN EXAMPLE IMPELEMTATION ###

    def fcn_preprocessing(self, in_data):
        ''' in_data: numpy array. Input to be preprocessed
            returns flattened array.
        '''
        if len(in_data.shape[1:]) > 1:
            out_data = np.reshape(in_data, (len(in_data), in_data.shape[1]*in_data.shape[2]))
            return out_data
        return in_data

    def fcn(self, shape, save_model_png = False): 
        
        """
        Builds sequential model.
        ds: string. Name of the dataset that will be used as input for the model;
        shape: tuple. Shape of the dataset not counting number of events. This model can only take in datasets with one input dataset;
        save_model_png = bool. Save .png of the model in the execution dir.
        """

        if self.ds == 'airshower' or self.ds == 'belle':
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

    fcn_metrics =   [
                    tf.keras.metrics.BinaryAccuracy(name = "acc"), 
                    tf.keras.metrics.AUC(name = "AUC")
                    ]

    fcn_compile_args = {'optimizer': tf.keras.optimizers.Adam(0.001),
                        'loss': tf.keras.losses.BinaryCrossentropy(),
                        'metrics': fcn_metrics,
                        }
    fcn_callbacks = [
                        tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta =0.0001, patience=15, restore_best_weights = True),
                        tf.keras.callbacks.ModelCheckpoint('./',  monitor='val_loss', save_best_only=True, save_weights_only=True)
                        ] 
    fcn_fit_args =  {
                    'shuffle': True,
                    'validation_split': 0.2,
                    'epochs': 100,
                    'callbacks': fcn_callbacks,
                    'batch_size': 300,
                    }

    fcn_compatible_datasets = ["top", "spinodal", "EOSL"]
