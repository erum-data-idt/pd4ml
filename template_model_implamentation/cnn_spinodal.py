### This is a template file meant to be a guideline to smooth out the implementation of our models in the same framework. 
### This is a template file meant to be a guideline to smooth out the implementation of our models in the same framework. 
import tensorflow as tf
import numpy as np


class Network:

    def __init__(self):
        pass

    callbacks = [
                        tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta =0.0001, patience=15, restore_best_weights = True),
                        tf.keras.callbacks.ModelCheckpoint('./',  monitor='val_loss', save_best_only=True, save_weights_only=True)
                        ]                                              ##list of callbacks to be used in model.
    metrics   = [
                    tf.keras.metrics.BinaryAccuracy(name = "acc"), 
                    tf.keras.metrics.AUC(name = "AUC")
                    ]                                              ##list of metrics to be used
    compile_args = {'optimizer':tf.keras.optimizers.Adam(0.001),
              'loss':tf.keras.losses.BinaryCrossentropy(from_logits=True),
              'metrics':metrics}                      ##dictionary of the arguments to be passed to the method compile()
    fit_args = {
                    'shuffle': True,
                    'validation_split': 0.2,
                    'epochs': 200,
                    'callbacks': callbacks,
                    'batch_size': 100,
                    }                      ##dictionary of the arguments to be passed to the method fit()

    compatible_datasets = ["spinodal"]         ## we would also ask you to add a list of the datasets that would be compatible with your implementation 

    def preprocessing(self, in_data):
        """
        Method should take as an input the list of datasets to be used as an iput for the model
        and after the application of all the preprocessing routin, it should return the modified data
        in the desired shapes
        """
        out_data=np.reshape(in_data,(len(in_data),20,20,1))

        return out_data


    def model(self, ds, shape, save_model_png = False):

        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Conv2D(15, (3, 3), activation='relu',input_shape=shape,strides=(1,1), padding='same'))
        model.add(tf.keras.layers.MaxPooling2D((2, 2)))
        model.add(tf.keras.layers.Conv2D(15, (3, 3), activation='relu', padding='same'))
        model.add(tf.keras.layers.MaxPooling2D((2, 2)))
        model.add(tf.keras.layers.Conv2D(25, (3, 3), activation='relu', padding='same'))
        model.add(tf.keras.layers.Dropout(0.4))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(10, activation='relu'))
        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
        
        if save_model_png:
            tf.keras.utils.plot_model(model, to_file='./{}_model.png'.format(ds))
        
        
        return model


