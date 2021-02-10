### This is a template file meant to be a guideline to smooth out the implementation of our models in the same framework. 
### This is a template file meant to be a guideline to smooth out the implementation of our models in the same framework. 
import tensorflow as tf
import numpy as np
from utils_particle_net import convert, Dataset, _outputs, lr_schedule
from template import NetworkABC
from erum_data_data.erum_data_data import TopTagging


class _DotDict:
    pass


class Network(NetworkABC):

    def __init__(self):
        pass


    metrics   = [tf.keras.metrics.BinaryAccuracy(name = "acc")]  ##list of metrics to be used
    compile_args = {'loss':'binary_crossentropy',#'categorical_crossentropy'
                    'optimizer':tf.keras.optimizers.Adam(learning_rate=lr_schedule(0)),
                    'metrics': metrics
                   }                      ##dictionary of the arguments to be passed to the method compile()

    callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath='./',
                             monitor='val_acc',
                             verbose=1,
                             save_best_only=True),
                 tf.keras.callbacks.LearningRateScheduler(lr_schedule),
                 #tf.compat.v1.keras.callbacks.ProgbarLogger( ), #tf.keras.callbacks.ProgbarLogger(),
                 tf.keras.callbacks.EarlyStopping(monitor='val_acc',
                                                  min_delta =0.0001,
                                                  patience=15,
                                                  restore_best_weights = True),
                ]                                              ##list of callbacks to be used in model.
    fit_args = {'batch_size': 1024,
                'epochs': 30,
                'validation_split': 0.2,
                'shuffle': True,
                'callbacks': callbacks
               }                      ##dictionary of the arguments to be passed to the method fit()

    compatible_datasets = [TopTagging]         ## we would also ask you to add a list of the datasets that would be compatible with your implementation 

    def preprocessing(self, X):
        """
        Method should take as an input the list of datasets to be used as an iput for the model
        and after the application of all the preprocessing routine, it should return the modified data
        in the desired shapes
        """
        X = np.reshape(X, (len(X), (X.shape[1]*X.shape[2])))
        v = convert(X)
        dataset = Dataset(v, data_format='channel_last')
        #   write your preprocessing routine here
        return dataset

    def get_shapes(self, dataset):
        """
        Method should take as an input the list of datasets to be used as an iput for the model
        and after the application of all the preprocessing routine, it should return the modified data
        in the desired shapes
        """
        input_shapes = {k:dataset[k].shape[1:] for k in dataset.X}
        return input_shapes


    
    def model(self, input_shapes):
        r"""ParticleNet model from `"ParticleNet: Jet Tagging via Particle Clouds"
        <https://arxiv.org/abs/1902.08570>`_ paper.
        Parameters
        ----------
        input_shapes : dict
            The shapes of each input (`points`, `features`, `mask`).
        """

        setting = _DotDict()
        setting.num_class = 2 #num_classes
        # conv_params: list of tuple in the format (K, (C1, C2, C3))
        setting.conv_params = [
            (16, (64, 64, 64)),
            (16, (128, 128, 128)),
            (16, (256, 256, 256)),
            ]
        # conv_pooling: 'average' or 'max'
        setting.conv_pooling = 'average'
        # fc_params: list of tuples in the format (C, drop_rate)
        setting.fc_params = [(256, 0.1)]
        setting.num_points = input_shapes['points'][0]

        points = tf.keras.Input(name='points', shape=input_shapes['points'])
        features = tf.keras.Input(name='features', shape=input_shapes['features']) if 'features' in input_shapes else None
        mask = tf.keras.Input(name='mask', shape=input_shapes['mask']) if 'mask' in input_shapes else None
        outputs = _outputs(points, features, mask, setting, name='top_model')

        return tf.keras.Model(inputs=[points, features, mask], outputs=outputs, name='ParticleNet')

    def model_lite(self, input_shapes):
        r"""ParticleNet-Lite model from `"ParticleNet: Jet Tagging via Particle Clouds"
    <https://arxiv.org/abs/1902.08570>`_ paper.
    Parameters
    ----------
    input_shapes : dict
        The shapes of each input (`points`, `features`, `mask`).
        """
        setting = _DotDict()
        setting.num_class = 2
        # conv_params: list of tuple in the format (K, (C1, C2, C3))
        setting.conv_params = [
            (7, (32, 32, 32)),
            (7, (64, 64, 64)),
            ]
        # conv_pooling: 'average' or 'max'
        setting.conv_pooling = 'average'
        # fc_params: list of tuples in the format (C, drop_rate)
        setting.fc_params = [(128, 0.1)]
        setting.num_points = input_shapes['points'][0]

        points = tf.keras.Input(name='points', shape=input_shapes['points'])
        features = tf.keras.Input(name='features', shape=input_shapes['features']) if 'features' in input_shapes else None
        mask = tf.keras.Input(name='mask', shape=input_shapes['mask']) if 'mask' in input_shapes else None
        outputs = _outputs(points, features, mask, setting, name='ParticleNet')

        return tf.keras.Model(inputs=[points, features, mask], outputs=outputs, name='ParticleNet_lite')
