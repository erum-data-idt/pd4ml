from template import NetworkABC

import tensorflow as tf
import numpy as np

from erum_data_data.erum_data_data import EOSL

class Network(NetworkABC):
	model_name = '_cnnEOS_'
	metrics = [tf.keras.metrics.BinaryAccuracy(name="acc")]
	compile_args = {'loss': tf.keras.losses.binary_crossentropy,
					'optimizer': tf.keras.optimizers.Adamax(lr=0.0001),
					'metrics': metrics
					}
	compatible_datasets = [EOSL]
	callbacks = [
                    tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss", min_delta=0.0001, patience=150, restore_best_weights=True
                    ),
                    tf.keras.callbacks.ModelCheckpoint(
                    "./eosCnn_checkpoint", monitor="val_loss", save_best_only=True, save_weights_only=True
                    ),
                    ]
	fit_args = {'epochs': 1000, 
		        'shuffle': True,
		        'batch_size': 32,
				'validation_split': 0.2,
				'callbacks': callbacks
				}
	def preprocessing(self, x):
		x = x[0]
		if tf.keras.backend.image_data_format() == 'channels_first':
			x = x.reshape(x.shape[0], 1, x.shape[1], x.shape[2])
		else:
			x = x.reshape(x.shape[0], x.shape[1], x.shape[2], 1)
		return x

	def get_shapes(self, x):
		shapes = x.shape[1:]
		return shapes

	def model(self, ds, shapes):
		model = tf.keras.Sequential(name=ds.name+self.model_name)

		model.add(tf.keras.layers.Conv2D(16, (8, 8), padding='same', kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(0.00001), input_shape=shapes))
		model.add(tf.keras.layers.BatchNormalization(axis=-1))
		model.add(tf.keras.layers.PReLU())

		model.add(tf.keras.layers.Dropout(0.2))

		model.add(tf.keras.layers.Conv2D(16, (7, 7), padding='same', kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(0.00001)))
		model.add(tf.keras.layers.AveragePooling2D((2,2)))
		model.add(tf.keras.layers.BatchNormalization())
		model.add(tf.keras.layers.PReLU())

		model.add(tf.keras.layers.Dropout(0.2))

		model.add(tf.keras.layers.Conv2D(32, (6, 6), padding='same', kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(0.00001)))
		model.add(tf.keras.layers.AveragePooling2D((2,2)))
		model.add(tf.keras.layers.BatchNormalization(axis=-1))
		model.add(tf.keras.layers.PReLU())

		model.add(tf.keras.layers.Dropout(0.2))

		model.add(tf.keras.layers.Flatten())

		model.add(tf.keras.layers.Dense(128, kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(0.00001)))
		model.add(tf.keras.layers.BatchNormalization())
		model.add(tf.keras.layers.PReLU())

		model.add(tf.keras.layers.Dropout(0.5))

		model.add(tf.keras.layers.Dense(1, kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(0.00001), activation='sigmoid'))

		return model


