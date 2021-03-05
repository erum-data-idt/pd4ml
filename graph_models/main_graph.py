### This is a template file meant to be a guideline to smooth out the implementation of our models in the same framework.
##	William Korcari: william.korcari@desy.de

## import of the models
from simple_graph_net import Network
##	utils.py is the file that contains all the self-built methods of this script.
from utils import train_plots
from utils import roc_auc
from utils import test_accuracy
from utils import test_f1_score

from os import chdir
import numpy as np
import tensorflow as tf
#########################################
#####  EXAMPLE IMPLEMENTATION OF FCN  ###

nn = Network()

datasets = nn.compatible_datasets

# set to true to cache tf data datasets
# (if enough memory is available)
cache_tf_data = True

for ds in datasets:

    if not hasattr(ds, "load_graph_tf_data"):
        x_train, y_train = ds.load_graph('train', path = './datasets')
        x_test, y_test = ds.load_graph('test', path = './datasets')
        model = nn.model(ds, shapes=nn.get_shapes(x_train))
        model.compile(**nn.compile_args)
        print(model.summary())
        history = model.fit(x=x_train, y=y_train, **nn.fit_args)
        y_pred = model.predict(x_test).ravel()
    else:
        fit_args = dict(nn.fit_args)
        batch_args = dict(
            validation_split=fit_args.pop("validation_split"),
            batch_size=fit_args.pop("batch_size")
        )
        tf_ds_train, tf_ds_val = ds.load_graph_tf_data('train', path = './datasets', **batch_args)
        tf_ds_test = ds.load_graph_tf_data('test', path = './datasets', **dict(batch_args, validation_split=None))
        example_batch = next(iter(tf_ds_train))
        model = nn.model(ds, shapes=nn.get_shapes(example_batch[0]))
        model.compile(**nn.compile_args)
        print(model.summary())
        if not cache_tf_data:
            history = model.fit(tf_ds_train, validation_data=tf_ds_val, **fit_args)
        else:
            history = model.fit(
                tf_ds_train.cache().repeat(),
                validation_data=tf_ds_val.cache().repeat(),
                steps_per_epoch=int(tf.data.experimental.cardinality(tf_ds_train)),
                validation_steps=int(tf.data.experimental.cardinality(tf_ds_val)),
                **fit_args
            )
        y_pred = model.predict(tf_ds_test.map(lambda x, y: x)).ravel()
        y_test = np.concatenate([batch.numpy() for batch in tf_ds_test.map(lambda x, y: y)])

    ##	From here on, one should be able to use already defined methods as showed in the following lines.
    ##	Let us know if you face any issues with that.

    label = '_test_graph'

    test_accuracy(y_pred, y_test, ds.name+label, nn.model_name)
    test_f1_score(y_pred, y_test, ds.name+label, nn.model_name)

    train_plots(history, ds.name+label, True)
    roc_auc(y_pred, y_test, ds.name+label, True)
