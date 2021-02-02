### This is a template file meant to be a guideline to smooth out the implementation of our models in the same framework. 
##	William.

import erum_data_data as edd
import tensorflow as tf

##	utils.py is the file that contains all the self-built methods of this script.
##	Please add the model generating function and the preprocessing function in that file.
from template import ModelTemplate		    	#import your model function

from utils import train_plots
from utils import roc_auc
from utils import test_accuracy
from utils import test_f1_score

datasets = ["top", "spinodal", "EOSL", "aorshower", "LHCO", "belle"]

ds = 'top' ## Choose your dataset among those listed above.

X_train, y_train  = edd.load(ds, dataset='train', cache_dir = './'+ ds, cache_subdir = 'datasets')
X_test, y_test = edd.load(ds, dataset='test', cache_dir = './'+ ds, cache_subdir = 'datasets')
#########################################
#####  EXAMPLE IMPLEMENTATION OF FCN  ###


fcn_model = ModelTemplate(ds)

fcn_train = fcn_model.fcn_preprocessing(X_train[0])
fcn_test = fcn_model.fcn_preprocessing(X_test[0])

print(type(fcn_train), fcn_train.shape)

fcn = fcn_model.fcn(shape = fcn_train.shape[1:])

fcn.compile(**fcn_model.fcn_compile_args)
history = fcn.fit(x = fcn_train, y = y_train, **fcn_model.fcn_fit_args)

##	From here on, one should be able to use already defined methods as showed in the following lines. 
##	Let me know if you face any issues with that.

#training history plots
train_plots(history, ds, True)

#evaluation plots and scores
y_pred = model.predict(X_test).ravel()
roc_auc(y_pred, y_test, ds, True)
test_accuracy(y_pred, y_test, ds)
test_f1_score(y_pred, y_test, ds)

model.load_weights(checkpoint_path)

