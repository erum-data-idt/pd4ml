### This is a template file meant to be a guideline to smooth out the implementation of our models in the same framework.
##	William Korcari: william.korcari@desy.de

import erum_data_data as edd

##	Please add the model generating function and the preprocessing function in that file.
# from fcn import Network		    	#import your model function
from gcn_belle import Network

##	utils.py is the file that contains all the self-built methods of this script.
from utils import train_plots
from utils import roc_auc
from utils import test_accuracy
from utils import test_f1_score


#########################################
#####  EXAMPLE IMPLEMENTATION OF FCN  ###

nn = Network()

datasets = nn.compatible_datasets

for ds in datasets:

    X_train, y_train = edd.load(ds, dataset="train", cache_dir="./" + ds, cache_subdir="datasets")
    X_test, y_test = edd.load(ds, dataset="test", cache_dir="./" + ds, cache_subdir="datasets")

    x_train = nn.preprocessing(X_train)
    x_test = nn.preprocessing(X_test)

    model = nn.model(ds, shapes=[x.shape[1:] for x in x_train])
    model.compile(**nn.compile_args)
    history = model.fit(x=x_train, y=y_train, **nn.fit_args)

    ##	From here on, one should be able to use already defined methods as showed in the following lines.
    ##	Let us know if you face any issues with that.

    # training history plots
    train_plots(history, ds, True)

    # evaluation plots and scores
    y_pred = model.predict(x_test).ravel()
    roc_auc(y_pred, y_test, ds, True)
    test_accuracy(y_pred, y_test, ds)
    test_f1_score(y_pred, y_test, ds)
