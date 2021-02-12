### This is a template file meant to be a guideline to smooth out the implementation of our models in the same framework.
##	William Korcari: william.korcari@desy.de

## import of the models
#from fcn import Network		    	#import your model function
#from gcn_belle import Network
from cnn_spinodal import Network
#from particle_net import Network
##	utils.py is the file that contains all the self-built methods of this script.
from utils import train_plots
from utils import roc_auc
from utils import test_accuracy
from utils import test_f1_score

from os import chdir
#########################################
#####  EXAMPLE IMPLEMENTATION OF FCN  ###

nn = Network()

datasets = nn.compatible_datasets

for ds in datasets:

    X_train, y_train = ds.load(split="train")
    X_test, y_test = ds.load(split="test")
    x_train = nn.preprocessing(X_train)
    x_test = nn.preprocessing(X_test)
    
     
    model = nn.model(ds, shapes=nn.get_shapes(x_train))
    model.compile(**nn.compile_args)
    history = model.fit(x=x_train, y=y_train, **nn.fit_args)

    ##	From here on, one should be able to use already defined methods as showed in the following lines.
    ##	Let us know if you face any issues with that.

    # training history plots
    train_plots(history, ds.name, True)

    # evaluation plots and scores
    y_pred = model.predict(x_test).ravel()
    roc_auc(y_pred, y_test, ds.name, True)
    test_accuracy(y_pred, y_test, ds.name, nn.model_name)
    test_f1_score(y_pred, y_test, ds.name, nn.model_name)
