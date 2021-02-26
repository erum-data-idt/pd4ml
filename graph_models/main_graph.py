### This is a template file meant to be a guideline to smooth out the implementation of our models in the same framework.
##	William Korcari: william.korcari@desy.de

## import of the models
#from fcn import Network		    	#import your model function
from simple_graph_net import Network
#from simple_graph_net_belle2 import Network
#from gcn_belle import Network
#from cnn_spinodal import Network
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

    X_feats, X_adj, y_train = ds.load_graph('train', path = './datasets')
    x_train = {}
    x_train['features'] = X_feats
    x_train['adj_matrix'] = X_adj

    X_feats, X_adj, y_test = ds.load_graph('test', path = './datasets')
    x_test = {}
    x_test['features'] = X_feats
    x_test['adj_matrix'] = X_adj

    model = nn.model(ds, shapes=nn.get_shapes(x_train))
    model.compile(**nn.compile_args)
    print(model.summary())
    history = model.fit(x=x_train, y=y_train, **nn.fit_args)

    ##	From here on, one should be able to use already defined methods as showed in the following lines.
    ##	Let us know if you face any issues with that.

    # training history plots
    label = 'test_graph'
    # evaluation plots and scores
    y_pred = model.predict(x_test).ravel()
    test_accuracy(y_pred, y_test, ds.name+label, nn.model_name)
    test_f1_score(y_pred, y_test, ds.name+label, nn.model_name)

    train_plots(history, ds.name+label, True)
    roc_auc(y_pred, y_test, ds.name+label, True)
