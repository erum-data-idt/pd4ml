##### a first shot at an evaluation script #####
## loads a network file, a saved model and runs the same evaluation as after training (minus history) 

import tensorflow as tf
from erum_data_data.erum_data_data import TopTagging, Spinodal, EOSL, Belle, Airshower

## import of the models
from simple_graph_net import Network
#from fcn import Network  
#from airshower_xmax import Network
#from gcn_belle import Network
#from cnn_spinodal import Network
#from particle_net import Network
#from eos_cnn import Network


#### Filepath to saved model ####
filepath = './trained_models/Airshower/Airshower_simple_graph_20210422_191633_model'
ds = Airshower       # dataset
load_graph = True   # load data as graph or not?

#############################
#############################

nn = Network()

# load data
if load_graph:
    x_test, y_test = ds.load_graph('test', path = './datasets')
else:
    X_train, y_train = ds.load(split="train")
    X_test, y_test = ds.load(split="test")
    nn.init_preprocessing(X_train)
    x_test = nn.preprocessing(X_test)

# load model
model = tf.keras.models.load_model(filepath)
print(model.summary())

# rerun evaluation
nn.evaluation(
    model=model,
    history=None,  # no history present
    dataset=ds,
    x_test=x_test,
    y_test=y_test,
    model_name = nn.model_name,
    path = filepath,
)

