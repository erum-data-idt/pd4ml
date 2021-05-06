##### a first shot at an evaluation script #####
## loads a network file, a saved model and runs the same evaluation as after training (minus history) 

import tensorflow as tf
from erum_data_data.erum_data_data import TopTagging, Spinodal, EOSL, Belle, Airshower

## import of the models
from graph_net import Network
#from fcn import Network  


#### Filepath to saved model ####
filepath = './trained_models/Airshower/Airshower_graph_net_20210506_172952_model'
ds = Airshower       # dataset

#############################
#############################

nn = Network()
build_graph = nn.build_graph

# load data
x_test, y_test = ds.load_data('test', path = './datasets', graph = build_graph)
x_test = nn.preprocessing(x_test)

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

