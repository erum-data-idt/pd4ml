import sys
import time
from abc import ABCMeta, abstractmethod
from typing import List, Dict
sys.path.append("..")
from utils.utils import train_plots, roc_auc, test_accuracy, test_f1_score
from template_model.template import NetworkABC

import numpy as np
import os
import sklearn
import h5py
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

import tensorflow as tf
from pd4ml.pd4ml import TopTagging, Spinodal, EOSL, Belle, Airshower


###########################
#Model
###########################

class Conv1D_bn(tf.keras.layers.Layer):
    """ Conv1D Layer with batch Normalization """
    def __init__(self,num_outputs,activation=None,bn=True,use_bias=True,**kwargs):
        super(Conv1D_bn, self).__init__()
        self.num_outputs = num_outputs
        self.activation = activation
        self.use_bias = use_bias
        self.bn = bn
    
    def build(self,input_shapes):
        self.conv1D = tf.keras.layers.Conv1D(self.num_outputs,1,strides=1,padding="valid",activation=None,use_bias=self.use_bias)
        self.batchNorm = tf.keras.layers.BatchNormalization()
        self.activationLayer = tf.keras.layers.Activation(activation=self.activation)
    
    def call(self,inputs):
        output = self.conv1D(inputs)
        if(self.bn): output = self.batchNorm(output)
        output = self.activationLayer(output)

        return output


class SA_Layer(tf.keras.layers.Layer):
    """ Self attention layer
        args:
            num_outputs: int
        inputs:
            [fts,mask]
            fts: (batch_size, num_points, num_outputs)
            mask: (batch_size, num_points)
     """
    def __init__(self, num_outputs):
        super(SA_Layer, self).__init__()
        self.num_outputs = num_outputs

    def build(self,inputs):
        self.query = Conv1D_bn(self.num_outputs//4,activation=None,use_bias=False)
        self.key = Conv1D_bn(self.num_outputs//4,activation=None,use_bias=False)
        self.value = Conv1D_bn(self.num_outputs,activation=None,use_bias=False)
        self.off = Conv1D_bn(self.num_outputs,activation="relu",use_bias=False)
    
    def compute_mask(self, inputs, mask=None):
        return None


    def call(self,inputs):
        tensor_inputs = inputs[0]
        mask_inputs = inputs[1]

        query = self.query(tensor_inputs)

        key = self.query(tensor_inputs)
        key = tf.transpose(key,perm=[0,2,1])

        value = self.value(tensor_inputs)
        value = tf.transpose(value,perm=[0,2,1])

        energy = tf.matmul(query,key)

        if mask_inputs == None:

            attention = tf.nn.softmax(energy)
            self_att = tf.matmul(value,attention)
            self_att = tf.transpose(self_att, perm=[0,2,1])

        else : 
            #Make zero-padded less important
            mask_offset = -1000*mask_inputs+tf.ones_like(mask_inputs)
            mask_matrix = tf.matmul(tf.expand_dims(mask_offset,-1),tf.transpose(tf.expand_dims(mask_offset,-1),perm=[0,2,1]))
            mask_matrix = mask_matrix - tf.ones_like(mask_matrix)
            energy = energy + mask_matrix

            attention = tf.nn.softmax(energy)

            zero_mask = tf.where(tf.equal(mask_matrix,0),tf.ones_like(mask_matrix),tf.zeros_like(mask_matrix))  

            attention = attention*zero_mask

            attention = attention / (1e-9 + tf.reduce_sum(attention,1, keepdims=True))
            self_att = tf.matmul(value,attention)
            self_att = tf.transpose(self_att, perm=[0,2,1])

        #offset
        off = self.off(tensor_inputs-self_att)
        output = tensor_inputs+off

        return output, attention




## Edge Conv ##



def knn(adj_matrix, k=20):
    """Get KNN based on the distance matrix.
    Args:
      pairwise distance: (batch_size, num_points, num_points)
      k: int

    Returns:
      nn_idx: (batch_size, num_points, k)
    """
    neg_adj = -adj_matrix
    _, nn_idx = tf.nn.top_k(neg_adj, k=k)  # values, indices
    return nn_idx


def get_edge_feature(point_cloud, nn_idx, k=20):
    """Construct inputs for convolution
    Args:
    point_cloud: (batch_size, num_points, num_dims) 
    nn_idx: (batch_size, num_points, k)
    k: int

    Returns:
    edge features: (batch_size, num_points, k, 2*num_dims)
    """

    batch_size = tf.shape(point_cloud)[0]
    num_points = tf.shape(point_cloud)[1]
    features = point_cloud
    batch_indices = tf.tile(tf.reshape(tf.range(batch_size), (-1, 1, 1, 1)), (1, num_points, k, 1))
    indices = tf.concat([batch_indices, tf.expand_dims(nn_idx, axis=3)], axis=3)  # (N, P, K, 2)
    knn_fts = tf.gather_nd(features, indices) #for each particle: features of every knn particle 
    knn_fts_center = tf.tile(tf.expand_dims(features, axis=2), (1, 1, k, 1))  # (N, P, K, C)
    edge_feature = tf.concat([knn_fts_center, tf.subtract(knn_fts, knn_fts_center)], axis=-1)  # (N, P, K, 2*C)
    return edge_feature

class Conv2D_bn(tf.keras.layers.Layer):
    """ Conv2D Layer with batch normalization """
    def __init__(self, num_outputs, activation=None, bn=True, use_bias=True, **kwargs):
        super(Conv2D_bn, self).__init__()
        self.num_outputs = num_outputs
        self.activation = activation
        self.use_bias = use_bias
        self.bn = bn

    def build(self, input_shapes):
        self.conv2D = tf.keras.layers.Conv2D(self.num_outputs, [1, 1], strides=[
                                             1, 1], padding="valid", activation=None, use_bias=self.use_bias)
        self.batchNorm = tf.keras.layers.BatchNormalization()
        self.activationLayer = tf.keras.layers.Activation(activation=self.activation)

    def call(self, inputs):
        output = self.conv2D(inputs)
        if(self.bn):
            output = self.batchNorm(output)
        output = self.activationLayer(output)
        return output


class EdgeConv_Block(tf.keras.layers.Layer):
    """ EdgeConv Block 
        args:
            num_outputs: int
            k: int
        inputs:
            [feature_input,adjacency_input]
            feature_input: (batch_size, num_points, num_dims)
            adjaency_input: (batch_size, num_points, num_points)
    """
    def __init__(self, num_outputs, k, first=False):
        super(EdgeConv_Block, self).__init__()
        self.num_outputs = num_outputs
        self.k = k

    def build(self, inputs):
        self.conv2d = Conv2D_bn(self.num_outputs, activation="relu")
        self.conv2d_1 = Conv2D_bn(self.num_outputs, activation="relu")

    def call(self, inputs):
        tensor_inputs = inputs[0]
        adj_inputs = inputs[1]

            
        #use adj matrix with feature difference from first feature
        #calculate difference between each feature of each particle
        num_points = tf.shape(tensor_inputs)[1]
        feature = tensor_inputs[:,:,0]
        
        
        minuend = tf.repeat(feature,num_points,axis=1)
        minuend = tf.reshape(minuend,[-1,num_points,num_points]) #shape: (batch_size,100,100)

        subtrahend = tf.tile(feature,[1,num_points])
        subtrahend = tf.reshape(subtrahend,[-1,num_points,num_points]) #shape: (batch_size,100,100)
        
        feature_difference_matrix = tf.abs(tf.subtract(subtrahend,minuend))
        adj = tf.multiply(adj_inputs,feature_difference_matrix)
        
        
        
        
        nn_idx = knn(adj, k=self.k)
        edge_feature_0 = get_edge_feature(
            tensor_inputs, nn_idx=nn_idx, k=self.k)

        #local features
        fts = self.conv2d(edge_feature_0)
        fts = self.conv2d_1(fts)


        features_0 = tf.reduce_mean(fts, axis=-2, keepdims=True)
        features_0 = tf.squeeze(features_0, axis=2)
        return features_0


### Models ###

def get_PCT(ds,shapes=None):
    """ Returns a point cloud transformer model  """

    feature_input = tf.keras.layers.Input(shape=shapes['features'], name="features")
    adjacency_input = tf.keras.layers.Input(shape=shapes['adj_matrix'], name="adj_matrix")


    boolean_mask = feature_input[:,:,0]==0
    mask = tf.cast(boolean_mask,float)

    k=20

    edge1 = EdgeConv_Block(128, k)([feature_input,adjacency_input])
    channel_size = 64
    edge2 = EdgeConv_Block(channel_size, 20)([edge1,adjacency_input])  
    fts = edge2
    self_att_1, attention1 = SA_Layer(channel_size)([fts,mask])
    self_att_2, attention2 = SA_Layer(channel_size)([self_att_1,mask])
    self_att_3, attention3 = SA_Layer(channel_size)([self_att_2,mask])
    concat = tf.concat([self_att_1,self_att_2,self_att_3,fts],axis=-1)

    output = Conv1D_bn(256,activation="relu")(concat)
    
    output = tf.reduce_mean(output,axis=1, keepdims=True)
    output = tf.keras.layers.Flatten()(output)

    output = tf.keras.layers.Dense(128,activation=None)(output)
    output = tf.keras.layers.BatchNormalization()(output)
    output = tf.keras.layers.Activation("relu")(output)

    output = tf.keras.layers.Dropout(0.5)(output)
    if ds.task == 'regression':
        output = tf.keras.layers.Dense(2,activation="linear")(output)
    else:
        output = tf.keras.layers.Dense(2,activation="softmax")(output)

    model = tf.keras.Model([feature_input,adjacency_input],output,name="PCT_"+ds.name)
    
    return model



def get_SPCT(ds,shapes=None):
    """ Returns a simple point cloud transformer model """
    feature_input = tf.keras.layers.Input(shape=shapes['features'], name="features")
    adjacency_input = tf.keras.layers.Input(shape=shapes['adj_matrix'], name="adj_matrix")

    inputs = feature_input

    boolean_mask = inputs[:,:,0]==0
    mask = tf.cast(boolean_mask,float)

    x = Conv1D_bn(128,activation="relu")(inputs)
    x = Conv1D_bn(64,activation="relu")(x)

    self_att_1, attention1 = SA_Layer(64)([x,mask])
    self_att_2, attention2 = SA_Layer(64)([self_att_1,mask])
    concat = tf.concat([self_att_1, self_att_2],axis=-1)

    output = Conv1D_bn(128,activation="relu",bn=False)(concat)

    output = tf.reduce_mean(output,axis=1, keepdims=True)
    output = tf.keras.layers.Flatten()(output)

    output = tf.keras.layers.Dense(64,activation=None)(output)
    output = tf.keras.layers.BatchNormalization()(output)
    output = tf.keras.layers.Activation("relu")(output)

    output = tf.keras.layers.Dropout(0.5)(output)
    if ds.task == 'regression':
        output = tf.keras.layers.Dense(2,activation="linear")(output)
    else:
        output = tf.keras.layers.Dense(2,activation="softmax")(output)

    model = tf.keras.Model(inputs,output,name="SPCT_"+ds.name)

    return model    



###### Network ######

def lr_schedule(epoch):
    lr = 1e-3
    for i in range(10):
        if epoch > i*20:
            lr = lr/2
    return max(lr,1e-6)



class Network(NetworkABC):


    build_graph = True

    #Model selection
    #model_name = "_SPCT_"
    model_name = "_PCT_"

    compatible_datasets = [ TopTagging,Spinodal,EOSL, Belle, Airshower]


    callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath='./',
                 monitor='val_loss',
                 verbose=1,
                 save_best_only=True),
     tf.keras.callbacks.LearningRateScheduler(lr_schedule),
     tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                      min_delta =0.0001,
                                      patience=15,
                                      restore_best_weights = True),
    ] 

    fit_args = {'batch_size': 64,
                'epochs': 200,
                'validation_split': 0.2,
                'shuffle': True,
                'callbacks': callbacks,
                'verbose': 2 #only print one line per epoch
               }


    def metrics(self,task) -> List:
        # list of metrics to be used
        if task == 'regression':
            return [tf.keras.metrics.MeanSquaredError()]
        else:
            return [tf.keras.metrics.SparseCategoricalAccuracy(name='acc')]

    def loss(self,task):
        if task == 'regression':
            return tf.keras.losses.MeanSquaredError()
        else:
            return tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    def compile_args(self,task) -> Dict:
        # dictionary of the arguments to be passed to the method compile()
        return {"metrics": self.metrics(task),"loss": self.loss(task),'optimizer':tf.keras.optimizers.Adam(learning_rate=lr_schedule(0))}



    def preprocessing(self, in_data):
        """
        Method should take as an input the list of datasets to be used as an input for the model
        and after the application of all the preprocessing routin, it should return the modified data
        in the desired shapes
        """ 

        data = in_data
        return data

    def get_shapes(self, in_data):
        """
        Method should take as an input the datasets to be used as an input for the model
        and compute their shapes
        """

        input_shapes = {k:in_data[k].shape[1:] for k in in_data}
        return input_shapes


    def model(self, ds, shapes=None):
        """
        Models are based on the paper `"Point Cloud Transformers applied to Collider Physics" <https://arxiv.org/abs/2102.05073>`.
        """
        print(f"Using Model: {self.model_name}")
        if(self.model_name == "_SPCT_"):       
            return get_SPCT(ds,shapes)
        else:
            return get_PCT(ds,shapes)
        
    
    def init_preprocessing(self, x_train):
        pass

    def model_tag(self, ds_name, model_name):
        ts = time.localtime()
        ts = time.strftime("%Y%m%d_%H%M%S", ts)
        tag = ds_name + model_name + ts
        return tag
 

    @property
    def task(self): 
        return self._task
    @task.setter
    def task(self, value):
        self._task = value

    def evaluation(self, **kwargs):
        model = kwargs.pop("model")
        history = kwargs.pop("history")
        dataset = kwargs.pop("dataset")
        x_test = kwargs.pop("x_test")
        y_test = kwargs.pop("y_test")
        model_name = kwargs.pop("model_name")
        path = kwargs.pop("path")

        plot_path = os.path.join(path,"Plots/")
        if not (os.path.isdir(plot_path)):
            os.makedirs(plot_path)


        # evaluation plots and scores
        y_pred = model.predict(x_test)[:,1]

        if dataset.task == 'regression':
            evaluateTransformer_regression(y_pred,y_test)
            if history != None:
                plot_loss(history, path, dataset.name, True)
        else:

            if history != None:
                train_plots(history, path, dataset.name, model_name, True)

            evaluateTransformer(y_pred, y_test, path, dataset.name, model_name, True)



######### evaluation #############

def _mean_resolution(y_true, y_pred):
    """ Metric to control for standard deviation """
    y_true = tf.cast(y_true, tf.float32)
    mean, var = tf.nn.moments((y_true - y_pred), axes=[0])
    return tf.reduce_mean(tf.sqrt(var))

def plot_loss(history, path, ds, save=False):
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title(ds + " model loss [training]")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "val"], loc="upper left")
    if save:
        path = os.path.join(path, "Plots/")
        if not (os.path.isdir(path)):
            os.makedirs(path)
        plt.savefig(f"{path}{ds}_simple_graph_train_loss.png", dpi=96)
    plt.clf()


def evaluateTransformer_regression(y_pred,y_test):
    from sklearn.metrics import mean_squared_error as MSE
    mse_score = MSE(y_test, y_pred)
    mean_res_score = _mean_resolution(y_test,y_pred)
    _str = "Test MSE score for Airshower dataset is: {} and Resolution score is: {} \n".format(mse_score, mean_res_score)
    print(_str)


def calc_FPR_to_TPR(tpr,fpr,TPR_value=0.5): 
    """ Calculation of a fpr value that corresponds to a given tpr value """
    index_05_tpr = np.argmin((tpr-TPR_value)**2)
    return fpr[index_05_tpr]

def evaluateTransformer(pred, labels, path, ds, model_name, save=False):
    """ Evaluate the performance of the transformer: calculate ACC, AUC and epsilon_B(epsilon_S)-Values. Roc Curves will also be plotted. """
    
    plot_path = os.path.join(path, "Plots/")
    score_path = os.path.join(path, "Scores/")
    if save:
        if not (os.path.isdir(plot_path)):
            os.makedirs(plot_path)
        if not (os.path.isdir(score_path)):
            os.makedirs(score_path)

    #ACC
    val = np.around(pred)
    acc = sklearn.metrics.accuracy_score(labels, val)
    _acc = f"ACC for {ds} Dataset is: {np.round(acc,4)} \n"
    print(_acc)


    #AUC
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(labels, pred, pos_label=1,drop_intermediate=False) 
    auc = sklearn.metrics.auc(fpr,tpr)
    _auc = f"AUC for {ds} Dataset is: {np.round(auc,4)} \n"
    print(_auc)


    #calculate Epsilon_B Values
    fpr_05 = calc_FPR_to_TPR(tpr,fpr,0.5)
    fpr_03 = calc_FPR_to_TPR(tpr,fpr,0.3)

    e05 = 1/fpr_05
    e03 = 1/fpr_03

    _e05 = f"1/Epsilon_B(Epsilon_S=0.5) for {ds} Dataset is: {np.round(e05,0)} \n"
    _e03 = f"1/Epsilon_B(Epsilon_S=0.3) for {ds} Dataset is: {np.round(e03,0)} \n"
    print(_e05)
    print(_e03)

    #plot ROC curve 1
    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr, label='Keras (area = {:.3f})'.format(auc))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    if save: 
        plt.savefig(plot_path+'{}{}roc_auc.png'.format(ds, model_name), dpi=96)
    plt.clf()


    #plot ROC curve 2
    plt.figure(2)
    mask = np.argwhere(fpr==0)
    fpr = np.delete(fpr,mask)
    tpr = np.delete(tpr,mask)
    plt.plot(tpr, 1/fpr, label='Keras (area = {:.3f})'.format(auc))
    plt.xlabel('$\epsilon_S$')
    plt.ylabel('$1/\epsilon_B$')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.yscale("log")
    if save:
        plt.savefig(plot_path+'{}{}roc_auc2.png'.format(ds, model_name), dpi=96)
    plt.clf()

    #plot output hist
    index_background = np.where(labels==0)
    index_signal = np.where(labels==1)
    pred_background = np.delete(pred,index_signal)
    pred_signal = np.delete(pred,index_background)

    plt.figure(3)
    plt.hist(pred_background,histtype="step",label="Background",bins=1000,density=True)
    plt.hist(pred_signal,histtype="step",label="Signal",bins=1000,density=True)
    plt.legend(loc="best")
    plt.yscale("log")
    plt.xlim(0,1)
    plt.title("Model output")
    plt.xlabel("output value")
    plt.ylabel("# events")

    if save:
        plt.savefig(plot_path+'{}{}output_hist.png'.format(ds, model_name), dpi=96)
    plt.clf()

    with open(score_path +'scores{}{}.txt'.format(model_name, ds), 'a') as file:
        file.truncate(0)
        file.write(_acc)
        file.write(_auc)
        file.write(_e05)
        file.write(_e03)
