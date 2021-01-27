import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

def seq_model(ds, n_hlayers, n_nodes, shape, save_model_png = False):
    
    """
    Builds sequential model.
    ds: string. Name of the dataset that will be used as input for the model;
    n_hlayers: integer. Number of desired hidden layer, must be > 0;
    n_nodes: integer. Number of desired nodes per layer, must be > 0;
    shape: tuple. Shape of the dataset;
    save_model_png = bool. Save .png of the model in the execution dir.
    """
    
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape = shape))
    if len(shape) > 1:
        model.add(tf.keras.layers.Flatten())
    tf.keras.layers.BatchNormalization()
    for l in range(n_hlayers):
        model.add(tf.keras.layers.Dense(n_nodes, activation = 'relu'))
    model.add(tf.keras.layers.Dense(1, activation = 'sigmoid'))
    
    if save_model_png:
        tf.keras.utils.plot_model(model, to_file='./{}_model.png'.format(ds))
    return model



def two_input_model(ds, n_hlayers, n_nodes, shape1, shape2, save_model_png = False):
    
    """
    Builds model using two input datasets.
    ds: string. Name of the dataset that will be used as input for the model;
    n_hlayers: integer. Number of desired hidden layer;
    n_nodes: integer. Number of desired nodes per layer;
    shape1: tuple. Shape of the 2D dataset;
    shape2: tuple. Shape of the 1D dataset;
    save_model_png = bool. Save .png of the model in the execution dir.
    """
    
    input1 = tf.keras.Input(shape = shape1)
    input2 = tf.keras.Input(shape = shape2)
    flatten = tf.keras.layers.Flatten()(input1)
    norm1 = tf.keras.layers.BatchNormalization()(flatten)
    norm2 =tf.keras.layers.BatchNormalization()(input2)
    dense1 = {}
    dense2 = {}
    dense1[0] = tf.keras.layers.Dense(n_nodes, activation = 'relu')(norm1)
    dense2[0] = tf.keras.layers.Dense(n_nodes, activation = 'relu')(norm2)
    
    hlayers_before_merge = int(n_hlayers*0.2)
    
    for n in range(1, hlayers_before_merge):
        dense1[n] = tf.keras.layers.Dense(n_nodes, activation = 'relu')(dense1[n-1])
        dense2[n] = tf.keras.layers.Dense(n_nodes, activation = 'relu')(dense2[n-1])
    merged = tf.keras.layers.Concatenate(axis=1)([dense1[hlayers_before_merge-1], dense2[hlayers_before_merge-1]])
    dense = {}
    dense[0] = tf.keras.layers.Dense(n_nodes, activation = 'relu')(merged)
    
    for n in range(1, n_hlayers-hlayers_before_merge):
        dense[n] = tf.keras.layers.Dense(n_nodes, activation = 'relu')(dense[n-1])
        
    
    output  = tf.keras.layers.Dense(1, activation = 'sigmoid')(dense[len(dense)-1])
    
    model = tf.keras.models.Model(inputs = [input1, input2], outputs = output)
    
    if save_model_png:
        tf.keras.utils.plot_model(model, to_file='./{}_model.png'.format(ds))
    return model


def generate_model(ds, n_hlayers, n_nodes, shape1, shape2 = None, save_model_png = False):
    
    """
    Executes the appropriate model generator method for the selected dataset and returns the model.
    ds: string. Name of the dataset that will be used as input for the model;
    n_hlayers: integer. Number of desired hidden layer;
    n_nodes: integer. Number of desired nodes per layer;
    shape1: tuple. Shape of the 2D dataset;
    shape2: tuple. Shape of the 1D dataset;
    save_model_png = bool. Save .png of the model in the execution dir.
    """
    
    if ds == 'airshower' or ds == 'belle':
        model = two_input_model(ds, n_hlayers, n_nodes, shape1, shape2, save_model_png)
        return model
    model = seq_model(ds, n_hlayers, n_nodes, shape1, save_model_png)
    return model
    

    
def train_plots(history, ds, save = False):
    
    """
    Produces plots (loss and accuracy) of training history for training and validation data.
    history: model history;
    ds: string. Input model name.
    
    """
    
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(ds + ' model loss [training]')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    if save:
        plt.savefig('{}_train_loss.png'.format(ds), dpi=96)
    plt.show()
    
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title(ds + ' model accuracy [training]')
    plt.ylabel('binary_accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    if save:
        plt.savefig('{}_train_accuracy.png'.format(ds), dpi=96)
    plt.show()


def roc_auc(y_pred, y_test, ds, save = False):
    
    """
    Plots ROC and reports AUC score.
    X_test: test data
    y_test: labels of X_test
    ds: name of the dataset
    """

    fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test, y_pred)
    auc_keras = auc(fpr_keras, tpr_keras)
    
    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    if save:
        plt.savefig('{}_roc_auc.png'.format(ds), dpi=96)
    plt.show()
    
def test_accuracy(y_pred, y_test, ds):
    
    """
    Returns accuracy score.
        X_test: test data
    y_test: labels of X_test
    ds: name of the dataset
    """
    
    from sklearn.metrics import accuracy_score
    rounded_pred = np.around(y_pred)
    print("Test accuracy score for {} dataset is: {}".format(ds, accuracy_score(y_test, rounded_pred )))
    
def test_f1_score(y_pred, y_test, ds):
    
    """
    Returns f1 score.
    X_test: test data
    y_test: labels of X_test
    ds: name of the dataset
    """
    
    from sklearn.metrics import f1_score
    rounded_pred = np.around(y_pred)
    print("Test F1 score for {} dataset is: {}".format(ds, f1_score(y_test, rounded_pred )))

