### This is a template file meant to be a guideline to smooth out the implementation of our models in the same framework. 
##  William.


import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
    

    
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

