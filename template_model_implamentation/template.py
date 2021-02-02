### This is a template file meant to be a guideline to smooth out the implementation of our models in the same framework. 
### This is a template file meant to be a guideline to smooth out the implementation of our models in the same framework. 
import tensorflow as tf
import numpy as np


class ModelTemplate:

    def __init__(self):
        pass

    callbacks = []                                              ##list of callbacks to be used in model.
    metrics   = []                                              ##list of metrics to be used
    compile_args = {'metrics': metrics}                      ##dictionary of the arguments to be passed to the method compile()
    fit_args = {'callbacks': callbacks}                      ##dictionary of the arguments to be passed to the method fit()

    compatible_datasets = []         ## we would also ask you to add a list of the datasets that would be compatible with your implementation 

    def preprocessing(self, in_data):
        """
        Method should take as an input the list of datasets to be used as an iput for the model
        and after the application of all the preprocessing routin, it should return the modified data
        in the desired shapes
        """
        
        #   write your preprocessing routin here
        return out_data


    def model(self, ds, shapes = None):
        '''
         model should take shapes of the input datasets (not counting the number of events)
         and return the desired model
        '''
        #   write your model here
        return model

