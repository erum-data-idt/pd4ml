# This is a template file meant to be a guideline to smooth
# out the implementation of our models in the same framework.
from abc import ABCMeta, abstractmethod
from typing import List, Dict
from utils import train_plots, roc_auc, test_accuracy, test_f1_score


class NetworkABC(metaclass=ABCMeta):
    def __init__(self):
        self._task = None
    @property    
    def callbacks(self) -> List:
        # list of callbacks to be used in model.
        return []

    def metrics(self) -> List:
        # list of metrics to be used
        return []

    def compile_args(self) -> Dict:
        # dictionary of the arguments to be passed to the method compile()
        return {"metrics": self.metrics()}
    @property
    def fit_args(self) -> Dict:
        # r dictionary of the arguments to be passed to the method fit()
        return {"callbacks": self.callbacks}

    @property
    @abstractmethod
    def compatible_datasets(self) -> List:
        # we would also ask you to add a list of the datasets that
        # would be compatible with your implementation
        pass

    def preprocessing(self, in_data):
        """
        Method should take as an input the list of datasets to be used as an input for the model
        and after the application of all the preprocessing routin, it should return the modified data
        in the desired shapes
        """

        # write your preprocessing routin here
        return in_data

    @abstractmethod
    def get_shapes(self, in_data):
        """
        Method should take as an input the datasets to be used as an input for the model
        and compute their shapes
        """

        # write your shape calculation here
        pass

    @abstractmethod
    def model(self, ds, shapes=None):
        """
        model should take shapes of the input datasets (not counting the number of events)
        and return the desired model
        """
        # write your model here
        pass
    
    def init_preprocessing(self, x_train):
        pass
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
        train_plots(history, dataset.name, True)

        # evaluation plots and scores
        y_pred = model.predict(x_test).ravel()
        roc_auc(y_pred, y_test, dataset.name, True)
        test_accuracy(y_pred, y_test, dataset.name, self.model_name)
        test_f1_score(y_pred, y_test, dataset.name, self.model_name)
