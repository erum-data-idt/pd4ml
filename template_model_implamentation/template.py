# This is a template file meant to be a guideline to smooth
# out the implementation of our models in the same framework.
from abc import ABCMeta, abstractmethod
from typing import List, Dict


class NetworkABC(metaclass=ABCMeta):
    @property
    def callbacks(self) -> List:
        # list of callbacks to be used in model.
        return []

    @property
    def metrics(self) -> List:
        # list of metrics to be used
        return []

    @property
    def compile_args(self) -> Dict:
        # dictionary of the arguments to be passed to the method compile()
        return {"metrics": self.metrics}

    @property
    def fit_args(self) -> Dict:
        # dictionary of the arguments to be passed to the method fit()
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
    def model(self, ds, shapes=None):
        """
        model should take shapes of the input datasets (not counting the number of events)
        and return the desired model
        """
        # write your model here
        pass
