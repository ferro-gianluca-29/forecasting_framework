from abc import ABC, abstractmethod


class Predictor(ABC):
    
    def __init__(self, verbose=False):
        self.verbose = verbose

    def prepare_data(self, train = None, valid = None, test = None):
        self.train = train
        self.valid = valid
        self.test = test

        
    @abstractmethod
    def train_model(self):
        """
        Trains a model using the provided training and validation datasets.
        
        :return: A tuple of the trained model and validation metrics.
        """
        pass
    
    @abstractmethod
    def plot_predictions(self, predictions, test_values):
        """
        Plots predictions against actual values for the test dataset.
        
        :param predictions: Array of predicted values.
        :param test_values: Array of actual values from the test set.
        """
        pass

    