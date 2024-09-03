from abc import ABC, abstractmethod


class Predictor(ABC):
    """
    An abstract base class used to define a blueprint for predictor models.
    """
    
    def __init__(self, verbose=False):
        """
        Initializes the Predictor object with common attributes.

        :param verbose: If True, prints detailed outputs during the execution of methods
        """
        self.verbose = verbose

    def prepare_data(self, train = None, valid = None, test = None):
        """
        Prepares the data sets for training, validation, and testing.

        :param train: Training dataset
        :param valid: Validation dataset (optional)
        :param test: Testing dataset
        """
        self.train = train
        self.valid = valid
        self.test = test

        
    @abstractmethod
    def train_model(self):
        """
        Trains a model using the provided training and validation datasets.

        :return: A tuple containing the trained model and validation metrics.
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

    