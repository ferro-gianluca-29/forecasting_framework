import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import pickle

class NAIVE_Predictor():

    def __init__(self,  run_mode, target_column, 
                 verbose=False):
        """
        Documentation
        """

        self.run_mode = run_mode
        self.verbose = verbose
        self.target_column = target_column

    def prepare_data(self, train = None, valid = None, test = None):
        self.train = train
        self.valid = valid
        self.test = test
        self.steps_ahead = self.test.shape[0]


    def forecast(self, forecast_type): 
        """
        Performs a naive forecast using the last observed value from the training set.

        :param train: The training set.
        :return: A pandas Series of naive forecasts.
        """
        try:

            
            # Create a list of predictions
            predictions = list()
            
            if forecast_type == "cl-multi":
                # Calculate the mean of the target column
                mean_value = self.train[self.target_column].mean()
                # Create a list of predictions based on the mean
                predictions = [mean_value] * self.steps_ahead
                predictions = pd.Series(predictions, index=self.test.index[:self.steps_ahead])
            else:
                last_observation = self.train.iloc[-1][self.target_column]
                predictions.append(last_observation)
                for t in range(1, self.steps_ahead):
                    last_observation = self.test.iloc[t-1, self.test.columns.get_loc(self.target_column)]
                    predictions.append(last_observation)
                predictions = pd.Series(predictions, index=self.test.index[:self.steps_ahead])    
            return predictions
        
        except Exception as e:
            print(f"An error occurred during the naive model creation: {e}")
            return None
        
    def naive_seasonal_forecast(self, period=24):

        """
        Performs a seasonal naive forecast using the last observed seasonal cycle.

        :param train: The training set.
        :param target_test: The test set.
        :param period: The seasonal period to consider for the forecast.
        :return: A pandas Series of naive seasonal forecasts.
        """

        # Naive seasonal forecast: Use the last observed value from the same season as the prediction
        # period to make the forecast.
        
        train = self.train
        target_test = self.test[self.target_column]

        # Create a list of predictions
        predictions = list()
        
        # For each step to predict
        for t in range(0, self.steps_ahead):
            # Get the last observed value from the same season as the prediction period
            last_observation = train.iloc[-period + t][self.target_column]
            # Append the last observed value to the predictions list
            predictions.append(last_observation)
            
        predictions = pd.Series(predictions, index=target_test.index[:self.steps_ahead])    
        # Return the predictions as a pd.Series with the same indexes as the test set
        return predictions
    

    def naive_mean_forecast(self):
        """
        Performs a naive forecast using the mean value of the training set.

        :param train: The training set.
        :return: A pandas Series of naive forecasts using the mean.
        """
        try:

            train = self.train
            # Calculate the mean of the target column
            mean_value = train[self.target_column].mean()
            
            # Create a list of predictions based on the mean
            predictions = [mean_value] * self.steps_ahead
            predictions = pd.Series(predictions, index=self.test.index[:self.steps_ahead])
            
            return predictions
        
        except Exception as e:
            print(f"An error occurred during the naive mean model creation: {e}")
            return None
    
    def unscale_predictions(self, predictions, folder_path):
        # Load scaler for unscaling data
        with open(f"{folder_path}/scaler.pkl", "rb") as file:
            scaler = pickle.load(file)
        
        # Unscale predictions
        predictions = np.array(predictions)
        predictions = predictions.reshape(-1, 1)
        predictions = scaler.inverse_transform(predictions) 
        predictions = predictions.flatten()
        predictions = pd.Series(predictions) 

    def plot_predictions(self, naive_predictions):
        """
        Plots naive predictions against the test data.

        :param naive_predictions: The naive predictions to plot.
        """
        
        self.steps_ahead = self.test.shape[0]
        test = self.test[:self.steps_ahead][self.target_column]
        plt.plot(test.index, test, 'b-', label='Test Set')
        plt.plot(test.index, naive_predictions, 'r--', label='Naive')
        plt.title(f'Naive prediction for feature: {self.target_column}')
        plt.xlabel('Time series index')
        plt.legend(loc='best')
        plt.tight_layout()
        plt.show()