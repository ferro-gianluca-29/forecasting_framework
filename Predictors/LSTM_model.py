from datetime import datetime
import pandas as pd
import numpy as np
import datetime as datetime
import pickle
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates  
from keras.layers import Dense,Flatten,Dropout,SimpleRNN,LSTM
from keras.models import Sequential
from keras.metrics import MeanAbsoluteError, MeanAbsolutePercentageError, RootMeanSquaredError

from keras.optimizers import Adam
from keras.losses import MeanSquaredError
from keras.callbacks import EarlyStopping

from skforecast.ForecasterRnn import ForecasterRnn
from skforecast.ForecasterRnn.utils import create_and_compile_model
from skforecast.model_selection_multiseries import backtesting_forecaster_multiseries

import warnings
warnings.filterwarnings("ignore")
from Predictors.Predictor import Predictor


class LSTM_Predictor(Predictor):
    """
    A class used to predict time series data using Long Short-Term Memory (LSTM) networks.
    """

    def __init__(self, run_mode, target_column=None, 
                 verbose=False, input_len=None, output_len=None, seasonal_model=False, set_fourier=False):
        """
        Constructs all the necessary attributes for the LSTM_Predictor object.

        :param run_mode: The mode in which the predictor runs
        :param target_column: The target column of the DataFrame to predict
        :param verbose: If True, prints detailed outputs during the execution of methods
        :param input_len: Number of past observations to consider for each input sequence
        :param output_len: Number of future observations to predict
        :param seasonal_model: Boolean, if true include seasonal adjustments like Fourier features
        :param set_fourier: Boolean, if true use Fourier transformation on the data
        """
        super().__init__(verbose=verbose)  

        self.run_mode = run_mode
        self.verbose = verbose
        self.target_column = target_column
        self.input_len = input_len
        self.output_len = output_len
        self.seasonal_model = seasonal_model
        self.set_fourier = set_fourier
        

    def train_model(self):
        """
        Trains an LSTM model using the training and validation datasets.

        :param X_train: Input data for training
        :param y_train: Target variable for training
        :param X_valid: Input data for validation
        :param y_valid: Target variable for validation
        :return: A tuple containing the trained LSTM model and validation metrics
        """
        try:

            
            model = create_and_compile_model(
                        series = self.train[[self.target_column]], # Series used as predictors
                        levels = self.target_column,                         # Target column to predict
                        lags = self.input_len,
                        steps = self.output_len,
                        recurrent_layer = "LSTM",
                        activation = "tanh",
                        recurrent_units = [40,40,40],
                        optimizer = Adam(learning_rate=0.01), 
                        loss = MeanSquaredError()
                                            )
            
            model.summary()


            forecaster = ForecasterRnn(
                                regressor = model,
                                levels = self.target_column,
                                transformer_series = None,
                                fit_kwargs={
                                    "epochs": 2,  # Number of epochs to train the model.
                                    "batch_size": 400,  # Batch size to train the model.
                                    "callbacks": [
                                        EarlyStopping(monitor="val_loss", patience=5)
                                    ],  # Callback to stop training when it is no longer learning.
                                    "series_val": self.valid[[self.target_column]],  # Validation data for model training.
                                },
                                    )    
            
            forecaster.fit(self.train[[self.target_column]])

            return forecaster
        
        except Exception as e:
            print(f"An error occurred during the model training: {e}")
            return None
        
    def test_model(self,forecaster):
        try:

            full_data = pd.concat([self.train, self.valid, self.test])
            _, predictions = backtesting_forecaster_multiseries(
                                    forecaster = forecaster,
                                    steps = self.output_len,
                                    series=full_data[[self.target_column]],
                                    levels=forecaster.levels,
                                    initial_train_size=len(self.train) + len(self.valid), # Training + Validation Data
                                    metric="mean_absolute_error",
                                    verbose=False, # Set to True for detailed information
                                    refit=False,
                                )
            return predictions
        
        except Exception as e:
            print(f"An error occurred during the model test: {e}")
            return None
        

    def unscale_data(self, predictions, y_test, folder_path):
        
        """
        Unscales the predictions and test data using the scaler saved during model training.

        :param predictions: The scaled predictions that need to be unscaled
        :param y_test: The scaled test data that needs to be unscaled
        :param folder_path: Path to the folder containing the scaler object
        """
        # Load scaler for unscaling data
        with open(f"{folder_path}/scaler.pkl", "rb") as file:
            scaler = pickle.load(file)
        
        # Unscale predictions
        predictions = predictions.to_numpy().reshape(-1, 1)
        predictions = scaler.inverse_transform(predictions) 
        predictions = predictions.flatten() 
        # Unscale test data
        y_test = pd.DataFrame(y_test)
        y_test = scaler.inverse_transform(y_test)
        y_test = pd.Series(y_test.flatten())

        return predictions, y_test                                
           

    def plot_predictions(self, predictions, y_test):
        """
        Plots LSTM model predictions against actual test data for each data window in the test set.

        :param predictions: Predictions made by the LSTM model
        :param y_test: Actual test values corresponding to the predictions
        """
        # Select the window for y_test and predictions
        window_num = 0
        test_window = y_test[window_num, :]
        pred_window = predictions[window_num, :]

        # Output predictions begin after input_len test timesteps
        start_date = self.test.index[self.input_len]
        # Create a date range for the x-axis with a frequency of 15 minutes
        date_range = pd.date_range(start=start_date, periods=len(test_window), freq='H')

        # Initialize the plot
        plt.figure(figsize=(12, 6))
        # Plot the actual test data
        plt.plot(date_range, test_window, 'b-', label='Test Set', linewidth=2)
        # Plot the LSTM predictions
        plt.plot(date_range, pred_window, 'r--', label='LSTM Predictions', linewidth=2)
        
        # Set the title and labels
        plt.title(f"LSTM Prediction for Window {window_num}")
        plt.xlabel('Date and Time')
        plt.ylabel('Value')
        # Display the legend
        plt.legend()
        # Add grid
        plt.grid(True)

        # Set the format for the x-axis dates to include day, month, hour, and minutes
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d-%m %H:%M'))
        # Automatically format the x-axis labels to be more readable
        plt.gcf().autofmt_xdate()

        # Ensure the layout fits well
        plt.tight_layout()
        # Show the plot
        plt.show()