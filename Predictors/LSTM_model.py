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

    
    def data_windowing(self):
        """
        Creates data windows suitable for input into LSTM models, optionally incorporating Fourier features for seasonality.

        :return: Arrays of input and output data windows for training, validation, and testing
        """

        input_len, output_len = self.input_len, self.output_len
        
        set_fourier = self.set_fourier
        seasonal_model = self.seasonal_model
        stride_train = 1
        stride_test = input_len
        train = self.train
        valid = self.valid
        test = self.test
        X_train, y_train, X_valid, y_valid, X_test, y_test = [], [], [], [], [], []
        indices_train, indices_valid, indices_test = [], [], []

        # Definisci le colonne base e le colonne di Fourier
        

        if self.run_mode in ["train", "train_test", "fine_tuning"]:
            # Processa train e valid sets
            for dataset, X, y, indices in [(train, X_train, y_train, indices_train),
                                        (valid, X_valid, y_valid, indices_valid)]:
                
                fourier_columns = [col for col in dataset.columns if col.startswith(('sin', 'cos'))]
                input_columns = [self.target_column] + fourier_columns if set_fourier else [self.target_column]
                first_window = True

                for i in range(0, len(dataset) - input_len - output_len + 1, stride_train):
                    X.append(dataset[input_columns].iloc[i:i + input_len].values)
                    y.append(dataset[self.target_column].iloc[i + input_len:i + input_len + output_len].values)
                    indices.append(i)
                    if first_window == True and seasonal_model == False:
                        print(f"X first window from {dataset['date'].iloc[i]} to {dataset['date'].iloc[i+input_len-1]}")
                        print(f"y first window from {dataset['date'].iloc[i+input_len]} to {dataset['date'].iloc[i+input_len+output_len-1]}")
                        first_window = False

        # Test set sempre processato
        if len(test) < input_len + output_len:
            print("Test data is too short for creating windows")
            return None
        else:
            
            fourier_columns = [col for col in test.columns if col.startswith(('sin', 'cos'))]
            input_columns = [self.target_column] + fourier_columns if set_fourier else [self.target_column]
            first_window = True

            
            for i in range(0, len(test) - input_len - output_len + 1, stride_test):
                X_test.append(test[input_columns].iloc[i:i + input_len].values)
                y_test.append(test[self.target_column].iloc[i + input_len:i + input_len + output_len].values)
                indices_test.append(i)
                if first_window == True and seasonal_model == False:
                    print(f"X_test first window from {test['date'].iloc[i]} to {test['date'].iloc[i+input_len-1]}")
                    print(f"y_test first window from {test['date'].iloc[i+input_len]} to {test['date'].iloc[i+input_len+output_len-1]}")
                    first_window = False

        # Conversione in array e ridimensionamento
        X_train, y_train = np.array(X_train), np.array(y_train)
        X_valid, y_valid = np.array(X_valid), np.array(y_valid)
        X_test, y_test = np.array(X_test), np.array(y_test)

        # Reshape dei dati di input per includere tutte le feature nel modello
        if X_train.size > 0:
            X_train = np.reshape(X_train, (X_train.shape[0], input_len, len(input_columns)))
        if X_valid.size > 0:
            X_valid = np.reshape(X_valid, (X_valid.shape[0], input_len, len(input_columns)))
        X_test = np.reshape(X_test, (X_test.shape[0], input_len, len(input_columns)))

        print("Data windowing complete")
        if self.run_mode == "test":
            return [X_test, y_test]
        else:
            return [X_train, y_train, X_valid, y_valid, X_test, y_test]
        

    def train_model(self, X_train, y_train, X_valid, y_valid):
        """
        Trains an LSTM model using the training and validation datasets.

        :param X_train: Input data for training
        :param y_train: Target variable for training
        :param X_valid: Input data for validation
        :param y_valid: Target variable for validation
        :return: A tuple containing the trained LSTM model and validation metrics
        """
        try:
            
            output_dim = self.output_len
            if self.output_len == 1:
                ret_seq_flag = False
            else:
                ret_seq_flag = True
            

            lstm_model = Sequential()

            lstm_model.add(LSTM(40,activation="tanh",return_sequences=True, 
                                #stateful = True,
                                input_shape=(X_train.shape[1], X_train.shape[2])))
            lstm_model.add(Dropout(0.15))

            lstm_model.add(LSTM(40,activation="tanh",
                                #stateful = True,
                                return_sequences=True,
                                ))
            lstm_model.add(Dropout(0.15))

            lstm_model.add(LSTM(40,activation="tanh", 
                                #stateful = True, 
                                return_sequences=ret_seq_flag,
                                ))
            lstm_model.add(Dropout(0.15))

            if self.output_len != 1: lstm_model.add(Flatten())

            lstm_model.add(Dense(output_dim))

            if self.verbose: lstm_model.summary()
            
            lstm_model.compile(optimizer="adam",
                               loss="MSE",
                               metrics=[MeanAbsoluteError(), MeanAbsolutePercentageError(), RootMeanSquaredError()])
            
            history= lstm_model.fit(X_train, y_train, epochs=100, validation_data=(X_valid, y_valid), batch_size=400)
            my_loss= lstm_model.history.history['loss']
            
            valid_metrics = {}
            valid_metrics['valid_loss'] = history.history['val_loss']
            valid_metrics['valid_mae'] = history.history['val_mean_absolute_error']
            valid_metrics['valid_mape'] = history.history['val_mean_absolute_percentage_error']
                        

            if self.verbose:
                # plot train and validation loss
                plt.plot(my_loss)
                plt.plot(history.history['val_loss'])
                plt.title('model train vs validation loss')
                plt.ylabel('loss')
                plt.xlabel('epoch')
                plt.legend(['train', 'validation'], loc='upper right')
                plt.show()
            
            return lstm_model, valid_metrics
        
        except Exception as e:
            print(f"An error occurred during the model training: {e}")
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
        num_samples, num_timesteps = predictions.shape
        predictions = predictions.reshape(-1, 1)
        predictions = scaler.inverse_transform(predictions)
        predictions = predictions.reshape(num_samples, num_timesteps)

        # Unscale test data
        num_samples, num_timesteps = y_test.shape
        y_test = y_test.reshape(-1, 1)
        y_test = scaler.inverse_transform(y_test)
        y_test = y_test.reshape(num_samples, num_timesteps)    

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
        # Assuming 'date' is the column containing datetime in the 'test_data' DataFrame
        # And each point in the test window corresponds to a record in the DataFrame
        start_date = self.test['date'].iloc[0]
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