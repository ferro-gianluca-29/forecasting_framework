from datetime import datetime
import pandas as pd
import numpy as np
import datetime as datetime
import pickle
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates  
import xgboost as xgb
from xgboost import plot_importance, plot_tree
from Predictors.Predictor import Predictor


class XGB_Predictor(Predictor):
    """
    A class used to predict time series data using XGBoost, a gradient boosting framework.
    """

    def __init__(self, run_mode, target_column=None, 
                 verbose=False,  seasonal_model=False, set_fourier=False):
        """
        Constructs all the necessary attributes for the XGB_Predictor object.

        :param run_mode: The mode in which the predictor runs
        :param target_column: The target column of the DataFrame to predict
        :param verbose: If True, prints detailed outputs during the execution of methods
        :param seasonal_model: Boolean, if true include seasonal adjustments like Fourier features
        :param set_fourier: Boolean, if true use Fourier transformation on the data
        """

        super().__init__(verbose=verbose)  

        self.run_mode = run_mode
        self.verbose = verbose
        self.target_column = target_column
        self.seasonal_model = seasonal_model
        self.set_fourier = set_fourier



    def create_time_features(self, df, lags = [1, 2, 3, 24], rolling_window = 24):
        """
        Creates time-based features for a DataFrame, optionally including Fourier features and rolling window statistics.

        :param df: DataFrame to modify with time-based features
        :param lags: List of integers representing lag periods to generate features for
        :param rolling_window: Window size for generating rolling mean and standard deviation
        :return: Modified DataFrame with new features, optionally including target column labels
        """

        label = self.target_column

        df['date'] = df.index
        df['hour'] = df['date'].dt.hour
        df['dayofweek'] = df['date'].dt.dayofweek
        df['quarter'] = df['date'].dt.quarter
        df['month'] = df['date'].dt.month
        df['year'] = df['date'].dt.year
        df['dayofyear'] = df['date'].dt.dayofyear
        df['dayofmonth'] = df['date'].dt.day
        df['weekofyear'] = df['date'].dt.isocalendar().week  # Changed liner

        if self.seasonal_model:

            if self.set_fourier:

                # Fourier features for daily, weekly, and yearly seasonality
                for period in [24, 7, 365]:
                    df[f'sin_{period}'] = np.sin(df.index.dayofyear / period * 2 * np.pi)
                    df[f'cos_{period}'] = np.cos(df.index.dayofyear / period * 2 * np.pi)

            else:
                # Lagged features
                for lag in lags:
                    df[f'lag_{lag}'] = df[label].shift(lag)

                # Rolling window features
                df[f'rolling_mean_{rolling_window}'] = df[label].shift().rolling(window=rolling_window).mean()
                df[f'rolling_std_{rolling_window}'] = df[label].shift().rolling(window=rolling_window).std()

            df = df.dropna()  # Drop rows with NaN values resulting from lag/rolling operations
            X = df.drop(['date', label], axis=1, errors='ignore')
        else:
            X = df[['hour','dayofweek','quarter','month','year',
                'dayofyear','dayofmonth','weekofyear']]
        if label:
            y = df[label]
            return X, y
        return X
    
    def train_model(self, X_train, y_train, X_valid, y_valid):

        """
        Trains an XGBoost model using the training and validation datasets.

        :param X_train: Input data for training
        :param y_train: Target variable for training
        :param X_valid: Input data for validation
        :param y_valid: Target variable for validation
        :return: A tuple containing the trained XGBoost model and validation metrics
        """

        try:
            # Define the XGBoost Regressor with improved parameters
            reg = xgb.XGBRegressor(
                n_estimators=100000,  # Number of boosting rounds (you can tune this)
                learning_rate=0.05,   # Learning rate (you can tune this)
                max_depth=5,          # Maximum depth of the trees (you can tune this)
                min_child_weight=1,   # Minimum sum of instance weight needed in a child
                gamma=0,              # Minimum loss reduction required to make a further partition
                subsample=0.8,        # Fraction of samples used for training
                colsample_bytree=0.8, # Fraction of features used for training
                reg_alpha=0,          # L1 regularization term on weights
                reg_lambda=1,         # L2 regularization term on weights
                objective='reg:squarederror',  # Objective function for regression
                random_state=42,       # Seed for reproducibility
                eval_metric=['rmse', 'mae'],
                early_stopping_rounds=100
                                 )
            # Train the model with early stopping and verbose mode
            XGB_model = reg.fit(
                X_train,
                y_train,
                eval_set=[(X_train, y_train), (X_valid, y_valid)],
                #eval_metric=['rmse', 'mae'],
                #early_stopping_rounds=100,
                verbose=False  # Set to True to see training progress
            )
            
            valid_metrics = XGB_model.evals_result()
            return XGB_model, valid_metrics
        
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
        predictions = predictions.reshape(-1, 1)
        predictions = scaler.inverse_transform(predictions) 
        predictions = predictions.flatten() 
        # Unscale test data
        y_test = pd.DataFrame(y_test)
        y_test = scaler.inverse_transform(y_test)
        y_test = pd.Series(y_test.flatten())

        return predictions, y_test


    def plot_predictions(self, predictions, test, time_values):

        """
        Plots predictions made by an XGBoost model against the test data.

        :param predictions: Predictions made by the XGBoost model
        :param test: The actual test data
        :param time_values: Time values corresponding to the test data
        """

        title = f"Predictions made by XGB model"
        plt.figure(figsize=(16,4))
        plt.plot(time_values, test, color='blue',label='Actual values')
        plt.plot(time_values, predictions, alpha=0.7, color='orange',label='Predicted values')
        plt.title(title)
        plt.xlabel('Date and Time')
        plt.ylabel('Normalized scale')
        plt.xticks(rotation=45)
        plt.legend()
        plt.show()
