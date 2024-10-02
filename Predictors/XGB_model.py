from datetime import datetime
import pandas as pd
import numpy as np
import datetime as datetime
import pickle
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates  


import skforecast
import sklearn
from xgboost import XGBRegressor
from sklearn.feature_selection import RFECV
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from skforecast.ForecasterAutoregDirect import ForecasterAutoregDirect

from skforecast.model_selection import backtesting_forecaster
from skforecast.model_selection import bayesian_search_forecaster
from skforecast.model_selection import backtesting_forecaster
from skforecast.model_selection import select_features
import shap


import xgboost as xgb
from xgboost import plot_importance, plot_tree
from Predictors.Predictor import Predictor


class XGB_Predictor(Predictor):
    """
    A class used to predict time series data using XGBoost, a gradient boosting framework.
    """

    def __init__(self, run_mode, target_column=None, 
                 verbose=False,  seasonal_model=False, input_len = None, output_len= 24, forecast_type= None, set_fourier=False):
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
        self.input_len = input_len
        self.output_len = output_len
        self.forecast_type = forecast_type
        self.set_fourier = set_fourier



    def create_time_features(self, df, data_freq, lags = [1, 2, 3, 24], rolling_window = 24):
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

            
        X = df[['hour','dayofweek','quarter','month','year',
            'dayofyear','dayofmonth','weekofyear']]
        X.reset_index(drop=True, inplace=True)
        X.set_index(df['date'], inplace=True)
        if X.index.duplicated().any():
            X = X[~X.index.duplicated(keep='first')]
            

        # Verify if the dataset changes with unexpected dimensions   (e.g. a PV dataset with daylight only hours is not continous, and pandas 
                                                                                    # resamples including nan values for night hours)
        # When encountering datasets with holes in datetime, the function has to behaviour differently 
        # (this row is an example, maybe it can be improved for more general situations)

        if len(X.asfreq(data_freq))  > 2 * len(X):
            X = X.asfreq(data_freq).dropna()
        else:
            X = X.asfreq(data_freq)

        X = X.interpolate(method='time')

        if label:
            y = df[label]
            y.reset_index(drop=True, inplace=True)
            y.index = df['date']
            if y.index.duplicated().any():
                y = y[~y.index.duplicated(keep='first')]

            if len(y.asfreq(data_freq))  > 2 * len(y):
                y = y.asfreq(data_freq).dropna()
            else:    
                y = y.asfreq(data_freq)

            y = y.interpolate(method='time')
            return X, y
        return X


    def hyperparameter_tuning(self, X_val, y_val):

        reg = XGBRegressor(
            n_estimators=1000,  # Number of boosting rounds (you can tune this)
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
            transformer_y = None,
            tree_method  = 'hist',
            device       = 'cuda',
                            )
        
        # Griglia dei lag
        lags_grid = [
            24,  # Ultime 24 ore
            48,  # Ultime 48 ore
            [1, 2, 3, 24, 25, 26, 168, 169, 170]  # Lag specifici: ultime ore, stesso orario del giorno precedente e della settimana precedente
        ]

        if self.forecast_type == 'ol-one':

            # Create forecaster
            forecaster = ForecasterAutoreg(
                regressor = reg,
                lags      = self.input_len
                #differentiation = 1
            )

        elif self.forecast_type == 'ol-multi':

            forecaster = ForecasterAutoregDirect(
                                        regressor     = reg,
                                        steps         = self.output_len,
                                        lags          = self.input_len,
                                        transformer_y = None,
                                        n_jobs        = 'auto'
                                        #differentiation = 1
                                    )


        def search_space(trial):
            search_space = {
                'n_estimators'     : trial.suggest_int('n_estimators', 1000, 1100, step=100),
                'learning_rate'    : trial.suggest_float('learning_rate', 0.01, 0.3),
                'max_depth'        : trial.suggest_int('max_depth', 3, 10),
                'min_child_weight' : trial.suggest_int('min_child_weight', 1, 10),
                'gamma'            : trial.suggest_float('gamma', 0, 5),
                'subsample'        : trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree' : trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'reg_alpha'        : trial.suggest_loguniform('reg_alpha', 1e-5, 1.0),
                'reg_lambda'       : trial.suggest_loguniform('reg_lambda', 1e-5, 10.0),
                'lags'             : trial.suggest_categorical('lags', lags_grid),
            }
            return search_space

        results_search, frozen_trial = bayesian_search_forecaster(
        forecaster         = forecaster,
        y                  = y_val,
        exog               = X_val,
        search_space       = search_space,
        steps              = self.output_len,
        refit              = False,
        metric             = 'mean_absolute_error',
        initial_train_size = len(self.train),
        fixed_train_size   = False,
        n_trials           = 20,
        random_state       = 123,
        return_best        = True,
        n_jobs             = 'auto',
        verbose            = True,
        show_progress      = True
                                   )
        return forecaster

    
    def train_model(self, X_train, y_train, X_valid, y_valid):
                                
        reg = XGBRegressor(
            n_estimators=1000,  # Number of boosting rounds (you can tune this)
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
            transformer_y = None,

            # use this two lines to enable GPU
            tree_method  = 'hist',
            device       = 'cuda',
                            )
        reg = reg.fit(
                X_train,
                y_train,
                eval_set=[(X_train, y_train), (X_valid, y_valid)],
                early_stopping_rounds=100,
                verbose=False  # Set to True to see training progress
            )
        
        if self.forecast_type == 'ol-one':

            forecaster = ForecasterAutoreg(
                regressor = reg, 
                lags      = self.input_len,
                #differentiation = 1
            )

        elif self.forecast_type == 'ol-multi':

            forecaster = ForecasterAutoregDirect(
                                        regressor     = reg,
                                        steps         = self.output_len,
                                        lags          = self.input_len,
                                        transformer_y = None,
                                        n_jobs        = 'auto'
                                        # weight_func = custom_weights  # uncomment to give zero weight to night values in PV datasets
                                    )
            

            # this line is not necessary if backtesting is done
            #forecaster.fit(y=y_train, exog = X_train)

        return forecaster
    
    def custom_weights(index):
        """
        Return 0 if the time of the index is outside the 7-18 range.
        """
        # Genera un intervallo per ogni giorno nel periodo di interesse che esclude le ore 18-7
        full_range = pd.date_range(start=index.min().floor('D'), end=index.max().ceil('D'), freq='H')
        working_hours = full_range[((full_range.hour >= 7) & (full_range.hour <= 18))]
        
        # Converti in DatetimeIndex per usare il metodo .isin()
        working_hours = pd.DatetimeIndex(working_hours)

        # Calcola i pesi: 1 se nell'intervallo, 0 altrimenti
        weights = np.where(index.isin(working_hours), 1, 0)

        return weights
        
    def test_model(self, model, X_data, y_data):

        _, predictions = backtesting_forecaster(
                        forecaster         = model,
                        y                  = y_data,
                        exog               = X_data,
                        steps              = self.output_len,
                        metric             = 'mean_absolute_error',
                        initial_train_size = len(self.train) + len(self.valid),
                        refit              = False,
                        n_jobs             = 'auto',
                        verbose            = True, # Change to False to see less information
                        show_progress      = True
                    )
        
        return predictions     

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


