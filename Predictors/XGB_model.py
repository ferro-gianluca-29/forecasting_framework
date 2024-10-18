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
from xgboost import XGBRegressor,plot_importance
from sklearn.feature_selection import RFECV
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from skforecast.ForecasterAutoregDirect import ForecasterAutoregDirect

from skforecast.model_selection import backtesting_forecaster
from skforecast.model_selection import bayesian_search_forecaster
from skforecast.model_selection import backtesting_forecaster
from skforecast.model_selection import select_features
import shap

from sklearn.metrics import root_mean_squared_error
from sklearn.feature_selection import RFECV

import xgboost as xgb
from xgboost import plot_importance, plot_tree
from Predictors.Predictor import Predictor

import ephem
import pytz


class XGB_Predictor(Predictor):
    """
    A class used to predict time series data using XGBoost, a gradient boosting framework.
    """

    def __init__(self, run_mode, target_column=None, 
                 verbose=False,  seasonal_model=False, input_len = None, output_len= 24, forecast_type= None, period=24):
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
        self.period = period

        self.selected_exog = []


    def hyperparameter_tuning(self):

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
        
        
        

        # EXOGENOUS VARIABLES
        
    
        for df in (self.train, self.valid, self.test):
            # Existing time features
            df['month_sin'] = np.sin(2 * np.pi * df.index.month / 12)
            df['month_cos'] = np.cos(2 * np.pi * df.index.month / 12)
            df['week_of_year_sin'] = np.sin(2 * np.pi * df.index.isocalendar().week / 52)
            df['week_of_year_cos'] = np.cos(2 * np.pi * df.index.isocalendar().week / 52)
            df['week_day_sin'] = np.sin(2 * np.pi * df.index.weekday / 7)
            df['week_day_cos'] = np.cos(2 * np.pi * df.index.weekday / 7)
            df['hour_day_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
            df['hour_day_cos'] = np.cos(2 * np.pi * df.index.hour / 24)

            # Aggiunta delle caratteristiche per il giorno del mese
            df['day_sin'] = np.sin(2 * np.pi * df.index.day / df.index.days_in_month)
            df['day_cos'] = np.cos(2 * np.pi * df.index.day / df.index.days_in_month)

            # Rolling means
            df['roll_mean_1_day'] = df[self.target_column].rolling(window=self.period, min_periods=1).mean()
            df['roll_mean_7_day'] = df[self.target_column].rolling(window=self.period*7, min_periods=1).mean()

        # Aggiornamento dell'elenco delle caratteristiche esogene
        exog_features = [
            'month_sin', 
            'month_cos',
            'week_of_year_sin',
            'week_of_year_cos',
            'week_day_sin',
            'week_day_cos',
            'hour_day_sin',
            'hour_day_cos',
            'day_sin',  # Aggiunta del seno del giorno
            'day_cos',  # Aggiunta del coseno del giorno
            'roll_mean_1_day',
            'roll_mean_7_day',
        ]

        self.selected_exog = exog_features
        
        # HYPERPARAMETER TUNING


        # Create forecaster for hyperparameter tuning

        forecaster = ForecasterAutoreg(
            regressor = reg,
            lags      = self.input_len
            #differentiation = 1
        )

      
        def search_space(trial):
            search_space = {
                'n_estimators'    : trial.suggest_int('n_estimators', 500, 2000, step=300),
                'max_depth'       : trial.suggest_int('max_depth', 3, 15),
                'learning_rate'   : trial.suggest_float('learning_rate', 0.001, 0.2, log=True),
                'subsample'       : trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'gamma'           : trial.suggest_float('gamma', 0, 5),
                'reg_alpha'       : trial.suggest_float('reg_alpha', 0, 1),
                'reg_lambda'      : trial.suggest_float('reg_lambda', 0, 1),
                'lags'            : trial.suggest_int('lags', 1, 24)
            }
            return search_space


        train_val_data = pd.concat([self.train, self.valid])

        results_search, frozen_trial = bayesian_search_forecaster(
        forecaster         = forecaster,
        y                  = train_val_data[self.target_column],
        exog               = train_val_data[exog_features],
        search_space       = search_space,
        steps              = self.output_len,
        refit              = False,
        metric             = 'mean_squared_error',
        initial_train_size = len(self.train),
        fixed_train_size   = False,
        n_trials           = 40,
        random_state       = 123,
        return_best        = True,
        n_jobs             = 'auto',
        verbose            = True,
        show_progress      = True
                                   )
        
        best_params = results_search['params'].iat[0]
        best_lags = results_search['lags'].iat[0]
        

        """# FEATURE SELECTION

        # Create forecaster for feature selection

        forecaster = ForecasterAutoreg(
            regressor = XGBRegressor(**best_params),
            lags      = best_lags
            #differentiation = 1
        )

        # Recursive feature elimination with cross-validation

        selector = RFECV(
            estimator              = reg,
            step                   = 1,
            cv                     = 3,
            min_features_to_select = 10,
            n_jobs                 = -1
        )

        selected_lags, self.selected_exog = select_features(
                forecaster      = forecaster,
                selector        = selector,
                y               = train_val_data[self.target_column],
                exog            = train_val_data[exog_features],
                select_only     = None,
                force_inclusion = None,
                subsample       = 0.5,
                random_state    = 123,
                verbose         = True,
            )
        
        forecaster = ForecasterAutoreg(
                    regressor = XGBRegressor(**best_params),
                    lags      = selected_lags
                )"""
        
        return forecaster
    
    

    
    def train_model(self):


        for df in (self.train, self.valid, self.test):
            # Existing time features
            df['month_sin'] = np.sin(2 * np.pi * df.index.month / 12)
            df['month_cos'] = np.cos(2 * np.pi * df.index.month / 12)
            df['week_of_year_sin'] = np.sin(2 * np.pi * df.index.isocalendar().week / 52)
            df['week_of_year_cos'] = np.cos(2 * np.pi * df.index.isocalendar().week / 52)
            df['week_day_sin'] = np.sin(2 * np.pi * df.index.weekday / 7)
            df['week_day_cos'] = np.cos(2 * np.pi * df.index.weekday / 7)
            df['hour_day_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
            df['hour_day_cos'] = np.cos(2 * np.pi * df.index.hour / 24)

            # Aggiunta delle caratteristiche per il giorno del mese
            df['day_sin'] = np.sin(2 * np.pi * df.index.day / df.index.days_in_month)
            df['day_cos'] = np.cos(2 * np.pi * df.index.day / df.index.days_in_month)

            # Rolling means
            df['roll_mean_1_day'] = df[self.target_column].rolling(window=self.period, min_periods=1).mean()
            df['roll_mean_7_day'] = df[self.target_column].rolling(window=self.period*7, min_periods=1).mean()

        # Aggiornamento dell'elenco delle caratteristiche esogene
        exog_features = [
            'month_sin', 
            'month_cos',
            'week_of_year_sin',
            'week_of_year_cos',
            'week_day_sin',
            'week_day_cos',
            'hour_day_sin',
            'hour_day_cos',
            'day_sin',  # Aggiunta del seno del giorno
            'day_cos',  # Aggiunta del coseno del giorno
            'roll_mean_1_day',
            'roll_mean_7_day',
        ]
        
        self.selected_exog = exog_features
                                
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
            random_state=42,      # Seed for reproducibility
            eval_metric=['rmse', 'mae'],
            transformer_y = None,



            # use this two lines to enable GPU
            #tree_method  = 'hist',
            #device       = 'cuda',
                            )

        
        def custom_weights(index):
            """
            Return 0 if the time of the index is outside the 7-18 range.
            """
            # Genera un intervallo per ogni giorno nel periodo di interesse che esclude le ore 18-7
            full_range = pd.date_range(start=index.min().floor('D'), end=index.max().ceil('D'), freq='H')
            working_hours = full_range[((full_range.hour >= 6) & (full_range.hour <= 20))]
            
            # Converti in DatetimeIndex per usare il metodo .isin()
            working_hours = pd.DatetimeIndex(working_hours)

            # Calcola i pesi: 1 se nell'intervallo, 0 altrimenti
            weights = np.where(index.isin(working_hours), 1, 0.4)

            return weights

        forecaster = ForecasterAutoreg(
            regressor = reg, 
            lags      = self.input_len,
          #  weight_func      = custom_weights
            #differentiation = 1
        )



            
        # this line is not necessary if backtesting is done
        forecaster.fit(y = self.train[self.target_column]
                       #exog = self.train[exog_features] 
                       )

        print(forecaster.get_feature_importances())

        return forecaster
    
    
        
    def test_model(self, model):

        
        
        full_data = pd.concat([self.train, self.valid, self.test])

        metric, predictions = backtesting_forecaster(
                        forecaster         = model,
                        y                  = full_data[self.target_column],
                        exog               = full_data[self.selected_exog],
                        steps              = self.output_len,
                        metric             = 'mean_absolute_percentage_error',
                        initial_train_size = len(self.train) + len(self.valid),
                        refit              = False,
                        n_jobs             = 'auto',
                        verbose            = True, # Change to False to see less information
                        show_progress      = True
                    )
        
        


        print(metric)

        rmse = root_mean_squared_error(self.test[self.target_column], predictions['pred'])

        print(f"RMSE Calcolato: {rmse}")

        predictions.rename(columns={'pred': self.target_column}, inplace=True)

        
        return predictions     


    def plot_predictions(self, predictions):
        """
        Plots the XGB model predictions against the test data.

        :param predictions: The predictions made by the LSTM model
        """
        test = self.test[self.target_column]
        plt.plot(test.index, test, 'b-', label='Test Set')
        plt.plot(test.index, predictions, 'k--', label='LSTM')
        plt.title(f'XGB prediction for feature: {self.target_column}')
        plt.xlabel('Time series index')
        plt.legend(loc='best')
        plt.tight_layout()
        plt.show()


