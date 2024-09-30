import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pmdarima
from pmdarima import ARIMA
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.deterministic import Fourier
from tools.time_series_analysis import ljung_box_test

import pmdarima
from pmdarima import ARIMA
from pmdarima import auto_arima

from skforecast.ForecasterSarimax import ForecasterSarimax
from skforecast.model_selection_sarimax import backtesting_sarimax
from skforecast.model_selection_sarimax import grid_search_sarimax





from tqdm import tqdm
import pickle
from Predictors.Predictor import Predictor

class ARIMA_Predictor(Predictor):
    """
    A class used to predict time series data using the ARIMA model.
    """

    def __init__(self, run_mode, target_column=None, 
                 verbose=False):
        """
        Initializes an ARIMA_Predictor object with specified settings.

        :param run_mode: The mode in which the predictor runs
        :param target_column: The target column of the DataFrame to predict
        :param verbose: If True, prints detailed outputs during the execution of methods
        """
        super().__init__(verbose=verbose)  

        self.run_mode = run_mode
        self.verbose = verbose
        self.target_column = target_column
        self.ARIMA_order = []
        

    def train_model(self):
        """
        Trains an ARIMA model using the training dataset.

        :return: A tuple containing the trained model, validation metrics, and the index of the last training/validation timestep
        """
        try:
            
            d = 0

            # Selection of the model with best AIC score
            model = auto_arima(
                        y=self.train[self.target_column],
                        start_p=0,
                        start_q=0,
                        max_p=4,
                        max_q=4,
                        seasonal=False,
                        test='adf',
                        d=None,  # Let auto_arima determine the optimal 'd'
                        trace=True,
                        error_action='warn',  # Show warnings for troubleshooting
                        suppress_warnings=False,
                        stepwise=True
                        )
            
            # for debug
            #model = ARIMA(order=(4, 1, 4))
            

            print(f"Best order found: {model.order}")
            self.ARIMA_order = model.order

            forecaster = ForecasterSarimax(
                 regressor=model)
             
            # Training the model with the best parameters found
            print("\nTraining the ARIMA model...")
            forecaster.fit(y=self.train[self.target_column])

            # Running the LJUNG-BOX test for residual correlation
            residuals = model.resid()
            ljung_box_test(residuals)
            print("Model successfully trained.")
            valid_metrics = None
            last_index = self.train.index[-1]
                

            return forecaster, valid_metrics, last_index
 
        except Exception as e:
            print(f"An error occurred during the model training: {e}")
            return None
        
        
    def test_model(self, forecaster, last_index, forecast_type, output_len, ol_refit = False):
        """
        Tests an ARIMA model by performing one-step ahead predictions and optionally refitting the model.

        :param model: The ARIMA model to be tested
        :param last_index: Index of last training/validation timestep
        :param forecast_type: Type of forecasting ('ol-one' for open-loop one-step ahead, 'cl-multi' for closed-loop multi-step)
        :param ol_refit: Boolean indicating whether to refit the model after each forecast
        :return: A pandas Series of the predictions
        """
        try:
            print("\nTesting ARIMA model...\n")
            
            self.forecast_type = forecast_type
            test = self.test
            self.steps_ahead = self.test.shape[0]
            full_data = pd.concat([self.train, self.test])

            if self.forecast_type == 'ol-one':
                steps = 1
            elif self.forecast_type == 'ol-multi':
                steps = output_len

            _, predictions = backtesting_sarimax(
                    forecaster            = forecaster,
                    y                     = full_data[self.target_column],
                    initial_train_size    = len(self.train),
                    steps                 = steps,
                    metric                = 'mean_absolute_error',
                    refit                 = False,
                    n_jobs                = "auto",
                    verbose               = True,
                    show_progress         = True
                )

            predictions.rename(columns={'pred': self.target_column}, inplace=True)     
            print("Model testing successful.")        
            return predictions
            
        except Exception as e:
            print(f"An error occurred during the model test: {e}")
            return None
        
    def plot_predictions(self, predictions):
        """
        Plots the ARIMA model predictions against the test data.

        :param predictions: The predictions made by the ARIMA model
        """
        test = self.test[:self.steps_ahead][self.target_column]
        plt.plot(test.index, test, 'b-', label='Test Set')
        plt.plot(test.index, predictions, 'k--', label='ARIMA')
        plt.title(f'ARIMA prediction for feature: {self.target_column}')
        plt.xlabel('Time series index')
        plt.legend(loc='best')
        plt.tight_layout()
        plt.show()

    
