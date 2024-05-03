import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, mean_absolute_error, r2_score
from sktime.performance_metrics.forecasting import mean_squared_percentage_error
from classes.model_testing import ModelTest

class PerfMeasure(ModelTest):
    """
    Class that extends ModelTest to provide additional methods for measuring and plotting performance
    metrics of forecasting models.
    """
    
    def get_performance_metrics(self, test, predictions, naive = False):
        """
        Calculates a set of performance metrics for model evaluation.

        :param test: The actual test data.
        :param predictions: Predicted values by the model.
        :param naive: Boolean flag to indicate if the naive predictions should be considered.
        :return: A dictionary of performance metrics including MSE, RMSE, MAPE, MSPE, MAE, and R-squared.
        """
        try:
            match self.model_type:
                
                case 'ARIMA'|'SARIMA'|'SARIMAX':
                    test = test[:self.steps_ahead][self.target_column]
                    non_zero_indices = np.where(test != 0)
                    # Handle zero values in test_data for MAPE and MSPE calculations
                    if naive == False:
                        # predictions is a series containing a series in each element, so it must be trasformed to a simple series of values, keeping the original index
                        predictions_non_zero = pd.Series({index: item.iloc[0] for index, item in predictions.iteritems()})
                    else: 
                        predictions_non_zero = predictions.iloc[non_zero_indices]
                    test_non_zero = test.iloc[non_zero_indices]
                case 'LSTM':
                    non_zero_indices = np.where(test != 0)
                    predictions_non_zero = predictions[non_zero_indices]
                    test_non_zero = test[non_zero_indices]
                case 'XGB':
                    non_zero_indices = np.where(test != 0)
                    predictions_non_zero = predictions[non_zero_indices]
                    test_non_zero = test.iloc[non_zero_indices]

            performance_metrics = {}
            mse = mean_squared_error(test, predictions)
            rmse = np.sqrt(mse)
            performance_metrics['MSE'] = mse
            performance_metrics['RMSE'] = rmse
            performance_metrics['MAPE'] = mean_absolute_percentage_error(test_non_zero, predictions_non_zero)
            performance_metrics['MSPE'] = mean_squared_percentage_error(test_non_zero, predictions_non_zero)
            performance_metrics['MAE'] = mean_absolute_error(test, predictions)
            performance_metrics['R_2'] = r2_score(test, predictions)
            return performance_metrics
        
        except Exception as e:
            print(f"An error occurred during performance measurement: {e}")
            return None
