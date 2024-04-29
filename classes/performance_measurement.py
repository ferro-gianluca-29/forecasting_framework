import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, mean_absolute_error, r2_score
from sktime.performance_metrics.forecasting import mean_squared_percentage_error
from classes.model_testing import ModelTest

class PerfMeasure(ModelTest):
    
    def get_performance_metrics(self, test, predictions, naive = False):
        try:
            if self.model_type == 'ARIMA' or self.model_type == 'SARIMA':
                test = test[:self.steps_ahead][self.target_column]
                non_zero_indices = np.where(test != 0)
                # Handle zero values in test_data for MAPE and MSPE calculations
                if naive == False:
                    predictions_non_zero = pd.Series({index: item.iloc[0] for index, item in predictions.iteritems()})
                else: 
                    predictions_non_zero = predictions.iloc[non_zero_indices]
            else:
                non_zero_indices = np.where(test != 0)
                predictions_non_zero = predictions.iloc[non_zero_indices]
                
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

    def plot_stats_performance(self, model_type, metrics, metrics_naive):
        try:
            metric_names = list(metrics.keys())
            metric_values = list(metrics.values())
            naive_metric_names = list(metrics_naive.keys())
            naive_metric_values = list(metrics_naive.values())
            
            plt.figure()
            x = np.arange(len(metric_names))
            width = 0.3
            
            plt.bar(x, metric_values, width, color='orange', label=model_type)
            if model_type == 'SARIMAX' or model_type == 'PROPHET':
                plt.bar(x + width, naive_metric_values, width, color='blue', label='Seasonal Naive Model')
            else:
                plt.bar(x + width, naive_metric_values, width, color='blue', label='Naive Model')
            plt.ylabel('Metric Values')
            plt.title('Performance Comparison')
            plt.xticks(ticks=x + width/2, labels=metric_names)
            plt.legend()
            plt.show()
            
        except Exception as e:
            print(f"An error occurred during chart creation: {e}")
            
    def print_stats_performance(self, model_type, metrics, metrics_naive):
        try:
            print(f'\n===== Model Performance: {model_type} =====')
            for key, value in metrics.items():
                print(f'{key}: {value}')
            
            if model_type == 'SARIMAX' or model_type == 'PROPHET':
                print(f'\n===== Model Performance: Seasonal Naive =====')
                for key, value in metrics_naive.items():
                    print(f'{key}: {value}')
            else:
                print(f'\n===== Model Performance: Naive =====')
                for key, value in metrics_naive.items():
                    print(f'{key}: {value}')
                    
        except Exception as e:
            print(f"An error occurred during performance printing: {e}")
