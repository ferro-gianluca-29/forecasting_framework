import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, mean_absolute_error
from classes.model_testing import ModelTest

class PerfMeasure(ModelTest):
    
    def get_performance_metrics(self, test, predictions):
        try:
            test = test[:self.steps_ahead][self.target_column]
            performance_metrics = {}
            mse = mean_squared_error(test, predictions)
            rmse = np.sqrt(mse)
            performance_metrics['MSE'] = mse
            performance_metrics['RMSE'] = rmse
            performance_metrics['MAPE'] = mean_absolute_percentage_error(test, predictions)
            performance_metrics['MAE'] = mean_absolute_error(test, predictions)
            
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
