import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm
from statsmodels.tsa.deterministic import Fourier
import numpy as np
import matplotlib.dates as mdates  

class ModelTest():
    """
    A class for testing and visualizing the predictions of various types of forecasting models.

    :param model_type: The type of model to test ('ARIMA', 'SARIMAX', etc.).
    :param model: The model object to be tested.
    :param test: The test set.
    :param target_column: The target column in the dataset.
    :param forecast_type: The type of forecasting to be performed ('ol-one', etc.).
    """
    def __init__(self, model_type, model, test, target_column, forecast_type):
                
        self.model_type = model_type
        self.model = model
        self.test = test
        self.target_column = target_column
        self.predictions = list()
        self.forecast_type = forecast_type            
        self.steps_ahead = self.test.shape[0]
        

    def test_ARIMA_model(self, last_index, steps_jump = None, ol_refit = False):
        """
        Tests an ARIMA model by performing one step-ahead predictions and optionally refitting the model.

        :param steps_jump: Optional parameter to skip steps in the forecasting.
        :param ol_refit: Boolean indicating whether to refit the model after each forecast.
        :param last_index: index of last training/validation timestep 
        :return: A pandas Series of the predictions.
        """
        try:
            print("\nTesting ARIMA model...\n")
            
            test = self.test
            test_start_index = last_index
            test.index = range(test_start_index, test_start_index + len(test))
            
            match self.forecast_type:

                case "ol-one":

                    # ROLLING FORECASTS (ONE STEP-AHEAD, OPEN LOOP)
                    for t in tqdm(range(0, self.steps_ahead), desc="Rolling Forecasts"):
                        # Forecast one step at a time
                        y_hat = self.model.forecast()
                        # Append the forecast to the list
                        self.predictions.append(y_hat)
                        # Take the actual value from the test set to predict the next
                        y = test.iloc[t, test.columns.get_loc(self.target_column)]
                        # Update the model with the actual value
                        if ol_refit:
                            self.model = self.model.append([y], refit = True)
                        else:
                            self.model = self.model.append([y], refit = False)
                            
                    predictions = pd.Series(data=self.predictions, index=self.test.index[:self.steps_ahead])
                    print("Model testing successful.")        
                    return predictions
                
                case "cl-multi":

                    predictions = self.model.forecast(steps = self.steps_ahead)
                    predictions = pd.Series(predictions)
                    #predictions = pd.Series(data=self.model.forecast(steps=self.test.shape[0]), index=self.test.index)

                    return predictions
            
        except Exception as e:
            print(f"An error occurred during the model test: {e}")
            return None
        
    def test_SARIMAX_model(self, last_index, steps_jump = None, exog_test = None, ol_refit = False, period = 24, set_Fourier = False): 
        """
        Tests a SARIMAX model by performing one-step or multi-step ahead predictions, optionally using exogenous variables or applying refitting.

        :param last_index: Index of the last training/validation timestep.
        :param steps_jump: Optional parameter to skip steps in the forecasting.
        :param exog_test: Optional exogenous variables for the test set.
        :param ol_refit: Boolean indicating whether to refit the model after each forecast.
        :param period: The period for Fourier terms if set_Fourier is True.
        :param set_Fourier: Boolean flag to determine if Fourier terms should be included.
        :return: A pandas Series of the predictions.
        """
        try:    
            print("\nTesting SARIMAX model...\n")
            
            test = self.test
            test_start_index = last_index
            test_end_index = test_start_index + len(test)
            test.index = range(test_start_index, test_end_index)

            if set_Fourier:
                K = 3
                fourier = Fourier(period = period, order=K)
                test_fourier_terms = fourier.out_of_sample(steps=len(test), index=test.index)
                test_fourier_terms.index = range(test_start_index, test_end_index)

            match self.forecast_type:

                case "ol-one":

                    if set_Fourier:
                        # ROLLING FORECASTS (ONE STEP-AHEAD OPEN LOOP)

                        for t in tqdm(range(0, self.steps_ahead), desc="Rolling Forecasts"):

                            y_hat = self.model.forecast(exog = test_fourier_terms.iloc[t:t+1])
                            # Insert the forecast into the list
                            self.predictions.append(y_hat)
                            # Take the actual value from the test set to predict the next
                            y = test.iloc[t, test.columns.get_loc(self.target_column)]
                            # Update the model with the actual value and exogenous
                            if ol_refit:
                                self.model = self.model.append([y], exog = test_fourier_terms.iloc[t:t+1], refit=True)
                            else:
                                self.model = self.model.append([y], exog = test_fourier_terms.iloc[t:t+1], refit=False)
                    else:
                        # ROLLING FORECASTS (ONE STEP-AHEAD OPEN LOOP)
                        for t in tqdm(range(0, self.steps_ahead), desc="Rolling Forecasts"):
                            y_hat = self.model.forecast()
                            # Insert the forecast into the list
                            self.predictions.append(y_hat)
                            # Take the actual value from the test set to predict the next
                            y = test.iloc[t, test.columns.get_loc(self.target_column)]
                            # Update the model with the actual value and exogenous
                            if ol_refit:
                                self.model = self.model.append([y], refit=True)
                            else:
                                self.model = self.model.append([y], refit=False)

                    predictions = pd.Series(data=self.predictions, index=test.index[:self.steps_ahead])
                    print("Model testing successful.")
                    return predictions
                
                case "cl-multi":
                    if set_Fourier:
                        predictions = self.model.forecast(steps = self.steps_ahead, exog = test_fourier_terms)
                    else:
                        predictions = self.model.forecast(steps = self.steps_ahead)
                    predictions = pd.Series(predictions)
                    return predictions
                
        except Exception as e:
            print(f"An error occurred during the model test: {e}")
            return None 

    def naive_forecast(self, train): 
        """
        Performs a naive forecast using the last observed value from the training set.

        :param train: The training set.
        :return: A pandas Series of naive forecasts.
        """
        try:
            # Create a list of predictions
            predictions = list()
            
            if self.forecast_type == "cl-multi":
                # Calculate the mean of the target column
                mean_value = train[self.target_column].mean()
                # Create a list of predictions based on the mean
                predictions = [mean_value] * self.steps_ahead
                predictions = pd.Series(predictions, index=self.test.index[:self.steps_ahead])
            else:
                last_observation = train.iloc[-1][self.target_column]
                predictions.append(last_observation)
                for t in range(1, self.steps_ahead):
                    last_observation = self.test.iloc[t-1, self.test.columns.get_loc(self.target_column)]
                    predictions.append(last_observation)
                predictions = pd.Series(predictions, index=self.test.index[:self.steps_ahead])    
            return predictions
        
        except Exception as e:
            print(f"An error occurred during the naive model creation: {e}")
            return None

    def naive_seasonal_forecast(self, train, target_test, period=24):
        """
        Performs a seasonal naive forecast using the last observed seasonal cycle.

        :param train: The training set.
        :param target_test: The test set.
        :param period: The seasonal period to consider for the forecast.
        :return: A pandas Series of naive seasonal forecasts.
        """
        # Naive seasonal forecast: Use the last observed value from the same season as the prediction
        # period to make the forecast.
        
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

    def naive_mean_forecast(self, train):
        """
        Performs a naive forecast using the mean value of the training set.

        :param train: The training set.
        :return: A pandas Series of naive forecasts using the mean.
        """
        try:
            # Calculate the mean of the target column
            mean_value = train[self.target_column].mean()
            
            # Create a list of predictions based on the mean
            predictions = [mean_value] * self.steps_ahead
            predictions = pd.Series(predictions, index=self.test.index[:self.steps_ahead])
            
            return predictions
        
        except Exception as e:
            print(f"An error occurred during the naive mean model creation: {e}")
            return None

    def ARIMA_plot_pred(self, best_order, predictions, naive_predictions=None):
        """
        Plots the ARIMA model predictions against the test data and optionally against naive predictions.

        :param best_order: The order of the ARIMA model used.
        :param predictions: The predictions made by the ARIMA model.
        :param naive_predictions: Optional naive predictions for comparison.
        """
        test = self.test[:self.steps_ahead][self.target_column]
        plt.plot(test.index, test, 'b-', label='Test Set')
        plt.plot(test.index, predictions, 'k--', label=f'ARIMA({best_order[0]}, {best_order[1]}, {best_order[2]})')
        if naive_predictions is not None:
            plt.plot(test.index, naive_predictions, 'r--', label='Naive')
        plt.title(f'{self.model_type} prediction for feature: {self.target_column}')
        plt.xlabel('Time series index')
        plt.legend(loc='best')
        plt.tight_layout()
        plt.show()
    
    def NAIVE_plot_pred(self, naive_predictions):
        """
        Plots naive predictions against the test data.

        :param naive_predictions: The naive predictions to plot.
        """
        
        test = self.test[:self.steps_ahead][self.target_column]
        plt.plot(test.index, test, 'b-', label='Test Set')
        plt.plot(test.index, naive_predictions, 'r--', label='Naive')
        plt.title(f'{self.model_type} prediction for feature: {self.target_column}')
        plt.xlabel('Time series index')
        plt.legend(loc='best')
        plt.tight_layout()
        plt.show()



    def LSTM_plot_pred(self, predictions, y_test):
        """
        Plots LSTM model predictions for each data window in the test set.

        :param predictions: Predictions made by the LSTM model.
        :param y_test: Actual test values corresponding to the predictions.
        """
        
        for window_num in range(0, y_test.shape[0]):
            # Select the window for y_test and predictions
            test_window = y_test[window_num, :]
            pred_window = predictions[window_num, :]
            # Assuming 'date' is the column containing datetime in the 'test_data' DataFrame
            # And each point in the test window corresponds to a record in the DataFrame
            start_date = self.test['date'].iloc[0]
            # Create a date range for the x-axis with a frequency of 15 minutes
            date_range = pd.date_range(start=start_date, periods=len(test_window), freq='15T')

            # Initialize the plot
            plt.figure(figsize=(12, 6))
            # Plot the actual test data
            plt.plot(date_range, test_window, 'b-', label='Test Set', linewidth=2)
            # Plot the LSTM predictions
            plt.plot(date_range, pred_window, 'r--', label='LSTM Predictions', linewidth=2)
            
            # Set the title and labels
            plt.title('LSTM Prediction for First Window')
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


    def XGB_plot_pred(self, test, predictions, time_values):
        """
        Plots predictions made by an XGBoost model against the test data.

        :param test: The actual test data.
        :param predictions: The predictions made by the model.
        :param time_values: Time values corresponding to the test data.
        """
        title = f"Predictions made by {self.model_type} model"
        plt.figure(figsize=(16,4))
        plt.plot(time_values, test, color='blue',label='Actual values')
        plt.plot(time_values, predictions, alpha=0.7, color='orange',label='Predicted values')
        plt.title(title)
        plt.xlabel('Date and Time')
        plt.ylabel('Normalized scale')
        plt.xticks(rotation=45)
        plt.legend()
        plt.show()