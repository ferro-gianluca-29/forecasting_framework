import pandas as pd
from matplotlib import pyplot as plt

class ModelTest():
    """
    A class for testing and visualizing the predictions of various types of forecasting models.

    :param model_type: The type of model to test ('ARIMA', 'SARIMAX', etc.).
    :param model: The model object to be tested.
    :param test: The test set.
    :param target_column: The target column in the dataset.
    :param forecast_type: The type of forecasting to be performed ('ol-one', etc.).
    :param steps_ahead: Number of forecasting steps to perform.
    """
    def __init__(self, model_type, model, test, target_column, forecast_type, steps_ahead):

        self.model_type = model_type
        self.model = model
        self.test = test
        self.target_column = target_column
        self.predictions = list()
        self.forecast_type = forecast_type            
        self.steps_ahead = steps_ahead

    def test_ARIMA_model(self, steps_jump = None, ol_refit = False):
        """
        Tests an ARIMA model by performing one step-ahead predictions and optionally refitting the model.

        :param steps_jump: Optional parameter to skip steps in the forecasting.
        :param ol_refit: Boolean indicating whether to refit the model after each forecast.
        :return: A pandas Series of the predictions.
        """
        try:
            print("\nTesting ARIMA model...\n")
            
            # ROLLING FORECASTS (ONE STEP-AHEAD, OPEN LOOP)

            for t in range(0, self.steps_ahead):
                # Forecast one step at a time
                y_hat = self.model.forecast()
                # Append the forecast to the list
                self.predictions.append(y_hat)
                # Take the actual value from the test set to predict the next
                y = self.test.iloc[t, self.test.columns.get_loc(self.target_column)]
                # Update the model with the actual value
                if ol_refit:
                    self.model = self.model.append([y], refit = True)
                else:
                    self.model = self.model.append([y], refit = False)
                    
            predictions = pd.Series(data=self.predictions, index=self.test.index[:self.steps_ahead])
            print("Model testing successful.")        
            return predictions

            
        except Exception as e:
            print(f"An error occurred during the model test: {e}")
            return None
        
    def test_SARIMAX_model(self, steps_jump = None, exog_test = None, ol_refit = False): 
        """
        Tests a SARIMAX model by performing one step-ahead predictions, using exogenous variables, and optionally refitting.

        :param steps_jump: Optional parameter to skip steps in the forecasting.
        :param exog_test: Optional exogenous variables for the test set.
        :param ol_refit: Boolean indicating whether to refit the model after each forecast.
        :return: A pandas Series of the predictions.
        """
        try:    
            print("\nTesting SARIMAX model...\n")
            
            # ROLLING FORECASTS (ONE STEP-AHEAD OPEN LOOP)

            match self.model_type:

                case 'SARIMA':
                    for t in range(0, self.steps_ahead):
                        y_hat = self.model.forecast()
                        # Insert the forecast into the list
                        self.predictions.append(y_hat)
                        # Take the actual value from the test set to predict the next
                        y = self.test.iloc[t, self.test.columns.get_loc(self.target_column)]
                        # Update the model with the actual value and exogenous
                        if ol_refit:
                            self.model = self.model.append([y], refit=True)
                        else:
                            self.model = self.model.append([y], refit=False)
                    predictions = pd.Series(data=self.predictions, index=self.test.index[:self.steps_ahead])            
                    print("Model testing successful.")
                    return predictions
                
                case 'SARIMAX':
                    for t in range(0, self.steps_ahead):
                        y_hat = self.model.forecast(exog = exog_test.iloc[t:t+1])
                        # Insert the forecast into the list
                        self.predictions.append(y_hat)
                        # Take the actual value from the test set to predict the next
                        y = self.test.iloc[t, self.test.columns.get_loc(self.target_column)]
                        # Take the exogenous values from the test set to predict the next
                        new_exog = exog_test.iloc[t:t+1]
                        # Set the index for new_exog equal to the last index of the model
                        new_exog.index = [self.model.data.row_labels[-1] + 1]                        
                        # Update the model with the actual value and exogenous
                        if ol_refit:
                            self.model = self.model.append([y], exog = new_exog, refit=True)
                        else:
                            self.model = self.model.append([y], exog = new_exog, refit=False) 
                    predictions = pd.Series(data=self.predictions, index=self.test.index[:self.steps_ahead])            
                    print("Model testing successful.")
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

            last_observation = train.iloc[-1][self.target_column]
            predictions = [last_observation] * self.steps_ahead
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
        plt.title(f'{self.forecast_type} prediction for feature: {self.target_column}')
        plt.xlabel('Time series index')
        plt.legend(loc='best')
        plt.tight_layout()
        plt.show()

    def SARIMAX_plot_pred(self, best_order, naive_predictions=None):
        """
        Plots the SARIMAX model predictions against the test data and optionally against naive predictions.

        :param best_order: The order of the SARIMAX model used.
        :param naive_predictions: Optional naive predictions for comparison.
        """
        target_test = self.test[[self.target_column]]
        test = target_test[:self.steps_ahead][self.target_column]
        plt.plot(test.index, test, 'b-', label='Test Set')
        plt.plot(test.index, self.predictions, 'k--', label=f'SARIMAX({best_order[0]}, {best_order[1]}, {best_order[2]}, {best_order[3]}, {best_order[4]}, {best_order[5]})')
        if naive_predictions is not None:
            plt.plot(test.index, naive_predictions, 'r--', label='Naive seasonal')
        plt.title(f'{self.forecast_type} prediction for feature: {self.target_column}')
        plt.xlabel('Time series index')
        plt.legend(loc='best')
        plt.tight_layout()
        plt.show()


    def plot_pred(self, test, predictions, time_values):
        """
        Plots model predictions against the test data.

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

   