import pandas as pd
from matplotlib import pyplot as plt

class ModelTest():

    def __init__(self, model, test, target_column, forecast_type, steps_ahead):

        self.model = model
        self.test = test
        self.target_column = target_column
        self.predictions = list()
        self.forecast_type = forecast_type            
        self.steps_ahead = steps_ahead

    def test_ARIMA_model(self, steps_jump = None, ol_refit = False):

        try:
            print("\nTesting ARIMA model...\n")
    
            # MULTI STEP-AHEAD FORECASTS (CLOSED LOOP)
            if self.forecast_type == 'cl-multi':

                # Make the forecast
                self.predictions = self.model.forecast(steps=self.steps_ahead)
                print("Model testing successful.")
                return self.predictions
            
            # ROLLING FORECASTS (ONE STEP-AHEAD, OPEN LOOP)
            elif self.forecast_type == 'ol-one':

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
                print("Model testing successful.")        
                return self.predictions

            # ROLLING FORECASTS (MULTI STEP-AHEAD, OPEN LOOP)
            elif self.forecast_type == 'ol-multi':

                for t in range(0, self.steps_ahead, steps_jump):
                    # Forecast 'steps_jump' steps at a time
                    y_hat = self.model.forecast(steps=steps_jump)
                    # Append the forecasts to the list
                    self.predictions.extend(y_hat)
                    # Take the actual value from the test set to predict the 'step_jump' next
                    y = self.test.iloc[t, self.test.columns.get_loc(self.target_column)]
                    # Update the model with the actual value
                    if ol_refit:
                        self.model = self.model.append([y], refit=True)
                    else:
                        self.model = self.model.append([y], refit=False)
                print("Model testing successful.")
                return self.predictions
            
        except Exception as e:
            print(f"An error occurred during the model test: {e}")
            return None
        
    def test_SARIMAX_model(self, steps_jump = None, exog_test = None, ol_refit = False): 
            try:    
                print("\nTesting SARIMAX model...\n")

                # MULTI STEP-AHEAD FORECASTS (CLOSED LOOP)
                if self.forecast_type == 'cl-multi':

                    # Make the forecast
                    self.predictions = self.model.forecast(steps=self.steps_ahead, exog=exog_test[:self.steps_ahead])
                    print("Model testing successful.")
                    return self.predictions
                
                # ROLLING FORECASTS (ONE STEP-AHEAD OPEN LOOP)
                if self.forecast_type == 'ol-one':

                    for t in range(0, self.steps_ahead):
                        # Forecast one step at a time
                        y_hat = self.model.forecast(exog = exog_test.iloc[t:t+1])
                        # Insert the forecast into the list
                        self.predictions.append(y_hat)
                        # Take the actual value from the test set to predict the next
                        y = self.test.iloc[t, self.test.columns.get_loc(self.target_column)]
                        # Take the exogenous values from the test set to predict the next
                        new_exog = exog_test.iloc[t:t+1]
                        # Update the model with the actual value and exogenous
                        if ol_refit:
                            self.model = self.model.append([y], exog = new_exog, refit=True)
                        else:
                            self.model = self.model.append([y], exog = new_exog, refit=False) 
                    print("Model testing successful.")
                    return self.predictions
   
                # ROLLING FORECASTS (MULTI STEP-AHEAD OPEN LOOP)
                elif self.forecast_type == 'ol-multi':

                    for t in range(0, int(self.steps_ahead/steps_jump)):
                        # Forecast 'period' steps at a time
                        y_hat = self.model.forecast(steps=steps_jump, exog=exog_test.iloc[t*steps_jump:(t+1)*steps_jump])
                        # Append the forecasts to the list
                        self.predictions.extend(y_hat)
                        # Take the actual value from the test set to predict the 'period' next
                        y = self.test.iloc[t*steps_jump:(t+1)*steps_jump, self.test.columns.get_loc(self.target_column)]
                        # Take the exogenous values from the test set to predict the 'period' next
                        new_exog = exog_test.iloc[t*steps_jump:(t+1)*steps_jump]
                        # Update the model with the actual value and exogenous
                        if ol_refit:
                            self.model = self.model.append(y, exog=new_exog, refit=True)
                        else:
                            self.model = self.model.append(y, exog=new_exog, refit=False) 
                    print("Model testing successful.")        
                    return self.predictions
                
            except Exception as e:
                print(f"An error occurred during the model test: {e}")
                return None 

    def naive_forecast(self, train): 
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
        Displays ARIMA model predictions on a graph.
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
        Displays SARIMAX model predictions on a graph.
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