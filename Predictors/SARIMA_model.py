import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.deterministic import Fourier
from tools.time_series_analysis import SARIMAX_optimizer, ljung_box_test
from tqdm import tqdm
import pickle
from Predictors.Predictor import Predictor

class SARIMA_Predictor(Predictor):
    """
    A class used to predict time series data using Seasonal ARIMA (SARIMA) models.
    """

    def __init__(self, run_mode, target_column=None, period = 24,
                 verbose=False, set_fourier=False):
        """
        Constructs all the necessary attributes for the SARIMA_Predictor object.

        :param run_mode: The mode in which the predictor runs
        :param target_column: The target column of the DataFrame to predict
        :param period: Seasonal period of the SARIMA model
        :param verbose: If True, prints detailed outputs during the execution of methods
        :param set_fourier: Boolean, if true use Fourier transformation on the data
        """

        super().__init__(verbose=verbose)  

        self.run_mode = run_mode
        self.verbose = verbose
        self.target_column = target_column
        self.set_fourier = set_fourier
        self.period = period
        self.SARIMA_order = []
        

    def train_model(self):
        """
        Trains a SARIMAX model using the training dataset and exogenous variables, if specified.

        :return: A tuple containing the trained model, validation metrics, and the index of the last training/validation timestep
        """
        try:    

            period = self.period    
            target_train = self.train[[self.target_column]]
            
            #best_order = SARIMAX_optimizer(target_train, self.target_column, period, d = 0, D = 0, verbose = self.verbose)

            #if optimizer is too slow, set the order after plotting ACF and PACF:  
            best_order = (0,1,0,2,1,2)

            self.SARIMA_order = best_order
            print("\nTraining the SARIMAX model...")

            if self.valid is None:

                if self.set_fourier == True:
                    sarima_order = best_order[:3]
                    sarima_seasonal_order = list(best_order[3:6])
                    sarima_seasonal_order.append(period)
                    K = 3
                    fourier = Fourier(period=period, order=K)
                    train_fourier_terms = fourier.in_sample(target_train.index)

                    model = SARIMAX(target_train,
                                    order = sarima_order,
                                   # seasonal_order = sarima_seasonal_order, # if optimizer is disabled, comment this line
                                    exog = train_fourier_terms,
                                    enforce_stationarity=False,
                                    enforce_invertibility=False,
                                    low_memory = True
                                    )
                else:
                    model = SARIMAX(target_train, order = (best_order[0], best_order[1], best_order[2]),
                                        seasonal_order=(best_order[3], best_order[4], best_order[5], period),
                                        )
                model_fit = model.fit()
                valid_metrics = None
                
                last_index = model_fit.data.row_labels[-1] + 1
                # Running the LJUNG-BOX test for residual correlation
                ljung_box_test(model_fit)
                print("Model successfully trained.")


            else:
                valid = self.valid[self.target_column]
                # Number of time steps to forecast
                nforecasts = 3 
                # Choose whether to refit the model at each step
                refit_model = False
                model = SARIMAX(target_train, order = (best_order[0], best_order[1], best_order[2]),
                                seasonal_order=(best_order[3], best_order[4], best_order[5], period),
                                simple_differencing=False
                            )
                model_fit = model.fit()

                last_train_index = model_fit.data.row_labels[-1] + 1
                valid_start_index = last_train_index
                valid.index = range(valid_start_index, valid_start_index + len(valid))
                
                # Dictionary to store forecasts
                forecasts = {}
                forecasts[self.train.index[-1]] = model_fit.forecast(steps=nforecasts)
                # Recursive evaluation through the rest of the sample 
                for t in valid.index:
                    if t + nforecasts > max(valid.index):
                        print(f"No more valid data available at timestep {t} to continue training. Ending training.")
                        break  # Exit the loop if there are not enough valid data
                    new_obs = valid.loc[t:t]
                    model_fit = model_fit.append(new_obs, refit=refit_model)
                    forecasts[new_obs.index[0]] = model_fit.forecast(steps=nforecasts)

                # Combine all forecasts into a DataFrame
                forecasts = pd.concat(forecasts, axis=1)

                # Calculate and print forecast errors
                forecast_errors = forecasts.apply(lambda column: valid - column).reindex(forecasts.index)
                
                # Reshape errors by horizon and calculate RMSE
                def flatten(column):
                    return column.dropna().reset_index(drop=True)

                flattened = forecast_errors.apply(flatten)
                flattened.index = (flattened.index + 1).rename('horizon')
                perc_forecast_errors = forecasts.apply(lambda column: (valid - column)/ valid).reindex(forecasts.index).apply(flatten)
                valid_metrics = {}
                valid_metrics['valid_rmse'] = (flattened**2).mean(axis=1)**0.5
                valid_metrics['valid_mse'] = (flattened**2).mean(axis=1)
                valid_metrics['valid_mae'] = ((flattened).abs()).mean(axis=1)
                valid_metrics['valid_mape'] = ((perc_forecast_errors).abs()).mean(axis=1)

                last_index = model_fit.data.row_labels[-1] + 1

            return model_fit, valid_metrics, last_index
        
        except Exception as e:
                print(f"An error occurred during the model training: {e}")
                return None
        

    def test_model(self, model, last_index, forecast_type, ol_refit = False, period = 24, set_Fourier = False): 
        """
        Tests a SARIMAX model by performing one-step or multi-step ahead predictions, optionally using exogenous variables or applying refitting.

        :param model: The SARIMAX model to be tested
        :param last_index: Index of the last training/validation timestep
        :param forecast_type: Type of forecasting ('ol-one' for open-loop one-step ahead, 'cl-multi' for closed-loop multi-step)
        :param ol_refit: Boolean indicating whether to refit the model after each forecast
        :param period: The period for Fourier terms if set_fourier is true
        :param set_fourier: Boolean flag to determine if Fourier terms should be included
        :return: A pandas Series of the predictions
        """
        try:    
            print("\nTesting SARIMA model...\n")
            
            test = self.test
            test_start_index = last_index
            test_end_index = test_start_index + len(test)
            test.index = range(test_start_index, test_end_index)
            self.steps_ahead = self.test.shape[0]
            self.forecast_type = forecast_type
            

            if set_Fourier:
                K = 3
                fourier = Fourier(period = period, order=K)
                test_fourier_terms = fourier.out_of_sample(steps=len(test), index=test.index)
                test_fourier_terms.index = range(test_start_index, test_end_index)

            match self.forecast_type:

                case "ol-one":

                    predictions = []

                    if set_Fourier:
                        # ROLLING FORECASTS (ONE STEP-AHEAD OPEN LOOP)

                        for t in tqdm(range(0, self.steps_ahead), desc="Rolling Forecasts"):

                            y_hat = model.forecast(exog = test_fourier_terms.iloc[t:t+1])
                            # Insert the forecast into the list
                            predictions.append(y_hat)
                            # Take the actual value from the test set to predict the next
                            y = test.iloc[t, test.columns.get_loc(self.target_column)]
                            # Update the model with the actual value and exogenous
                            if ol_refit:
                                model = model.append([y], exog = test_fourier_terms.iloc[t:t+1], refit=True)
                            else:
                                model = model.append([y], exog = test_fourier_terms.iloc[t:t+1], refit=False)
                    else:
                        # ROLLING FORECASTS (ONE STEP-AHEAD OPEN LOOP)
                        for t in tqdm(range(0, self.steps_ahead), desc="Rolling Forecasts"):
                            y_hat = model.forecast()
                            # Insert the forecast into the list
                            predictions.append(y_hat)
                            # Take the actual value from the test set to predict the next
                            y = test.iloc[t, test.columns.get_loc(self.target_column)]
                            # Update the model with the actual value and exogenous
                            if ol_refit:
                                model = model.append([y], refit=True)
                            else:
                                model = model.append([y], refit=False)

                    predictions = pd.Series(data=predictions, index=test.index[:self.steps_ahead])
                    print("Model testing successful.")
                    return predictions
                
                case "ol-multi":
                    
                    predictions = []
                    for t in tqdm(range(0, self.steps_ahead, period), desc="Rolling Forecasts"):
                        # Forecast a period of steps at a time
                        y_hat = model.forecast(steps=period)
                        # Append the forecasts to the list
                        predictions.extend(y_hat)
                        # Take the actual value from the test set to predict the next period
                        y = test.iloc[t, test.columns.get_loc(self.target_column)]
                        # Update the model with the actual value
                        if ol_refit:
                            model = model.append([y], refit=True)
                        else:
                            model = model.append([y], refit=False)

                    predictions = pd.Series(data=predictions, index=test.index[:self.steps_ahead])
                    print("Model testing successful.")
                    return predictions
                
                case "cl-multi":
                    if set_Fourier:
                        predictions = model.forecast(steps = self.steps_ahead, exog = test_fourier_terms)
                    else:
                        predictions = model.forecast(steps = self.steps_ahead)
                    predictions = pd.Series(predictions)
                    return predictions
                
        except Exception as e:
            print(f"An error occurred during the model test: {e}")
            return None 
        

    def unscale_predictions(self, predictions, folder_path):
        """
        Unscales the predictions using the scaler saved during model training.

        :param predictions: The scaled predictions that need to be unscaled
        :param folder_path: Path to the folder containing the scaler object
        """
        # Load scaler for unscaling data
        with open(f"{folder_path}/scaler.pkl", "rb") as file:
            scaler = pickle.load(file)
        
        # Unscale predictions
        predictions = np.array(predictions)
        predictions = predictions.reshape(-1, 1)
        predictions = scaler.inverse_transform(predictions) 
        predictions = predictions.flatten()
        predictions = pd.Series(predictions) 

        return predictions


    def plot_predictions(self, predictions):
        """
        Plots the SARIMA model predictions against the test data.

        :param predictions: The predictions made by the SARIMA model
        """
        test = self.test[:self.steps_ahead][self.target_column]
        plt.plot(test.index, test, 'b-', label='Test Set')
        plt.plot(test.index, predictions, 'k--', label='ARIMA')
        plt.title(f'SARIMA prediction for feature: {self.target_column}')
        plt.xlabel('Time series index')
        plt.legend(loc='best')
        plt.tight_layout()
        plt.show()

    
