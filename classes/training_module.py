from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from utils.time_series_analysis import ARIMA_optimizer, SARIMAX_optimizer, ljung_box_test
from prophet import Prophet
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

class ModelTraining():
    def __init__(self, model_type: str, train, valid = None, target_column = None, 
                 verbose = False):
        
        self.verbose = verbose
        self.model_type = model_type
        self.train = train
        self.valid = valid
        self.target_column = target_column
        self.ARIMA_order = []
        self.SARIMAX_order = []
        
    def train_ARIMA_model(self): 
        
        try:
            #best_order = ARIMA_optimizer(self.train, self.target_column, self.verbose)
            best_order = (1,1,1)
            self.ARIMA_order = best_order
            print("\nTraining the ARIMA model...")

            # Training the model with the best parameters found
            if self.valid is None:

                model = ARIMA(self.train[self.target_column], order=(best_order[0], best_order[1], best_order[2]))                
                model_fit = model.fit()

                # Running the LJUNG-BOX test for residual correlation
                ljung_box_test(model_fit)
                print("Model successfully trained.")
                valid_metrics = None

            else:

                valid = self.valid[self.target_column]
                # Number of time steps to forecast
                nforecasts = 3  
                # Choose whether to refit the model at each step
                refit_model = False  
                model = ARIMA(self.train[self.target_column], order=(best_order[0], best_order[1], best_order[2]))                
                model_fit = model.fit()
                # Dictionary to store forecasts
                forecasts = {}
                forecasts[self.train.index[-1]] = model_fit.forecast(steps=nforecasts)
                # Recursive evaluation through the rest of the sample
                for t in range(valid.index[0],valid.index[0] + 10):
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
                valid_rmse = (flattened**2).mean(axis=1)**0.5
                valid_mse = (flattened**2).mean(axis=1)
                valid_mae = ((flattened).abs()).mean(axis=1)
                perc_forecast_errors = forecasts.apply(lambda column: (valid - column)/ valid).reindex(forecasts.index).apply(flatten)
                valid_mape = ((perc_forecast_errors).abs()).mean(axis=1)
                valid_metrics = [valid_rmse, valid_mse, valid_mae, valid_mape]

                # Running the LJUNG-BOX test for residual correlation
                ljung_box_test(model_fit)
                print("Model successfully trained.")

            return model_fit, valid_metrics
 
        except Exception as e:
            print(f"An error occurred during the model training: {e}")
            return None       
        
    def train_SARIMAX_model(self, target_train, exog_train, exog_valid = None, period = 24): 
        try:        
            target_train = self.train[[self.target_column]]
            #best_order = SARIMAX_optimizer(target_train, self.target_column, period, exog_train, self.verbose)
            best_order = (1,1,1,1,1,1)
            self.SARIMAX_order = best_order
            print("\nTraining the SARIMAX model...")

            # Training the model with the best parameters found
            if self.valid is None:
                if self.model_type == 'SARIMAX':
                    exog = exog_train
                else:
                    exog = None
                model = SARIMAX(target_train, exog = exog, order = (best_order[0], best_order[1], best_order[2]),
                                    seasonal_order=(best_order[3], best_order[4], best_order[5], period),
                                    simple_differencing=False
                                    )
                model_fit = model.fit()
                valid_metrics = None
                # Running the LJUNG-BOX test for residual correlation
                ljung_box_test(model_fit)
                print("Model successfully trained.")
            else: 
                valid = self.valid[self.target_column]
                # Number of time steps to forecast
                nforecasts = 3 
                # Choose whether to refit the model at each step
                refit_model = False
                model = SARIMAX(target_train, exog_train, order = (best_order[0], best_order[1], best_order[2]),
                                    seasonal_order=(best_order[3], best_order[4], best_order[5], period),
                                    simple_differencing=False
                                    )
                model_fit = model.fit()
                # Dictionary to store forecasts
                forecasts = {}
                forecasts[self.train.index[-1]] = model_fit.forecast(exog = exog_valid.iloc[:nforecasts], steps=nforecasts)
                # Recursive evaluation through the rest of the sample
                for t in valid.index:
                    if t + nforecasts > max(exog_valid.index):
                        print(f"No more exog data available at timestep {t} to continue training. Ending training.")
                        print(forecasts)
                        break  # Exit the loop if there are not enough exog data
                    new_obs = valid.loc[t:t]
                    new_exog = exog_valid.loc[t:t]
                    model_fit = model_fit.append(new_obs, exog = new_exog, refit=refit_model)
                    forecasts[new_obs.index[0]] = model_fit.forecast(exog = exog_valid.loc[t:t+nforecasts - 1], steps=nforecasts)

                # Combine all forecasts into a DataFrame
                forecasts = pd.concat(forecasts, axis=1)

                # Calculate and print forecast errors
                forecast_errors = forecasts.apply(lambda column: valid - column).reindex(forecasts.index)
                
                # Reshape errors by horizon and calculate RMSE
                def flatten(column):
                    return column.dropna().reset_index(drop=True)

                flattened = forecast_errors.apply(flatten)
                flattened.index = (flattened.index + 1).rename('horizon')
                valid_rmse = (flattened**2).mean(axis=1)**0.5
                valid_mse = (flattened**2).mean(axis=1)
                valid_mae = ((flattened).abs()).mean(axis=1)
                perc_forecast_errors = forecasts.apply(lambda column: (valid - column)/ valid).reindex(forecasts.index).apply(flatten)
                valid_mape = ((perc_forecast_errors).abs()).mean(axis=1)
                valid_metrics = [valid_rmse, valid_mse, valid_mae, valid_mape]
                
                # Running the LJUNG-BOX test for residual correlation
                ljung_box_test(model_fit)
                print("Model successfully trained.")

            return model_fit, valid_metrics
        
        except Exception as e:
            print(f"An error occurred during the model training: {e}")
            return None           
        
    
