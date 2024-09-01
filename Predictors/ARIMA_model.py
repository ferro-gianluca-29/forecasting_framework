import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.deterministic import Fourier
from tools.time_series_analysis import ARIMA_optimizer, ljung_box_test
from tqdm import tqdm
import pickle
from Predictors.Predictor import Predictor

class ARIMA_Predictor(Predictor):

    def __init__(self, run_mode, target_column=None, 
                 verbose=False, set_fourier=False):
        """
        Initializes an ARIMA_Predictor object with specified settings.

        :param target_column: The target column of the DataFrame to predict.
        :param verbose: If True, prints detailed outputs during the execution of methods.
        :param set_fourier: Boolean, if true use Fourier transformation on the data.
        """
        super().__init__(verbose=verbose)  

        self.run_mode = run_mode
        self.verbose = verbose
        self.target_column = target_column
        self.set_fourier = set_fourier
        self.ARIMA_order = []
        

    def train_model(self):
        """
        Trains an ARIMA model using the training dataset. 

        :return: A tuple containing the trained model, validation metrics and the index of last training/validation timestep.
        """
        try:
            
            best_order = list(ARIMA_optimizer(self.train, self.target_column, self.verbose))
            best_order[1] = 1
            # for debug: 
            #best_order = (2,2,2)
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
                last_index = model_fit.data.row_labels[-1] + 1


            else:

                valid = self.valid[self.target_column]
                # Number of time steps to forecast
                nforecasts = 3  
                nsteps_ahead = self.valid_steps
                # Choose whether to refit the model at each step
                refit_model = False  
                model = ARIMA(self.train[self.target_column], order=(best_order[0], best_order[1], best_order[2]))                
                model_fit = model.fit()

                last_train_index = model_fit.data.row_labels[-1] + 1
                valid_start_index = last_train_index
                valid.index = range(valid_start_index, valid_start_index + len(valid))    
        
                # Dictionary to store forecasts
                forecasts = {}
                forecasts[self.train.index[-1]] = model_fit.forecast(steps=nforecasts)
                # Recursive evaluation through the rest of the sample
                for t in range(valid.index[0], valid.index[0] + nsteps_ahead):
                    new_obs = valid.loc[[t]]
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

                # Running the LJUNG-BOX test for residual correlation
                ljung_box_test(model_fit)
                print("Model successfully trained.")

            return model_fit, valid_metrics, last_index
 
        except Exception as e:
            print(f"An error occurred during the model training: {e}")
            return None
        
        
    def test_model(self, model, last_index, forecast_type, ol_refit = False):
        """
        Tests an ARIMA model by performing one step-ahead predictions and optionally refitting the model.

        :param ol_refit: Boolean indicating whether to refit the model after each forecast.
        :param last_index: index of last training/validation timestep 
        :return: A pandas Series of the predictions.
        """
        try:
            print("\nTesting ARIMA model...\n")
            
            self.forecast_type = forecast_type
            test = self.test
            self.steps_ahead = self.test.shape[0]
            test_start_index = last_index
            test.index = range(test_start_index, test_start_index + len(test))
            
            match self.forecast_type:

                case "ol-one":

                    # ROLLING FORECASTS (ONE STEP-AHEAD, OPEN LOOP)
                    for t in tqdm(range(0, self.steps_ahead), desc="Rolling Forecasts"):
                        # Forecast one step at a time
                        y_hat = model.forecast()
                        # Append the forecast to the list
                        self.predictions.append(y_hat)
                        # Take the actual value from the test set to predict the next
                        y = test.iloc[t, test.columns.get_loc(self.target_column)]
                        # Update the model with the actual value
                        if ol_refit:
                            model = model.append([y], refit = True)
                        else:
                            model = model.append([y], refit = False)
                            
                    predictions = pd.Series(data=self.predictions, index=self.test.index[:self.steps_ahead])
                    print("Model testing successful.")        
                    return predictions
                
                case "cl-multi":

                    predictions = model.forecast(steps = self.steps_ahead)
                    predictions = pd.Series(predictions)
                    #predictions = pd.Series(data=model.forecast(steps=self.test.shape[0]), index=self.test.index)

                    return predictions
            
        except Exception as e:
            print(f"An error occurred during the model test: {e}")
            return None
        

    def unscale_predictions(self, predictions, folder_path):
        # Load scaler for unscaling data
        with open(f"{folder_path}/scaler.pkl", "rb") as file:
            scaler = pickle.load(file)
        
        # Unscale predictions
        predictions = np.array(predictions)
        predictions = predictions.reshape(-1, 1)
        predictions = scaler.inverse_transform(predictions) 
        predictions = predictions.flatten()
        predictions = pd.Series(predictions) 


    def plot_predictions(self, predictions):
        """
        Plots the ARIMA model predictions against the test data.

        :param predictions: The predictions made by the ARIMA model.
        """
        test = self.test[:self.steps_ahead][self.target_column]
        plt.plot(test.index, test, 'b-', label='Test Set')
        plt.plot(test.index, predictions, 'k--', label='ARIMA')
        plt.title(f'ARIMA prediction for feature: {self.target_column}')
        plt.xlabel('Time series index')
        plt.legend(loc='best')
        plt.tight_layout()
        plt.show()

    
