import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import xgboost as xgb
from xgboost import plot_importance, plot_tree
from utils.time_series_analysis import ARIMA_optimizer, SARIMAX_optimizer, ljung_box_test
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from keras.layers import Dense,Dropout,SimpleRNN,LSTM
from keras.models import Sequential
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
        
    def train_LSTM_model(self, X_train, y_train, X_test, y_test):
        try:
            lstm_model = Sequential()

            lstm_model.add(LSTM(40,activation="tanh",return_sequences=True, input_shape=(X_train.shape[1],1)))
            lstm_model.add(Dropout(0.15))

            lstm_model.add(LSTM(40,activation="tanh",return_sequences=True))
            lstm_model.add(Dropout(0.15))

            lstm_model.add(LSTM(40,activation="tanh",return_sequences=False))
            lstm_model.add(Dropout(0.15))

            lstm_model.add(Dense(1))

            if self.verbose: lstm_model.summary()
            
            lstm_model.compile(optimizer="adam",loss="MSE")
            history= lstm_model.fit(X_train, y_train, epochs=200, validation_data=(X_test, y_test),batch_size=1000)
            my_loss= lstm_model.history.history['loss']
            valid_loss = history.history['val_loss']

            if self.verbose:
                # plot train and validation loss
                plt.plot(my_loss)
                plt.plot(valid_loss)
                plt.title('model train vs validation loss')
                plt.ylabel('loss')
                plt.xlabel('epoch')
                plt.legend(['train', 'validation'], loc='upper right')
                plt.show()
            
            return lstm_model, valid_loss
        
        except Exception as e:
            print(f"An error occurred during the model training: {e}")
            return None
        
    def train_XGB_model(self, X_train, y_train, X_test, y_test):
        try:
            # Define the XGBoost Regressor with improved parameters
            reg = xgb.XGBRegressor(
                n_estimators=100000,  # Number of boosting rounds (you can tune this)
                learning_rate=0.05,   # Learning rate (you can tune this)
                max_depth=5,          # Maximum depth of the trees (you can tune this)
                min_child_weight=1,   # Minimum sum of instance weight needed in a child
                gamma=0,              # Minimum loss reduction required to make a further partition
                subsample=0.8,        # Fraction of samples used for training
                colsample_bytree=0.8, # Fraction of features used for training
                reg_alpha=0,          # L1 regularization term on weights
                reg_lambda=1,         # L2 regularization term on weights
                objective='reg:squarederror',  # Objective function for regression
                random_state=42       # Seed for reproducibility
                                   )
            # Train the model with early stopping and verbose mode
            XGB_model = reg.fit(
                X_train,
                y_train,
                eval_set=[(X_train, y_train), (X_test, y_test)],
                eval_metric=['rmse', 'mae'],
                early_stopping_rounds=100,
                verbose=True  # Set to True to see training progress
            )

            valid_metrics = XGB_model.evals_result()

            return XGB_model, valid_metrics
        
        except Exception as e:
            print(f"An error occurred during the model training: {e}")
            return None               
    

