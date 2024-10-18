import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.deterministic import Fourier
from tools.time_series_analysis import  ljung_box_test

from skforecast.ForecasterSarimax import ForecasterSarimax
from skforecast.Sarimax import Sarimax
from skforecast.model_selection_sarimax import backtesting_sarimax
from skforecast.model_selection_sarimax import grid_search_sarimax
import pmdarima
from pmdarima import auto_arima


from keras.layers import Dense,Flatten,Dropout,SimpleRNN,LSTM
from keras.models import Sequential
from keras.metrics import MeanAbsoluteError, MeanAbsolutePercentageError, RootMeanSquaredError

from keras.optimizers import Adam
from keras.losses import MeanSquaredError
from keras.callbacks import EarlyStopping

from statsmodels.tsa.seasonal import STL


from keras.layers import LSTM, Dropout, Dense, Reshape



from sklearn.preprocessing import MinMaxScaler

from skforecast.ForecasterRnn import ForecasterRnn
from skforecast.ForecasterRnn.utils import create_and_compile_model
from skforecast.model_selection_multiseries import backtesting_forecaster_multiseries


from tqdm import tqdm
import pickle
from Predictors.Predictor import Predictor

class Hybrid_Predictor(Predictor):
    """
    A class used to predict time series data using Seasonal ARIMA (SARIMA) models.
    """

    def __init__(self, run_mode, input_len, output_len, target_column=None, period = 24,
                 verbose=False, forecast_type='ol-one'):
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
        self.forecast_type = forecast_type
        self.period = period
        self.input_len = input_len
        self.output_len = output_len
        self.SARIMA_order = []
        

    def train_model(self, input_len, output_len):
        """
        Trains a SARIMAX model using the training dataset and exogenous variables, if specified.

        :return: A tuple containing the trained model, validation metrics, and the index of the last training/validation timestep
        """
        try:    


            # DECOMPOSE THE TRAINING SET

            #select the target column from the training set
            target_train = self.train[self.target_column]

            stl = STL(target_train, period = self.period)
            result = stl.fit()

            #result.plot()

            # add trend and seasonal components
            train_trend_seasonal = result.trend + result.seasonal
            train_trend_seasonal.index = self.train.index
            train_resid = result.resid

            train_trend_seasonal = pd.DataFrame(train_trend_seasonal)
            train_trend_seasonal = train_trend_seasonal.rename(columns={train_trend_seasonal.columns[0]: self.target_column})




            # CREATE SARIMA MODEL 

            d = 0
            D = 0

            # Selection of the model with best AIC score
            """model = auto_arima(
                        y=self.train[self.target_column],
                        start_p=0,
                        start_q=0,
                        max_p=4,
                        max_q=4,
                        seasonal=True,
                        m = self.period,
                        test='adf',
                        d=None,  # Let auto_arima determine the optimal 'd'
                        D=None,
                        trace=True,
                        error_action='warn',  # Show warnings for troubleshooting
                        suppress_warnings=False,
                        stepwise=True
                        )
            
            order = model.order
            seasonal_order = model.seasonal_order"""

            period = self.period  
            target_train = self.train[self.target_column]

            # Select directly the order (Comment if using the AIC search)
            order = (2,1,1)
            seasonal_order = (2,0,1, 24)
            
            best_order = (order, seasonal_order)
            print(f"Best order found: {best_order}")
            

            self.SARIMA_order = best_order
            print("\nTraining the SARIMAX model...")

            sarima_model = Sarimax( order = order,
                                        seasonal_order=seasonal_order,
                                        #maxiter = 500
                                        )
            

            # FIT SARIMA ON TREND_SEASONAL COMPONENT
            
            sarima_model.fit(y=target_train)    

            sarima_residuals = pd.DataFrame(sarima_model.sarimax_res.resid, columns=[self.target_column])
   

            # CREATE LSTM MODEL WITH KERAS FUNCTIONS

            def build_model(input_len, output_len, units=128, dropout_rate=0.2, learning_rate=0.001):
                
                optimizer = Adam(learning_rate=learning_rate)
                loss = 'mean_squared_error'
                input_shape = (input_len, 1)  
                
                model = Sequential()
                model.add(LSTM(units, activation='tanh', return_sequences=False, input_shape=input_shape))
                model.add(Dropout(dropout_rate)) 
                model.add(Dense(output_len, activation='linear'))
                model.add(Reshape((output_len, 1)))  

                model.compile(optimizer=optimizer, loss=loss)
                return model

            model = build_model(self.input_len, self.output_len)

            lstm_forecaster = ForecasterRnn(
                                regressor = model,
                                levels = self.target_column,
                                transformer_series = None,
                                fit_kwargs={
                                    "epochs": 2,  # Number of epochs to train the model.
                                    "batch_size": 100,  # Batch size to train the model.
                                           },
                                    )    
            
            # scale residuals before feeding them to the LSTM
            scaler = MinMaxScaler()
            # fit the scaler on the training set
            sarima_residuals = sarima_residuals.applymap(lambda x: x.replace(',', '.') if isinstance(x, str) else x)
            scaler.fit(sarima_residuals[sarima_residuals.columns])

            # fit scaler on train data to later scale test and predictions
            scaler_train = MinMaxScaler()
            temp_train = self.train.applymap(lambda x: x.replace(',', '.') if isinstance(x, str) else x)
            scaler_train.fit(temp_train[temp_train.columns[0:temp_train.columns.shape[0] - 1]])

            # scale training data    
            sarima_residuals[sarima_residuals.columns] = scaler.transform(sarima_residuals[sarima_residuals.columns])

            lstm_forecaster.fit(sarima_residuals[[self.target_column]])

            steps = output_len


            predictions = []


            if self.forecast_type == 'ol_multi':

                for i in tqdm(range(0, len(self.test), steps), desc="Forecasting"):
                    current_steps = min(steps, len(self.test) - i)  # Adjust steps if remaining steps are less

                    # Forecast with SARIMA
                    sarima_pred = sarima_model.predict(steps=current_steps
                                                            )

                    # Forecast residuals with LSTM
                    # Prepare residuals input for LSTM (use the latest residuals)

                    lstm_pred = lstm_forecaster.predict(steps=current_steps)
                    # Inverse scale the residuals
                    lstm_pred = scaler.inverse_transform(lstm_pred.to_numpy().reshape(-1, 1)).flatten()

                    # Combine predictions
                    combined_pred = sarima_pred.values.flatten() + lstm_pred.flatten()

                    # Append combined predictions
                    predictions.extend(combined_pred)

                    # Update history with actual values (if available) for next iteration
                    actual_values = self.test[self.target_column].iloc[i:i+current_steps]
                    sarima_model.append(actual_values, refit=False)

            elif self.forecast_type == 'ol-one':
                # One-step ahead forecasting loop
                for i in tqdm(range(len(self.test)), desc="Forecasting"):
                    # Forecast with SARIMA for one step
                    sarima_pred = sarima_model.predict(steps=1)

                    # Forecast residual with LSTM for one step
                    lstm_pred = lstm_forecaster.predict(steps=1)
                    # Inverse scale the residual
                    lstm_pred = scaler.inverse_transform(lstm_pred.to_numpy().reshape(-1, 1)).flatten()[0]

                    # Combine predictions
                    combined_pred = sarima_pred.values[0] + lstm_pred

                    # Append combined prediction
                    predictions.append(combined_pred)

                    # Update history with actual value for next iteration
                    actual_value = self.test[self.target_column].iloc[i]
                    sarima_model.append([actual_value], refit=False)

            prediction_index = self.test.index
            predictions_df = pd.DataFrame({self.target_column: predictions}, index=prediction_index)

            return sarima_model, predictions_df, scaler_train

        
        except Exception as e:
                print(f"An error occurred during the model training: {e}")
                return None
        

    def test_model(self, forecaster, last_index, forecast_type, output_len, ol_refit = False, period = 24): 
        """ METHOD NOT USED FOR HYBRID PREDICTOR"""
        try:    
            
            
            return 
                
        
        except Exception as e:
            print(f"An error occurred during the model test: {e}")
            return None 
        

    def plot_predictions(self, predictions):
        """
        Plots the SARIMA model predictions against the test data.

        :param predictions: The predictions made by the SARIMA model
        """
        test = self.test[self.target_column]
        plt.plot(test.index, test, 'b-', label='Test Set')
        plt.plot(test.index, predictions, 'k--', label='ARIMA')
        plt.title(f'SARIMA prediction for feature: {self.target_column}')
        plt.xlabel('Time series index')
        plt.legend(loc='best')
        plt.tight_layout()
        plt.show()


    
