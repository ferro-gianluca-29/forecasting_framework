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


from OnlineSTL import OnlineSTL

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


            # CREATE SARIMA MODEL 

            d = 0
            D = 0

            # Selection of the model with best AIC score
            """sarima_model = auto_arima(
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
            
            order = sarima_model.order
            seasonal_order = sarima_model.seasonal_order"""


            # DECOMPOSE THE TRAINING SET WITH STATSMODELS STL

            #select the target column from the training set
            target_train = self.train[self.target_column]

            stl = STL(target_train, period = self.period)
            result = stl.fit()

            #result.plot()

            # add trend and seasonal components 
            train_trend_seasonal = result.trend + result.seasonal
            train_trend_seasonal = pd.DataFrame(train_trend_seasonal)
            train_trend_seasonal = train_trend_seasonal.rename(columns={train_trend_seasonal.columns[0]: self.target_column})

            # extract residual component for the LSTM
            train_resid = result.resid
            train_resid = pd.DataFrame(train_resid)
            train_resid = train_resid.rename(columns={train_resid.columns[0]: self.target_column})

            period = self.period  
            target_train = self.train[self.target_column]

            # Select directly the order (Comment if using the AIC search)
            order = (4,1,0)
            seasonal_order = (2,1,0, 24)
            
            best_order = (order, seasonal_order)
            print(f"Best order found: {best_order}")
            

            self.SARIMA_order = best_order
            print("\nTraining the SARIMAX model...")

            sarima_model = Sarimax( order = order,
                                        seasonal_order=seasonal_order,
                                        #maxiter = 500
                                        )
            

            # FIT SARIMA ON TREND_SEASONAL COMPONENT
            
            sarima_model.fit(y=train_trend_seasonal[self.target_column])    

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

            lstm_model = build_model(self.input_len, self.output_len)

            lstm_forecaster = ForecasterRnn(
                                regressor = lstm_model,
                                levels = self.target_column,
                                lags = self.input_len,
                                transformer_series = MinMaxScaler(),
                                fit_kwargs={
                                    "epochs": 300,  # Number of epochs to train the model.
                                    "batch_size": 32,  # Batch size to train the model.
                                           },
                                    )    

            lstm_forecaster.fit(train_resid[[self.target_column]])

            predictions = []


            # Instantiate online_stl object

            periods = [self.period]

            online_stl = OnlineSTL(self.train[self.target_column], periods = periods)

            if self.forecast_type == 'ol-one':

                # Prepara l'indice completo da usare durante il ciclo
                last_dates_full = self.train.index[-self.input_len:].append(self.test.index)

                # Inizializza la lista delle predizioni
                predictions = []

                # Inizializza test_residuals come DataFrame per mantenere gli indici
                test_residuals = pd.Series(index=self.test.index, dtype=float)

                # Prima fase: finché non abbiamo abbastanza residui di test
                for i in tqdm(range(self.input_len), desc="Forecasting: Using last training timesteps..."):

                    # Previsione del componente stagionale e decomposizione del nuovo punto dati
                    trend, seasonal, residual = online_stl.update(self.test[self.target_column].iloc[i])
                    test_residuals.iloc[i] = residual

                    # Concatena i dati del train con i residui di test calcolati finora (escludendo l'indice attuale)
                    last_window_df = pd.concat([
                    train_resid[self.target_column].iloc[-(self.input_len - i):],
                    test_residuals.iloc[:i]
                            ], axis=0).to_frame(name=self.target_column)

                    # Previsione del componente residuale con LSTM per un passo
                    lstm_pred = lstm_forecaster.predict(steps=1, last_window=last_window_df)

                    # Forecast seasonal trend component with SARIMA for one step
                    sarima_pred = sarima_model.predict(steps=1)

                    # Combine predictions
                    combined_pred = sarima_pred.iloc[0, 0] + lstm_pred.iloc[0, 0]

                    # Append combined prediction
                    predictions.append(combined_pred)

                    # Update history with current trend_seasonal timestep for next iteration
                    actual_value = seasonal + trend
                    sarima_model.append([actual_value], refit=False)


                # Seconda fase: quando i residui di test sono sufficienti
                for i in tqdm(range(self.input_len, len(self.test)), desc="Forecasting: Using predicted STL components"):

                    # Previsione del componente stagionale e decomposizione del nuovo punto dati
                    trend, seasonal, residual = online_stl.update(self.test[self.target_column].iloc[i])
                    test_residuals.iloc[i] = residual

                    # Usa solo i residui di test recenti
                    recent_residuals = test_residuals.iloc[i - self.input_len + 1: i + 1]

                    # Converti i residui recenti in un DataFrame
                    last_window_df = recent_residuals.to_frame(name=self.target_column)

                    # Previsione del componente residuale con LSTM per un passo
                    lstm_pred = lstm_forecaster.predict(steps=1, last_window=last_window_df)

                    # Forecast seasonal trend component with SARIMA for one step
                    sarima_pred = sarima_model.predict(steps=1)

                    # Combine predictions
                    combined_pred = sarima_pred.iloc[0, 0] + lstm_pred.iloc[0, 0]

                    # Append combined prediction
                    predictions.append(combined_pred)

                    # Update history with current trend_seasonal timestep for next iteration
                    actual_value = seasonal + trend
                    sarima_model.append([actual_value], refit=False)

            prediction_index = self.test.index
            predictions_df = pd.DataFrame({self.target_column: predictions}, index=prediction_index)


            # fit scaler on train data to later scale test and predictions
            scaler = MinMaxScaler()
            temp_train = self.train.applymap(lambda x: x.replace(',', '.') if isinstance(x, str) else x)
            scaler.fit(temp_train[temp_train.columns[0:temp_train.columns.shape[0] - 1]])

            return sarima_model, predictions_df, scaler

        
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
        

    def plot_predictions(self, predictions, test):
        """
        Plots the SARIMA model predictions against the test data.

        :param predictions: The predictions made by the SARIMA model
        """
        test = test[self.target_column]
        plt.plot(test.index, test, 'b-', label='Test Set')
        plt.plot(test.index, predictions, 'k--', label='ARIMA')
        plt.title(f'SARIMA prediction for feature: {self.target_column}')
        plt.xlabel('Time series index')
        plt.legend(loc='best')
        plt.tight_layout()
        plt.show()


    
