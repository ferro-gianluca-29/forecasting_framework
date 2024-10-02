import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from keras.optimizers import Adam
from keras.losses import MeanSquaredError
from keras.callbacks import EarlyStopping

from xgboost import XGBRegressor


from skforecast.ForecasterRnn import ForecasterRnn
from skforecast.ForecasterRnn.utils import create_and_compile_model
from skforecast.model_selection_multiseries import backtesting_forecaster_multiseries

from sklearn.linear_model import Ridge
from sklearn.ensemble  import StackingRegressor
from sklearn.model_selection  import KFold

from skforecast.ForecasterAutoreg import ForecasterAutoreg
from skforecast.model_selection import backtesting_forecaster
from skforecast.model_selection import grid_search_forecaster


from scikeras.wrappers import KerasRegressor

from tscv import GapKFold


from tqdm import tqdm
import pickle
from Predictors.Predictor import Predictor

class ENSEMBLE_Predictor(Predictor):
    """
    A class used to predict time series data using the ENSEMBLE model.
    """

    def __init__(self, run_mode, target_column=None, 
                 verbose=False,  seasonal_model=False, input_len = None, output_len= 24, forecast_type= None, set_fourier=False):
  

        super().__init__(verbose=verbose)  

        self.run_mode = run_mode
        self.verbose = verbose
        self.target_column = target_column
        

    def train_model(self, input_len, output_len):
    
        try:

            lstm_model = create_and_compile_model(
                        series = self.train[[self.target_column]], # Series used as predictors
                        levels = self.target_column,                         # Target column to predict
                        lags = input_len,
                        steps = output_len,
                        recurrent_layer = "LSTM",
                        activation = "tanh",
                        recurrent_units = [40,40,40],
                        optimizer = Adam(learning_rate=0.01), 
                        loss = MeanSquaredError()
                                            )
            
            lstm_regressor = KerasRegressor(
                        model=lstm_model,
                        epochs=5,  # Adjust as needed
                        batch_size=400,  # Adjust as needed
                        verbose=1
                    )
            
            print("LSTM model summary:")
            lstm_model.summary()
            
            
            xgb_model = XGBRegressor(
            n_estimators=1000,  # Number of boosting rounds (you can tune this)
            learning_rate=0.05,   # Learning rate (you can tune this)
            max_depth=5,          # Maximum depth of the trees (you can tune this)
            min_child_weight=1,   # Minimum sum of instance weight needed in a child
            gamma=0,              # Minimum loss reduction required to make a further partition
            subsample=0.8,        # Fraction of samples used for training
            colsample_bytree=0.8, # Fraction of features used for training
            reg_alpha=0,          # L1 regularization term on weights
            reg_lambda=1,         # L2 regularization term on weights
            objective='reg:squarederror',  # Objective function for regression
            random_state=42,       # Seed for reproducibility
            eval_metric=['rmse', 'mae'],
            transformer_y = None,

            # use this two lines to enable GPU
            #tree_method  = 'hist',
            #device       = 'cuda',
                                    )
            
            estimators = [
                ('lstm', lstm_regressor),
                ('xgb', xgb_model),
            ]

            full_data = pd.concat([self.train, self.valid, self.test])

            



            stacking_regressor = StackingRegressor(
                                    estimators = estimators,
                                    final_estimator = Ridge(),
                                    cv = GapKFold(n_splits=5, gap_before=0, gap_after=0)
                                )
            
            forecaster = ForecasterAutoreg(
                 regressor = stacking_regressor,
                 lags = input_len
             )
            
            # Grid search of hyperparameters
            param_grid = {
                'final_estimator__alpha': [0.001, 0.01, 0.1, 1, 10],
            }

            # Lags used as predictors
            lags_grid = [24]

            results_grid = grid_search_forecaster(
                            forecaster         = forecaster,
                            y                  = full_data[self.target_column][:len(self.train) + len(self.valid)],
                            param_grid         = param_grid,
                            lags_grid          = lags_grid,
                            steps              = output_len,
                            refit              = False,
                            metric             = 'mean_squared_error',
                            initial_train_size = len(self.train),
                            fixed_train_size   = True,
                            return_best        = True,
                            n_jobs             = 'auto',
                            verbose            = False
                        )

            results_grid.head()
            
            

            _, predictions = backtesting_forecaster(
                            forecaster         = forecaster,
                            y                  = full_data[self.target_column],
                            #exog               = ,
                            initial_train_size = len(self.train) + len(self.valid),
                            steps              = output_len,
                            refit              = False,
                            metric             = 'mean_squared_error',
                            n_jobs             = 'auto',
                            verbose            = True, # Change to False to see less information
                            show_progress      = True
                      )        

            
            predictions.rename(columns={'pred': self.target_column}, inplace=True)    
            return forecaster, predictions
 
        except Exception as e:
            print(f"An error occurred during the model training: {e}")
            return None
        
        
    