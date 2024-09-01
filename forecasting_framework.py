#  LIBRARY IMPORTS

import argparse
import pandas as pd
<<<<<<< Updated upstream
<<<<<<< Updated upstream:main_code.py
from matplotlib import pyplot as plt
from classes.data_preprocessing import DataPreprocessor
from classes.data_loader import DataLoader
from classes.training_module import ModelTraining
from classes.model_testing import ModelTest
from classes.performance_measurement import PerfMeasure
import datetime
from utils.utilities import ts_analysis, save_data, save_buffer, load_trained_model
from utils.time_series_analysis import multiple_STL
=======
=======
>>>>>>> Stashed changes
import numpy as np
import datetime
import pickle

from keras.models import load_model
import xgboost as xgb
from xgboost import plot_importance
from statsmodels.tsa.deterministic import Fourier

from tools.data_preprocessing import DataPreprocessor
from tools.data_loader import DataLoader
from tools.performance_measurement import PerfMeasure
from tools.utilities import save_data, save_buffer, load_trained_model
from tools.time_series_analysis import time_s_analysis, multiple_STL, prepare_seasonal_sets


from Predictors.LSTM_model import LSTM_Predictor
from Predictors.XGB_model import XGB_Predictor
from Predictors.ARIMA_model import ARIMA_Predictor
from Predictors.SARIMA_model import SARIMA_Predictor
from Predictors.NAIVE_model import NAIVE_Predictor


<<<<<<< Updated upstream
>>>>>>> Stashed changes:forecasting_framework.py
=======
>>>>>>> Stashed changes

# END OF LIBRARY IMPORTS #
  
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import warnings
warnings.filterwarnings('ignore') 

def main():
<<<<<<< Updated upstream
    
    # ARGUMENT PARSING
=======
    """
    Main function to execute time series forecasting tasks based on user-specified arguments.
    This function handles the entire workflow from data loading, preprocessing, model training, testing,
    and evaluation, based on the configuration provided via command-line arguments.
    """
    
    # ARGUMENT PARSING
    """
    Parsing of command-line arguments to set up the environment and specify model training, testing, and evaluation parameters.
    """
>>>>>>> Stashed changes
    parser = argparse.ArgumentParser(description='Time series forecasting')

    # General arguments
    parser.add_argument('--verbose', action='store_true', required=False, default=False, help='If specified, minimizes the additional information provided during the program launch')
    parser.add_argument('--ts_analysis', action='store_true', required=False, default=False, help='If True, performs an analysis on the time series')
    parser.add_argument('--run_mode', type=str, required=True, help='Running mode (training, testing, both, or fine tuning)')

    # Dataset arguments
    parser.add_argument('--dataset_path', type=str, required=True, help='Dataset path')
<<<<<<< Updated upstream
    parser.add_argument('--seasonal_split', action='store_true', required=False, default=False, help='If True, makes a split that takes into account seasonality')
=======
    parser.add_argument('--date_format', type=str, required=True, help='Format of date time')
    parser.add_argument('--date_list', type=str, nargs='+', help='List with start and end of dates for training, validation and test set')
>>>>>>> Stashed changes
    parser.add_argument('--train_size', type=float, required=False, default=0.7, help='Training set size')
    parser.add_argument('--val_size', type=float, required=False, default=0.2, help='Validation set size')
    parser.add_argument('--test_size', type=float, required=False, default=0.1, help='Test set size')
    parser.add_argument('--scaling', action='store_true', help='If True, data will be scaled')
<<<<<<< Updated upstream
    parser.add_argument('--validation', action='store_true', required=False, help='If True, the validation set is created (not for ARIMA-SARIMAX)' )
=======
    parser.add_argument('--validation', action='store_true', required=False, help='If True, the validation set is created' )
>>>>>>> Stashed changes
    parser.add_argument('--target_column', type=str, required=True, help='Name of the target column for forecasting')
    parser.add_argument('--time_column_index', type=int, required=False, default=0, help='Index of the column containing the timestamps')

    # Model arguments
    parser.add_argument('--model_type', type=str, required=True, help='Type of model to use (ARIMA, SARIMAX, PROPHET, CONV, LSTM, CNN_LSTM)')
<<<<<<< Updated upstream
    parser.add_argument('--forecast_type', type=str, required=False, help='Type of forecast: ol-multi= open-loop multi step ahead; ol-one= open loop one step ahead, cl-multi= closed-loop multi step ahead. Not necessary for PROPHET')
    parser.add_argument('--steps_ahead', type=int, required=False, default=100, help='Number of time steps ahead to forecast')
    parser.add_argument('--steps_jump', type=int, required=False, default=50, help='Number of steps to skip')
    parser.add_argument('--exog', nargs='+', type=str, required=False, default = None, help='Exogenous columns for the SARIMAX model')
    parser.add_argument('--period', type=int, required=False, default=24, help='Seasonality period for the SARIMAX model')    

    # Fine tuning arguments    
    parser.add_argument('--model_path', type=str, required=False, default=None, help='Path of the pre-trained model' )    
    parser.add_argument('--ol_refit', action='store_true', required=False, default=False, help='For ARIMA and SARIMAX models: If specified, in OL forecasts the model is retrained for each added observation ')
       
=======
    # Statistical models
    parser.add_argument('--forecast_type', type=str, required=False, help='Type of forecast: ol-multi= open-loop multi step ahead; ol-one= open loop one step ahead, cl-multi= closed-loop multi step ahead. Not necessary for PROPHET')
    parser.add_argument('--valid_steps', type=int, required=False, default=10, help='Number of time steps to use during validation')
    parser.add_argument('--steps_jump', type=int, required=False, default=50, help='Number of steps to skip in open loop multi step predictions')
    parser.add_argument('--exog', nargs='+', type=str, required=False, default = None, help='Exogenous columns for the SARIMAX model')
    parser.add_argument('--period', type=int, required=False, default=24, help='Seasonality period')  
    parser.add_argument('--set_fourier', action='store_true', required=False, default=False, help='If True, Fourier exogenous variables are used')
    # Other models
    parser.add_argument('--seasonal_model', action='store_true', help='If True, in the case of LSTM the seasonal component is fed into the model, while for XGB models Fourier features are added')
    parser.add_argument('--input_len', type=int, required=False, default=24, help='Number of timesteps to use for prediction in each window in LSTM')
    parser.add_argument('--output_len', type=int, required=False, default=1, help='Number of timesteps to predict in each window in LSTM')
    
    #parser.add_argument('--seq_len', type=int, required=False, default=10, help='Input sequence length for predictions')

    # Test and fine tuning arguments    
    parser.add_argument('--model_path', type=str, required=False, default=None, help='Path of the pre-trained model' )    
    parser.add_argument('--ol_refit', action='store_true', required=False, default=False, help='For ARIMA and SARIMAX models: If specified, in OL forecasts the model is retrained for each added observation ')
    parser.add_argument('--unscale_predictions', action='store_true', required=False, default=False, help=' If specified, predictions and test data are unscaled')
    
>>>>>>> Stashed changes
    args = parser.parse_args()
    # END OF ARGUMENT PARSING

    verbose = args.verbose 

    try:
        
        # Create current model folder
        folder_name = args.model_type + "_" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        folder_path = f"./data/models/{folder_name}"
        os.makedirs(folder_path)
<<<<<<< Updated upstream
 
        #  DATA LOADING
        data_loader = DataLoader(args.dataset_path, args.target_column, args.time_column_index)
        df = data_loader.load_data()
        if df is None:
            raise ValueError("Unable to load dataset.")
        
        # END OF DATA LOADING
        
                
            

#################### PREPROCESSING  ####################
=======

        #######  DATA LOADING

        data_loader = DataLoader(args.dataset_path, args.date_format, args.model_type, args.target_column, args.time_column_index, args.date_list, args.exog)
        df, dates = data_loader.load_data()
        if df is None:
            raise ValueError("Unable to load dataset.")
        
        ####### END OF DATA LOADING
        
        ####### PREPROCESSING AND DATASET SPLIT  ########
>>>>>>> Stashed changes
        
        # Extract the file extension from the path 
        file_ext = os.path.splitext(args.dataset_path)[1]
        
<<<<<<< Updated upstream
        data_preprocessor = DataPreprocessor(file_ext, args.run_mode, args.model_type, df, args.target_column, 
                                             args.scaling, args.validation, args.train_size, args.val_size, args.test_size, args.seasonal_split,
                                             folder_path, args.model_path, verbose)
        
<<<<<<< Updated upstream:main_code.py
        # preprocessing and split for ARIMA, SARIMAX
        if args.run_mode == "test":
            test, exit = data_preprocessor.preprocess_data()
        else:
            if args.validation:
                train, test, valid, exit = data_preprocessor.preprocess_data()
                if exit:
                    raise ValueError("Unable to preprocess dataset.")
            else:
                train, test, exit = data_preprocessor.preprocess_data()
                valid = None
                if exit:
                    raise ValueError("Unable to preprocess dataset.")
        
        # Splitting target and exogenous variable in training and test sets for the SARIMAX model
        if args.model_type == 'SARIMAX':
            target_train = train[[args.target_column]]
            exog_train = train[args.exog]
            target_valid = valid[[args.target_column]]
            exog_valid = valid[args.exog]
            target_test = test[[args.target_column]]
            exog_test = test[args.exog]

#################### FINE PREPROCESSING E SPLIT DEL DATSET ####################
        
        ############### Optional time series analysis ############
        if args.ts_analysis:
            ts_analysis(df, args.target_column, args.period)
            multiple_STL(train, args.target_column)

        ############## End of time series analysis ###########

######### PRINT INFO
        if verbose:
            print('\nThe preprocess_data function returned the following sets:\n')
            print('Train Set:\n')
            print(train.head())
            print('\nTrain set size: ', len(train))
            print('Test Set:\n')
            print(test.head())
            print('\nTest set size: ', len(test))
            if args.validation:
                print('Validation Set:\n')
                print(valid.head())
                print('\nValidation set size: ', len(valid))
            # Plot the series
            if args.model_type == 'PROPHET':
                plt.figure(figsize=(15, 5))
                plt.plot(df['ds'], df['y'])
                plt.title('Time series')
                plt.show()
            else:
                plt.figure(figsize=(15, 5))
                plt.plot(train[train.columns[0]], train[args.target_column])
                plt.xlabel(train.columns[0])
                plt.ylabel(args.target_column)
                plt.title('Training set')
                plt.show()
=======
=======
        data_preprocessor = DataPreprocessor(file_ext, args.run_mode, args.model_type, df, args.target_column, dates, 
                                             args.scaling, args.validation, args.train_size, args.val_size, args.test_size, 
                                             folder_path, args.model_path, verbose)

        # Preprocessing and split 

        ############### Optional time series analysis ############
        
>>>>>>> Stashed changes
        if args.ts_analysis:
            time_s_analysis(df, args.target_column, args.period)
            train, test, exit = data_preprocessor.preprocess_data()
            
            multiple_STL(train, args.target_column)
            return 0
            
        ############## End of time series analysis ###########


        #### Model Selection ####

        match args.model_type:

            case 'ARIMA':
                arima = ARIMA_Predictor(args.run_mode, args.target_column, 
                args.verbose, args.set_fourier)

            case 'SARIMA':
                sarima = SARIMA_Predictor(args.run_mode, args.target_column, args.period,
                args.verbose, args.set_fourier)

            case 'LSTM':
                lstm = LSTM_Predictor(args.run_mode, args.target_column, 
                args.verbose, args.input_len, args.output_len, args.seasonal_model, args.set_fourier)

            case 'XGB':
                xgb = XGB_Predictor(args.run_mode, args.target_column, 
                args.verbose, args.seasonal_model, args.set_fourier)

            case 'NAIVE':
                naive = NAIVE_Predictor(args.run_mode, args.target_column,
                args.verbose)

            
        #### End of model selection ####


        ### Preprocessing for test-only mode
        if args.run_mode == "test":

            test, exit = data_preprocessor.preprocess_data()

            match args.model_type:

                case 'ARIMA':
                    arima.prepare_data(test = test)
                    

                case 'SARIMA'|'SARIMAX':
                    
                    # Create the test set for target and exog variables
                    target_test = test[[args.target_column]]
                    if args.exog is not None:
                        exog_test = test[args.exog]
                    else: exog_test = None
                    sarima.prepare_data(test = target_test)
                
                case 'LSTM':
                    train = []
                    valid = []
                    lstm.prepare_data(train, valid, test)
                    X_test, y_test = lstm.data_windowing()

                case 'XGB':
                    train = []
                    valid = []
                    xgb.prepare_data(train, valid, test)
                    X_test, y_test = data_preprocessor.create_time_features(test)
                


            ### End of preprocessing for test-only mode

        else:

            ####### PREPROCESSING AND DATASET SPLIT ######
            match args.model_type:

                case 'ARIMA':
                    if args.validation:
                        train, test, valid, exit = data_preprocessor.preprocess_data()
                        if exit:
                            raise ValueError("Unable to preprocess dataset.")

                    else:
                        train, test, exit = data_preprocessor.preprocess_data()
                        valid = None      
                        if exit:
                            raise ValueError("Unable to preprocess dataset.")
                        
                    arima.prepare_data(train, valid, test)

                case 'SARIMA'|'SARIMAX':
                    # Set the exogenous variable column
                    if args.exog is None:
                        exog = args.target_column
                    else: 
                        exog = args.exog

                    if args.validation:
                        train, test, valid, exit = data_preprocessor.preprocess_data()
                        if exit:
                            raise ValueError("Unable to preprocess dataset.")
                        target_valid = valid[[args.target_column]]
                        exog_valid = valid[exog]
                    else:
                        train, test, exit = data_preprocessor.preprocess_data()
                        if exit:
                            raise ValueError("Unable to preprocess dataset.")
                        valid = None   
                        exog_valid = None   
                            
                    target_train = train[[args.target_column]]
                    exog_train = train[exog]

                    if args.run_mode == 'train_test':    
                        target_test = test[[args.target_column]]
                        exog_test = test[exog]
                    else: target_test = None

                    sarima.prepare_data(target_train, valid, target_test)

                case 'LSTM':
                    train, test, valid, exit = data_preprocessor.preprocess_data()
                    
                    if exit:
                        raise ValueError("Unable to preprocess dataset.")
                    
                    lstm.prepare_data(train, valid, test)
                    
                    if args.seasonal_model:
                        
                        """ Aggiunta feature di Fourier come nel caso XGB
                        for df in [train, test, valid]:
                            # Add Fourier features for daily, weekly, and yearly seasonality
                            for period in [24, 7, 365]:
                                df[f'sin_{period}'] = np.sin(df.index.dayofyear / period * 2 * np.pi)
                                df[f'cos_{period}'] = np.cos(df.index.dayofyear / period * 2 * np.pi)
                        """

                        if args.set_fourier:
                            # Aggiunta feature di Fourier con Statsmodels

                            K = 5
                            fourier = Fourier(period=args.period, order=K)
                            train_fourier_terms = fourier.in_sample(train.index)
                            valid_fourier_terms = fourier.in_sample(valid.index)
                            test_fourier_terms = fourier.in_sample(test.index)

                            train[train_fourier_terms.columns] = train_fourier_terms
                            valid[valid_fourier_terms.columns] = valid_fourier_terms
                            test[test_fourier_terms.columns] = test_fourier_terms

                            X_train, y_train, X_valid, y_valid, X_test, y_test = lstm.data_windowing()
                        
                        else:

                            # LSTM stagionale con decomposizione STL
                            train_decomposed, train_decomposed, test_decomposed =  prepare_seasonal_sets(train, valid, test, args.target_column, args.period)
                            lstm.prepare_data(train_decomposed, train_decomposed, test_decomposed)

                            X_train, y_train, X_valid, y_valid, X_test, y_test = lstm.data_windowing()
                        
                    else:
                        X_train, y_train, X_valid, y_valid, X_test, y_test = lstm.data_windowing()

                case 'XGB':
                    train, test, valid, exit = data_preprocessor.preprocess_data()

                    if exit:
                        raise ValueError("Unable to preprocess dataset.")
                    
                    xgb.prepare_data(train, valid, test)
                    X_train, y_train = xgb.create_time_features(train)
                    X_valid, y_valid = xgb.create_time_features(valid)
                    X_test, y_test = xgb.create_time_features(test)

                case 'NAIVE':
                    train, test, exit = data_preprocessor.preprocess_data()
                    valid = None
                    model = None      
                    if exit:
                        raise ValueError("Unable to preprocess dataset.")
                    naive.prepare_data(train, None, test)

            print(f"Training set dim: {train.shape[0]} \n")
            if args.run_mode == 'train_test': print(f"Test set dim: {test.shape[0]}")

        ########### END OF PREPROCESSING AND DATASET SPLIT ########

<<<<<<< Updated upstream
>>>>>>> Stashed changes:forecasting_framework.py
        
            if args.model_type == 'SARIMAX':
                print('\nTarget Train Set:')
                print(target_train.head())
                print('Target training set size: ', len(target_train))
                print('\nExog Train Set:')
                print(exog_train.head())
                print('Exog training set size: ', len(exog_train))
                print('\nTarget Test Set:')
                print(target_test.head())
                print('Target test set size: ', len(target_test))
                print('\nExog Test Set:')
                print(exog_test.head())
                print('Exog test set size: ', len(exog_test))

########## END OF PRINT INFO

        if args.run_mode == "fine_tuning" or args.run_mode == "test":

            #################### LOAD MODEL FOR TEST OR FINE TUNING ####################
            # NOTE: Using the append() method of statsmodels, the indices for fine tuning must be contiguous to those of the pre-trained model
            if (args.model_type == 'ARIMA'):

                # Load a pre-trained model
                pre_trained_model, prev_train_end_index, best_order = load_trained_model(args.model_type, args.model_path)
                
                # Update the indices so that the the indices are contiguous to those of the pre-trained model
                test_start_index = test.index[0] + prev_train_end_index
                test_end_index = test_start_index + len(test)
                test.index = range(test_start_index, test_end_index)
                
                if args.run_mode == "fine_tuning":
                     train.index = range(prev_train_end_index, prev_train_end_index + len(train))  
                     model = pre_trained_model.append(train[args.target_column], refit = True)          
                elif args.run_mode == "test":
                    # Load the model 
                    model = pre_trained_model   
 
            elif (args.model_type == 'SARIMAX'):

                # Load a pre-trained model
                pre_trained_model, prev_train_end_index, best_order = load_trained_model(args.model_type, args.model_path) 

                # Update the indices so that the the indices are contiguous to those of the pre-trained model
                
                test_start_index = test.index[0] + prev_train_end_index
                test_end_index = test_start_index + len(test)
                test.index = range(test_start_index, test_end_index)
                target_test.index = range(test_start_index, test_end_index)
                exog_test.index = range(test_start_index, test_end_index)
                    
<<<<<<< Updated upstream:main_code.py
                if args.run_mode == "fine_tuning":  
                    # Update the model with the new data
                    train.index = range(prev_train_end_index, prev_train_end_index + len(train)) 
                    exog_train.index = range(prev_train_end_index, prev_train_end_index + len(train))  
                    model = pre_trained_model.append(train[args.target_column], exog =  exog_train, refit = True)
                elif args.run_mode == "test":
                    # Load the model 
                    model = pre_trained_model
=======
=======
        
        if args.run_mode == "fine_tuning" or args.run_mode == "test":

            #################### MODEL LOADING FOR TEST OR FINE TUNING ####################
 
            # NOTE: Using the append() method of statsmodels, the indices for fine tuning must be contiguous to those of the pre-trained model
                
            match args.model_type:

                    case 'ARIMA':

                        # Load a pre-trained model
                        pre_trained_model, best_order = load_trained_model(args.model_type, args.model_path)
                        last_train_index = pre_trained_model.data.row_labels[-1] + 1
                        train_start_index = last_train_index

                        # Update the indices so that they are contiguous to those of the pre-trained model

                        test_start_index = test.index[0] + last_train_index
                        test_end_index = test_start_index + len(test)
                        test.index = range(test_start_index, test_end_index)

                        # Create last_index entry for testing function
                        last_index = test_start_index
                        
                        if args.run_mode == "fine_tuning":
                            
                            train.index = range(train_start_index, train_start_index + len(train))  
                            model = pre_trained_model.append(train[args.target_column], refit = True)          
                        elif args.run_mode == "test":
                            # Load the model 
                            model = pre_trained_model   
 
                    case 'SARIMAX'|'SARIMA': 

                        # Load a pre-trained model
                        pre_trained_model, best_order = load_trained_model(args.model_type, args.model_path)
                        last_train_index = pre_trained_model.data.row_labels[-1] + 1
                        train_start_index = last_train_index

                        # Update the indices so that the the indices are contiguous to those of the pre-trained model
                        
                        test_start_index = test.index[0] + last_train_index
                        test_end_index = test_start_index + len(test)
                        test.index = range(test_start_index, test_end_index)
                        target_test.index = range(test_start_index, test_end_index)
                        # Create last_index entry for testing function
                        last_index = test_start_index

                        if args.model_type == 'SARIMAX':
                            exog_test.index = range(test_start_index, test_end_index)
                            
                        if args.run_mode == "fine_tuning":  
                            # Update the model with the new data
                            train.index = range(train_start_index, train_start_index + len(train)) 
                            exog_train.index = range(train_start_index, train_start_index + len(train))  
                            if args.model_type == 'SARIMA':
                                model = pre_trained_model.append(train[args.target_column], refit = True)
                            elif args.model_type == 'SARIMAX':
                                model = pre_trained_model.append(train[args.target_column], exog = exog_train, refit = True)
                        elif args.run_mode == "test":
                            # Load the model 
                            model = pre_trained_model
                    
>>>>>>> Stashed changes
                    case 'LSTM':
                        model = load_model(f"{args.model_path}/model.h5")
                        if args.run_mode == 'fine_tuning':
                            history = model.fit(X_train, y_train, epochs=200, validation_data=(X_valid, y_valid),batch_size=1000)
                            valid_metrics = {}
                            valid_metrics['valid_loss'] = history.history['val_loss']
                            valid_metrics['valid_mae'] = history.history['val_mean_absolute_error']
                            valid_metrics['valid_mape'] = history.history['val_mean_absolute_percentage_error']
<<<<<<< Updated upstream
>>>>>>> Stashed changes:forecasting_framework.py

            ######################## # END OF LOAD MODEL ####################
=======

                            # Save training data
                            save_data("training", args.validation, folder_path, args.model_type, model, args.dataset_path, 
                                end_index = len(train),  valid_metrics = valid_metrics)

                    case 'XGB':
                        model = xgb.XGBRegressor()
                        model.load_model(f"{args.model_path}/model.json")
                        if args.run_mode == 'fine_tuning':
                            model = model.fit(
                            X_train,
                            y_train,
                            eval_set=[(X_train, y_train), (X_valid, y_valid)],
                            eval_metric=['rmse', 'mae'],
                            early_stopping_rounds=100,
                            verbose=False  # Set to True to see training progress
                        )
                            valid_metrics = model.evals_result()
                            
                            # Save training data
                            save_data("training", args.validation, folder_path, args.model_type, model, args.dataset_path, 
                                    end_index = len(train),  valid_metrics = valid_metrics)

            ######################## # END OF MODEL LOADING AND FINE TUNING ####################
>>>>>>> Stashed changes
    
        if args.run_mode == "train" or args.run_mode == "train_test":

                #################### MODEL TRAINING ####################
<<<<<<< Updated upstream
                model_training = ModelTraining(args.model_type, train, valid, args.target_column, verbose = False)

<<<<<<< Updated upstream:main_code.py
                if (args.model_type == 'ARIMA'):    
                    model, valid_metrics = model_training.train_ARIMA_model()
                    best_order = model_training.ARIMA_order
                    # Save a buffer containing the last elements of the training set for further test
                    buffer_size = 20
                    save_buffer(folder_path, train, args.target_column, size = buffer_size, file_name = 'buffer.json')
                    # Save training data 
                    save_data("training", folder_path, args.model_type, model, args.dataset_path, 
                              best_order = best_order, end_index = len(train), valid_metrics = valid_metrics)

                elif (args.model_type == 'SARIMAX'):    
                    model, valid_metrics = model_training.train_SARIMAX_model(target_train, exog_train, exog_valid, args.period)
                    best_order = model_training.SARIMAX_order
                    # Save a buffer containing the last elements of the training set for further test
                    buffer_size = 20
                    save_buffer(folder_path, train, args.target_column, size = buffer_size, file_name = 'buffer.json')
                    # Save training data
                    save_data("training", folder_path, args.model_type, model, args.dataset_path, 
                              best_order = best_order, end_index = len(train),  valid_metrics = valid_metrics)
=======
=======

>>>>>>> Stashed changes
                match args.model_type:

                    case 'ARIMA':    
                        model, valid_metrics, last_index = arima.train_model()
                        best_order = arima.ARIMA_order
                        if args.run_mode == 'train_test':
                            # Save a buffer containing the last elements of the training set for further test
                            buffer_size = test.shape[0]
                            save_buffer(folder_path, train, args.target_column, size = buffer_size, file_name = 'buffer.json')
                        # Save training data 
                        save_data("training", args.validation, folder_path, args.model_type, model, args.dataset_path, 
                                best_order = best_order, end_index = model.data.row_labels[-1] + 1, valid_metrics = valid_metrics)

                    case 'SARIMAX'|'SARIMA':   
                        model, valid_metrics, last_index  = sarima.train_model()
                        best_order = sarima.SARIMA_order
                        if args.run_mode == 'train_test':
                            # Save a buffer containing the last elements of the training set for further test
                            buffer_size = test.shape[0]
                            save_buffer(folder_path, train, args.target_column, size = buffer_size, file_name = 'buffer.json')
                        # Save training data
                        save_data("training", args.validation, folder_path, args.model_type, model, args.dataset_path, 
                                best_order = best_order, end_index = model.data.row_labels[-1] + 1,  valid_metrics = valid_metrics)
                    
                    case 'LSTM':
                        model, valid_metrics = lstm.train_model(X_train, y_train, X_valid, y_valid)
                        # Save training data
                        save_data("training", args.validation, folder_path, args.model_type, model, args.dataset_path, 
                                end_index = len(train),  valid_metrics = valid_metrics)
                    
                    case 'XGB':
                        model, valid_metrics = xgb.train_model(X_train, y_train, X_valid, y_valid)
                        # Save training data
                        save_data("training", args.validation, folder_path, args.model_type, model, args.dataset_path, 
                                end_index = len(train),  valid_metrics = valid_metrics)
<<<<<<< Updated upstream
>>>>>>> Stashed changes:forecasting_framework.py
=======
>>>>>>> Stashed changes

                #################### END OF MODEL TRAINING ####################
                    
        if args.run_mode == "train_test" or args.run_mode == "fine_tuning" or args.run_mode == "test":
<<<<<<< Updated upstream
                
            if args.run_mode == "test":

                # Load buffer from JSON file
                train = pd.read_json(f"{args.model_path}/buffer.json", orient='records') 

            #################### MODEL TESTING ####################

<<<<<<< Updated upstream:main_code.py
            model_test = ModelTest(model, test, args.target_column, args.forecast_type, args.steps_ahead)
            
            if(args.model_type == 'ARIMA'):
                # Model testing
                predictions = model_test.test_ARIMA_model(args.steps_jump, args.ol_refit)    
                # Create the naive model
                naive_predictions = model_test.naive_forecast(train)
                
            elif(args.model_type == 'SARIMAX'):
                # Model testing
                predictions = model_test.test_SARIMAX_model(args.steps_jump, exog_test, args.ol_refit)   
                # Create the naive model
                naive_predictions = model_test.naive_seasonal_forecast(target_train, target_test, args.period)
            
=======
=======


            ##### Manage buffer for statistical models    
            if args.run_mode == "test" and args.model_type in ['ARIMA','SARIMA','SARIMAX']:
                # Create a training set from the loaded buffer, that will be used for the naive models
                # for ARIMA: train; for SARIMAX: target_train
                # Load buffer from JSON file
                train = pd.DataFrame()
                train[args.target_column] = pd.read_json(f"{args.model_path}/buffer.json", orient='records') 
                target_train = pd.DataFrame()
                target_train[args.target_column] = pd.read_json(f"{args.model_path}/buffer.json", orient='records')
                print(f"Training set buffer dim: {train.shape[0]} \n")
                print(f"Test set dim: {test.shape[0]}")
            ##### End of manage buffer
           
           
            #################### MODEL TESTING ####################

>>>>>>> Stashed changes
        
            match args.model_type:

                case 'ARIMA':
                    # Model testing
                    predictions = arima.test_model(model, last_index, args.forecast_type, args.ol_refit)    

                    if args.unscale_predictions:

                        if args.run_mode == 'train_test':
                            path = folder_path
                        else:
                            path = args.model_path
                        arima.unscale_predictions(predictions, path)

                        # Unscale test data
                        # Load scaler for unscaling test data
                        with open(f"{path}/scaler.pkl", "rb") as file:
                            scaler = pickle.load(file)
                        test[args.target_column] = scaler.inverse_transform(test[[args.target_column]])
                    
                    predictions.to_csv('raw_data.csv', index = False)

                case 'SARIMAX'|'SARIMA':
                    last_index = model.data.row_labels[-1] + 1
                    # Model testing
                    predictions = sarima.test_model(model, last_index, args.forecast_type, args.ol_refit, args.period, args.set_fourier)   

                    if args.unscale_predictions:

                        if args.run_mode == 'train_test':
                            path = folder_path
                        else:
                            path = args.model_path
                        sarima.unscale_predictions(predictions, path)

                        # Unscale test data
                        # Load scaler for unscaling test data
                        with open(f"{path}/scaler.pkl", "rb") as file:
                            scaler = pickle.load(file)
                        
                        test[args.target_column] = scaler.inverse_transform(test[[args.target_column]])
                        target_test[args.target_column] = scaler.inverse_transform(target_test[[args.target_column]])

                    predictions.to_csv('raw_data.csv', index = False)

                case 'LSTM':
                    # Model testing
                    predictions = model.predict(X_test)

                    if args.unscale_predictions:
                        lstm.unscale_data(predictions, y_test, folder_path)
                    pd.Series(predictions.flatten()).to_csv('raw_data.csv', index = False)


                case 'XGB':
                    # Model testing
                    predictions = model.predict(X_test)
                    
                    if args.unscale_predictions:
                        xgb.unscale_data(predictions, y_test, folder_path)
                    pd.Series(predictions.flatten()).to_csv('raw_data.csv', index = False)

                        
                case 'NAIVE':
                    if args.seasonal_model:
                        predictions = naive.seasonal_forecast(args.period)
                    else:
                        predictions = naive.forecast(args.forecast_type)
                        
                    if args.unscale_predictions:

                        # Unscale predictions

                        naive.unscale_predictions(predictions, folder_path)

                        # Unscale test data
                        # Load scaler for unscaling test data
                        with open(f"{folder_path}/scaler.pkl", "rb") as file:
                            scaler = pickle.load(file)
                        
                        test[args.target_column] = scaler.inverse_transform(test[[args.target_column]])

                    predictions.to_csv('raw_data.csv', index = False)

                        
                    
<<<<<<< Updated upstream
>>>>>>> Stashed changes:forecasting_framework.py
=======
>>>>>>> Stashed changes
            #################### END OF MODEL TESTING ####################        
        
        if args.run_mode != "train":

            #################### PLOT PREDICTIONS ####################

<<<<<<< Updated upstream
<<<<<<< Updated upstream:main_code.py
            if(args.model_type == 'ARIMA'):
                model_test.ARIMA_plot_pred(best_order, predictions, naive_predictions)
            
            elif(args.model_type == 'SARIMAX'):
                model_test.SARIMAX_plot_pred(best_order, predictions, naive_predictions)
=======
=======
>>>>>>> Stashed changes
            match args.model_type:

                case 'ARIMA':
                    arima.plot_predictions(predictions)
            
                case 'SARIMAX'|'SARIMA':
                    sarima.plot_predictions(predictions)

                case 'LSTM':
                    if args.output_len != 1: lstm.plot_predictions(predictions, y_test)

                case 'XGB':
                    _ = plot_importance(model, height=0.9) 
                    time_values = y_test.index   
                    xgb.plot_predictions(predictions, y_test, time_values)

                case 'NAIVE':
                    naive.plot_predictions(predictions)     
<<<<<<< Updated upstream
>>>>>>> Stashed changes:forecasting_framework.py
=======
>>>>>>> Stashed changes

            #################### END OF PLOT PREDICTIONS ####################        
         
            #################### PERFORMANCE MEASUREMENT AND SAVING #################
<<<<<<< Updated upstream
            if predictions is not None:
                perf_measure = PerfMeasure(model, test, args.target_column, args.forecast_type, args.steps_ahead)
                if(args.model_type == 'ARIMA'):
                    # Compute performance metrics
                    metrics = perf_measure.get_performance_metrics(test, predictions) 
<<<<<<< Updated upstream:main_code.py
                    # Compute naive performance metrics
                    metrics_naive = perf_measure.get_performance_metrics(test, naive_predictions)
                    # Save the index of the last element of the training set
                    end_index = len(train)
                    # Save model data
                    save_data("test", folder_path, args.model_type, model, args.dataset_path, metrics, best_order, end_index)                
=======
=======


            perf_measure = PerfMeasure(args.model_type, model, test, args.target_column, args.forecast_type)
            
            match args.model_type:
            
                case 'ARIMA':
                    # Compute performance metrics
                    metrics = perf_measure.get_performance_metrics(test, predictions) 
>>>>>>> Stashed changes
                    # Save the index of the last element of the training set
                    end_index = len(train)
                    # Save model data
                    save_data("test", args.validation, folder_path, args.model_type, model, args.dataset_path, metrics, best_order, end_index)                
<<<<<<< Updated upstream
>>>>>>> Stashed changes:forecasting_framework.py

                elif(args.model_type == 'SARIMAX'):
                    # Compute performance metrics
                    metrics = perf_measure.get_performance_metrics(target_test, predictions)
<<<<<<< Updated upstream:main_code.py
                    # Compute naive seasonal performance metrics
                    metrics_seasonal_naive = perf_measure.get_performance_metrics(target_test, naive_predictions) 
                    # Save the index of the last element of the training set
                    end_index = len(train)
                    # Save model data
                    save_data("test", folder_path, args.model_type, model, args.dataset_path, metrics, best_order, end_index)  
                        
=======
=======

                case 'SARIMAX'|'SARIMA':
                    # Compute performance metrics
                    metrics = perf_measure.get_performance_metrics(target_test, predictions)
>>>>>>> Stashed changes
                    # Save the index of the last element of the training set
                    end_index = len(train)
                    # Save model data
                    save_data("test", args.validation, folder_path, args.model_type, model, args.dataset_path, metrics, best_order, end_index)  

                case 'LSTM':
                    # Compute performance metrics 
                    metrics = perf_measure.get_performance_metrics(y_test, predictions)
                    # Save the index of the last element of the training set
                    end_index = len(train)
                    # Save model data
                    save_data("test", args.validation, folder_path, args.model_type, model, args.dataset_path, metrics, end_index = end_index)  

                case 'XGB':
                    # Compute performance metrics 
                    metrics = perf_measure.get_performance_metrics(y_test, predictions)
                    # Save the index of the last element of the training set
                    end_index = len(train)
                    # Save model data
                    save_data("test", args.validation, folder_path, args.model_type, model, args.dataset_path, metrics, end_index = end_index)

                case 'NAIVE':
                    metrics = None
                    naive_metrics = perf_measure.get_performance_metrics(test, predictions, naive = True) 
                    save_data("test", args.validation, folder_path, args.model_type, model, args.dataset_path, naive_performance = naive_metrics)

<<<<<<< Updated upstream
>>>>>>> Stashed changes:forecasting_framework.py
            #################### END OF PERFORMANCE MEASUREMENT AND SAVING ####################
        


            #################### PRINT AND PLOT PERFORMANCE ####################
            if verbose:
                # Statistic Models
                if args.model_type == 'ARIMA':
                    perf_measure.print_stats_performance(args.model_type, metrics, metrics_naive)
                    perf_measure.plot_stats_performance(args.model_type, metrics, metrics_naive)
                
                elif args.model_type == 'SARIMAX':
                    perf_measure.print_stats_performance(args.model_type, metrics, metrics_seasonal_naive)
                    perf_measure.plot_stats_performance(args.model_type, metrics, metrics_seasonal_naive)             

            #################### END OF PRINT AND PLOT PERFORMANCE ####################
=======
            #################### END OF PERFORMANCE MEASUREMENT AND SAVING ####################

>>>>>>> Stashed changes
        
    except Exception as e:
        print(f"An error occurred in Main: {e}")

if __name__ == "__main__":
    main()
