#  LIBRARY IMPORTS

import argparse
import pandas as pd
from matplotlib import pyplot as plt
from classes.data_preprocessing import DataPreprocessor
from classes.data_loader import DataLoader
from classes.training_module import ModelTraining
from classes.model_testing import ModelTest
from classes.performance_measurement import PerfMeasure
import datetime
from utils.utilities import ts_analysis, save_data, save_buffer, load_trained_model
from utils.time_series_analysis import multiple_STL

# END OF LIBRARY IMPORTS #
  
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import warnings
warnings.filterwarnings('ignore') 

def main():
    
    # ARGUMENT PARSING
    parser = argparse.ArgumentParser(description='Time series forecasting')

    # General arguments
    parser.add_argument('--verbose', action='store_true', required=False, default=False, help='If specified, minimizes the additional information provided during the program launch')
    parser.add_argument('--ts_analysis', action='store_true', required=False, default=False, help='If True, performs an analysis on the time series')
    parser.add_argument('--run_mode', type=str, required=True, help='Running mode (training, testing, both, or fine tuning)')

    # Dataset arguments
    parser.add_argument('--dataset_path', type=str, required=True, help='Dataset path')
    parser.add_argument('--seasonal_split', action='store_true', required=False, default=False, help='If True, makes a split that takes into account seasonality')
    parser.add_argument('--train_size', type=float, required=False, default=0.7, help='Training set size')
    parser.add_argument('--val_size', type=float, required=False, default=0.2, help='Validation set size')
    parser.add_argument('--test_size', type=float, required=False, default=0.1, help='Test set size')
    parser.add_argument('--scaling', action='store_true', help='If True, data will be scaled')
    parser.add_argument('--validation', action='store_true', required=False, help='If True, the validation set is created (not for ARIMA-SARIMAX)' )
    parser.add_argument('--target_column', type=str, required=True, help='Name of the target column for forecasting')
    parser.add_argument('--time_column_index', type=int, required=False, default=0, help='Index of the column containing the timestamps')

    # Model arguments
    parser.add_argument('--model_type', type=str, required=True, help='Type of model to use (ARIMA, SARIMAX, PROPHET, CONV, LSTM, CNN_LSTM)')
    parser.add_argument('--forecast_type', type=str, required=False, help='Type of forecast: ol-multi= open-loop multi step ahead; ol-one= open loop one step ahead, cl-multi= closed-loop multi step ahead. Not necessary for PROPHET')
    parser.add_argument('--steps_ahead', type=int, required=False, default=10, help='Number of time steps ahead to forecast')
    parser.add_argument('--steps_jump', type=int, required=False, default=50, help='Number of steps to skip')
    parser.add_argument('--exog', nargs='+', type=str, required=False, default = None, help='Exogenous columns for the SARIMAX model')
    parser.add_argument('--period', type=int, required=False, default=24, help='Seasonality period for the SARIMAX model')    

    # Fine tuning arguments    
    parser.add_argument('--model_path', type=str, required=False, default=None, help='Path of the pre-trained model' )    
    parser.add_argument('--ol_refit', action='store_true', required=False, default=False, help='For ARIMA and SARIMAX models: If specified, in OL forecasts the model is retrained for each added observation ')
       
    args = parser.parse_args()
    # END OF ARGUMENT PARSING

    verbose = args.verbose 

    try:
        
        # Create current model folder
        folder_name = args.model_type + "_" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        folder_path = f"./data/models/{folder_name}"
        os.makedirs(folder_path)
 
        #  DATA LOADING
        data_loader = DataLoader(args.dataset_path, args.target_column, args.time_column_index)
        df = data_loader.load_data()
        if df is None:
            raise ValueError("Unable to load dataset.")
        
        # END OF DATA LOADING
        
                
            

#################### PREPROCESSING  ####################
        
        # Extract the file extension from the path 
        file_ext = os.path.splitext(args.dataset_path)[1]
        
        data_preprocessor = DataPreprocessor(file_ext, args.run_mode, args.model_type, df, args.target_column, 
                                             args.scaling, args.validation, args.train_size, args.val_size, args.test_size, args.seasonal_split,
                                             folder_path, args.model_path, verbose)
        
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
        if (args.model_type == 'SARIMAX') or (args.model_type == 'SARIMA'):
            if args.exog is  None:
                exog = args.target_column
            else: 
                exog = args.exog
            target_train = train[[args.target_column]]
            exog_train = train[exog]
            if args.validation:
                target_valid = valid[[args.target_column]]
                exog_valid = valid[exog]
            else: exog_valid = None
            target_test = test[[args.target_column]]
            exog_test = test[exog]

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
 
            elif (args.model_type == 'SARIMAX') or (args.model_type == 'SARIMA'):

                # Load a pre-trained model
                pre_trained_model, prev_train_end_index, best_order = load_trained_model(args.model_type, args.model_path) 

                # Update the indices so that the the indices are contiguous to those of the pre-trained model
                
                test_start_index = test.index[0] + prev_train_end_index
                test_end_index = test_start_index + len(test)
                test.index = range(test_start_index, test_end_index)
                target_test.index = range(test_start_index, test_end_index)
                exog_test.index = range(test_start_index, test_end_index)
                     
                if args.run_mode == "fine_tuning":  
                    # Update the model with the new data
                    train.index = range(prev_train_end_index, prev_train_end_index + len(train)) 
                    exog_train.index = range(prev_train_end_index, prev_train_end_index + len(train))  
                    model = pre_trained_model.append(train[args.target_column], exog =  exog_train, refit = True)
                elif args.run_mode == "test":
                    # Load the model 
                    model = pre_trained_model

            ######################## # END OF LOAD MODEL ####################
    
        if args.run_mode == "train" or args.run_mode == "train_test":

                #################### MODEL TRAINING ####################
                model_training = ModelTraining(args.model_type, train, valid, args.target_column, verbose = False)

                if (args.model_type == 'ARIMA'):    
                    model, valid_metrics = model_training.train_ARIMA_model()
                    best_order = model_training.ARIMA_order
                    # Save a buffer containing the last elements of the training set for further test
                    buffer_size = 20
                    save_buffer(folder_path, train, args.target_column, size = buffer_size, file_name = 'buffer.json')
                    # Save training data 
                    save_data("training", args.validation, folder_path, args.model_type, model, args.dataset_path, 
                              best_order = best_order, end_index = len(train), valid_metrics = valid_metrics)

                elif (args.model_type == 'SARIMAX') or (args.model_type == 'SARIMA'):  
                    model, valid_metrics = model_training.train_SARIMAX_model(target_train, exog_train, exog_valid, args.period)
                    best_order = model_training.SARIMAX_order
                    # Save a buffer containing the last elements of the training set for further test
                    buffer_size = 20
                    save_buffer(folder_path, train, args.target_column, size = buffer_size, file_name = 'buffer.json')
                    # Save training data
                    save_data("training", args.validation, folder_path, args.model_type, model, args.dataset_path, 
                              best_order = best_order, end_index = len(train),  valid_metrics = valid_metrics)

                #################### END OF MODEL TRAINING ####################
                    
        if args.run_mode == "train_test" or args.run_mode == "fine_tuning" or args.run_mode == "test":
                
            if args.run_mode == "test":

                # Load buffer from JSON file
                train = pd.read_json(f"{args.model_path}/buffer.json", orient='records') 

            #################### MODEL TESTING ####################

            model_test = ModelTest(args.model_type, model, test, args.target_column, args.forecast_type, args.steps_ahead)
            
            if(args.model_type == 'ARIMA'):
                # Model testing
                predictions = model_test.test_ARIMA_model(args.steps_jump, args.ol_refit)    
                # Create the naive model
                naive_predictions = model_test.naive_forecast(train)
                
            elif(args.model_type == 'SARIMAX') or (args.model_type == 'SARIMA'):
                # Model testing
                predictions = model_test.test_SARIMAX_model(args.steps_jump, exog_test, args.ol_refit)   
                # Create the naive model
                naive_predictions = model_test.naive_seasonal_forecast(target_train, target_test, args.period)
            
            #################### END OF MODEL TESTING ####################        
        
        if args.run_mode != "train":

            #################### PLOT PREDICTIONS ####################

            if(args.model_type == 'ARIMA'):
                model_test.ARIMA_plot_pred(best_order, predictions, naive_predictions)
            
            elif (args.model_type == 'SARIMAX') or (args.model_type == 'SARIMA'):
                model_test.SARIMAX_plot_pred(best_order, naive_predictions)

            #################### END OF PLOT PREDICTIONS ####################        
         
            #################### PERFORMANCE MEASUREMENT AND SAVING #################
            if predictions is not None:
                perf_measure = PerfMeasure(args.model_type, model, test, args.target_column, args.forecast_type, args.steps_ahead)
                if(args.model_type == 'ARIMA'):
                    # Compute performance metrics
                    metrics = perf_measure.get_performance_metrics(test, predictions) 
                    # Compute naive performance metrics
                    metrics_naive = perf_measure.get_performance_metrics(test, naive_predictions)
                    # Save the index of the last element of the training set
                    end_index = len(train)
                    # Save model data
                    save_data("test", args.validation, folder_path, args.model_type, model, args.dataset_path, metrics, best_order, end_index)                

                elif (args.model_type == 'SARIMAX') or (args.model_type == 'SARIMA'):
                    # Compute performance metrics
                    metrics = perf_measure.get_performance_metrics(target_test, predictions)
                    # Compute naive seasonal performance metrics
                    metrics_seasonal_naive = perf_measure.get_performance_metrics(target_test, naive_predictions) 
                    # Save the index of the last element of the training set
                    end_index = len(train)
                    # Save model data
                    save_data("test", args.validation, folder_path, args.model_type, model, args.dataset_path, metrics, best_order, end_index)  
                        
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
        
    except Exception as e:
        print(f"An error occurred in Main: {e}")

if __name__ == "__main__":
    main()
