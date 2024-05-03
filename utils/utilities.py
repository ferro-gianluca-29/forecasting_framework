import numpy as np
import matplotlib.pyplot as plt
import json
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import os
import pickle
import numpy as np
import pandas as pd
import sys

def conditional_print(verbose, *args, **kwargs):
    """
    Prints provided arguments if the verbose flag is set to True.

    :param verbose: Boolean, controlling whether to print.
    :param args: Arguments to be printed.
    :param kwargs: Keyword arguments to be printed.
    """
    if verbose:
        print(*args, **kwargs)

def save_data(save_mode, validation, path, model_type, model, dataset, performance = None, 
              best_order=None, end_index=None, valid_metrics = None):
    """
    Saves various types of data to files based on the specified mode.

    :param save_mode: String, 'training' or 'test', specifying the type of data to save.
    :param validation: Boolean, indicates if validation metrics should be saved.
    :param path: Path where the data will be saved.
    :param model_type: Type of model used.
    :param model: Model object to be saved.
    :param dataset: Name of the dataset used.
    :param performance: performance metrics to be saved.
    :param best_order: best model order to be saved.
    :param end_index: index of the last training point.
    :param valid_metrics: validation metrics to be saved.
    """
    try:
        os.makedirs(path, exist_ok=True)
    except OSError as error:
        print(f"Error creating the save directory: {error}")
        return
    
    file_mode = "a" if os.path.exists(f"{path}/model_details.txt") else "w"

    try:
        with open(f"{path}/model_details.txt", file_mode) as file:

            if save_mode == "training":
                # Training Info
                file.write(f"Training Info:\n")
                file.write(f"Model Type: {model_type}\n")
                file.write(f"Best Order: {best_order}\n")
                file.write(f"End Index: {end_index}\n")
                file.write(f"Dataset: {dataset}\n")
               #if validation:
                for metric in valid_metrics.keys():
                    file.write(f"{metric}:\n {valid_metrics[metric]}\n")
                file.write(f"Launch Command Used:{sys.argv[1:]}\n")
                # Save the model
                match model_type:
                    case 'LSTM':
                        model.save(f"{path}/model.h5")
                    case 'ARIMA'|'SARIMA'|'SARIMAX':
                        with open(f"{path}/model.pkl", "wb") as file:
                            pickle.dump(model, file)
                    case 'XGB':
                        model.save_model(f"{path}/model.json")

            elif save_mode == "test":
                # Test Info
                file.write(f"Test Info:\n")
                file.write(f"Performance: {performance}\n") 
                file.write(f"Launch Command Used:{sys.argv[1:]}\n")
            
            
        
    except IOError as error:
        print(f"Error during saving: {error}")
    
    print(f"\nSaving completed. Data has been saved in the folder {path}")

def save_buffer(folder_path, df, target_column, size = 20, file_name = 'buffer.json'):
    """
    Saves a buffer of the latest data points to a JSON file.

    :param folder_path: Directory path where the file will be saved.
    :param df: DataFrame from which data will be extracted.
    :param target_column: Column whose data is to be saved.
    :param size: Number of rows to save from the end of the DataFrame.
    :param file_name: Name of the file to save the data in.
    """
    # Select the last rows with the specified columns 
    target_col_index = df.columns.get_loc(target_column)
    buffer_df = df.iloc[-size:, target_col_index]
    
    # Convert the index timestamp column to string format
    buffer_df.index = buffer_df.index.astype(str)

    # Serialize the dataframe to a JSON string
    try:
        buffer_json = buffer_df.to_json(orient='records')
        
        # Write the JSON string to the specified file
        with open(f"{folder_path}/{file_name}", 'w') as file:
            file.write(buffer_json)
        
        print(f"Data successfully saved to file {file_name}")
    except Exception as e:
        print(f"An error occurred while saving the file: {e}")

def load_trained_model(model_type, folder_name):
    """
    Loads a trained model and its configuration from the selected directory.

    :param model_type: Type of the model to load ('ARIMA', 'SARIMAX', etc.).
    :param folder_name: Directory from which the model and its details will be loaded.
    :return: A tuple containing the loaded model and its order (if applicable).
    """
    model = None
    best_order = None

    try:
        if model_type in ['ARIMA', 'SARIMAX', 'SARIMA']:
            with open(f"{folder_name}/model.pkl", "rb") as file:
                model = pickle.load(file)

            with open(f"{folder_name}/model_details.txt", "r") as file:
                for line in file:
                    if "Best Order" in line:
                        best_order_values = line.split(":")[1].strip().strip("()").split(", ")
                        best_order = tuple(map(int, best_order_values)) if best_order_values != [''] else None

    except FileNotFoundError:
        print(f"The folder {folder_name} does not contain a trained model.")
    except Exception as error:
        print(f"Error during model loading: {error}")

    return model, best_order

