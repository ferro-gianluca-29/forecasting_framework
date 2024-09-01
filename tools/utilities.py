import numpy as np
import matplotlib.pyplot as plt
import json
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
<<<<<<< Updated upstream
from statsmodels.tsa.seasonal import STL
=======
>>>>>>> Stashed changes
import os
import pickle
import numpy as np
import pandas as pd
import sys

def conditional_print(verbose, *args, **kwargs):
<<<<<<< Updated upstream
    if verbose:
        print(*args, **kwargs)

# Function to extract metrics, model type, and forecast type
def extract_metrics(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

        metrics = {}
        model_type = ""
        forecast_type = ""
        for line in lines:
            if "Model Type" in line:
                model_type = line.split(':')[1].strip()
            elif "Dataset:" in line:
                dataset_name = line.split(':')[1].strip()
            elif "Command used for launch" in line:
                # Extract forecast type
                command_parts = line.split('--forecast_type')
                if len(command_parts) > 1:
                    forecast_type = command_parts[1].split('--')[0].strip()
                    # Remove any whitespace and special characters
                    forecast_type = forecast_type.strip().strip("',")
            elif "Performance" in line:
                # Extract and parse the performance line
                performance_data = line.split(':', 1)[1].strip().replace("'", '"')
                performance_dict = json.loads(performance_data)

                if model_type in ['ARIMA', 'SARIMAX', 'PROPHET']:
                    metrics['mse'] = performance_dict.get("mean_squared_error")
                    metrics['mape'] = performance_dict.get("mean_absolute_percentage_error")
                else:
                    metrics['mse'] = performance_dict.get("Deep Model", [None, None, None])[1]
                    metrics['mape'] = performance_dict.get("Deep Model", [None, None, None])[2]

        metrics['model_type'] = f"{model_type}{forecast_type}" if forecast_type else model_type
        return metrics, dataset_name


def ts_analysis(df, target_column, seasonal_period):

    # Print dataset statistics for each column
    print("Dataset statistics:\n", df.describe())

    # Number of NaNs per column
    print("Number of NaNs per column:\n", df.isna().sum())


    # Calculate the number of outliers across the dataset

    # Select only numeric columns
    numeric_cols = df.select_dtypes(include=[np.number])

    # Calculate the IQR (Interquartile Range) only for numeric columns
    Q1 = numeric_cols.quantile(0.25)
    Q3 = numeric_cols.quantile(0.75)
    IQR = Q3 - Q1

    # Create a mask for outliers only on numeric columns
    outliers_mask = (numeric_cols < (Q1 - 1.5 * IQR)) | (numeric_cols > (Q3 + 1.5 * IQR))
    num_outliers = outliers_mask.sum()
    print("Number of outliers per column:\n", num_outliers)
   
    # ADF test for stationarity
    adf_result = adfuller(df[target_column].dropna())
    p_value = adf_result[1]
    adf_statistic = adf_result[0]
    alpha = 0.05
   
    if p_value < alpha and adf_statistic < adf_result[4]['5%']:
        print("The series is stationary.")
    else:
        print("The series is not stationary.")

    # ACF and PACF plots
    print("\n===== ACF and PACF Plots =====")
    plot_acf(df[target_column].dropna())
    plt.show()
    
    plot_pacf(df[target_column].dropna())
    plt.show()

    # Time series decomposition into its trend, seasonality, and residuals components
    decomposition = STL(df[target_column], period=seasonal_period).fit()
    
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, ncols=1, sharex=True, figsize=(10,8))

    ax1.plot(decomposition.observed)
    ax1.set_ylabel('Observed')

    ax2.plot(decomposition.trend)
    ax2.set_ylabel('Trend')

    ax3.plot(decomposition.seasonal)
    ax3.set_ylabel('Seasonal')

    ax4.plot(decomposition.resid)
    ax4.set_ylabel('Residuals')

    fig.autofmt_xdate()
    plt.tight_layout()
    # Add title
    plt.suptitle(f"Time Series Decomposition with period {seasonal_period}")
    plt.show()
  
def save_data(save_mode, path, model_type, model, dataset, performance = None, 
              best_order=None, end_index=None, valid_metrics = None):
    
=======
    """
    Prints provided arguments if the verbose flag is set to True.

    :param verbose: Boolean, controlling whether to print.
    :param args: Arguments to be printed.
    :param kwargs: Keyword arguments to be printed.
    """
    if verbose:
        print(*args, **kwargs)

def save_data(save_mode, validation, path, model_type, model, dataset, performance = None, naive_performance = None,
              best_order=None, end_index=None, valid_metrics = None):
    """
    Saves various types of data to files based on the specified mode.

    :param save_mode: String, 'training' or 'test', specifying the type of data to save.
    :param validation: Boolean, indicates if validation metrics should be saved.
    :param path: Path where the data will be saved.
    :param model_type: Type of model used.
    :param model: Model object to be saved.
    :param dataset: Name of the dataset used.
    :param performance: model performance metrics to be saved.
    :param naive_performance: naive model performance metrics to be saved.
    :param best_order: best model order to be saved.
    :param end_index: index of the last training point.
    :param valid_metrics: validation metrics to be saved.
    """
>>>>>>> Stashed changes
    try:
        os.makedirs(path, exist_ok=True)
    except OSError as error:
        print(f"Error creating the save directory: {error}")
        return
    
<<<<<<< Updated upstream
    file_mode = "a" if os.path.exists(f"{path}/model_details.txt") else "w"

    try:
        with open(f"{path}/model_details.txt", file_mode) as file:
=======
    file_mode = "a" if os.path.exists(f"{path}/model_details_{model_type}.txt") else "w"

    try:
        with open(f"{path}/model_details_{model_type}.txt", file_mode) as file:
>>>>>>> Stashed changes

            if save_mode == "training":
                # Training Info
                file.write(f"Training Info:\n")
                file.write(f"Model Type: {model_type}\n")
                file.write(f"Best Order: {best_order}\n")
                file.write(f"End Index: {end_index}\n")
                file.write(f"Dataset: {dataset}\n")
<<<<<<< Updated upstream
                file.write(f"Validation RMSE:\n {valid_metrics[0]}\n")
                file.write(f"Validation MSE:\n {valid_metrics[1]}\n")
                file.write(f"Validation MAE:\n {valid_metrics[2]}\n")
                file.write(f"Validation MAPE:\n {valid_metrics[3]}\n")
                file.write(f"Launch Command Used:{sys.argv[1:]}\n")

                with open(f"{path}/model.pkl", "wb") as file:
                    pickle.dump(model, file)
=======

                if validation:
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
>>>>>>> Stashed changes

            elif save_mode == "test":
                # Test Info
                file.write(f"Test Info:\n")
<<<<<<< Updated upstream
                file.write(f"Performance: {performance}\n") 
=======
                file.write(f"Model Performance: {performance}\n") 
                file.write(f"Naive Model Performance: {naive_performance}\n") 
>>>>>>> Stashed changes
                file.write(f"Launch Command Used:{sys.argv[1:]}\n")
            
            
        
    except IOError as error:
        print(f"Error during saving: {error}")
    
    print(f"\nSaving completed. Data has been saved in the folder {path}")

def save_buffer(folder_path, df, target_column, size = 20, file_name = 'buffer.json'):
<<<<<<< Updated upstream

    # Select the last rows with the specified columns using iloc
    time_column_index = 0
    target_col_index = df.columns.get_loc(target_column)
    buffer_df = df.iloc[-size:, [time_column_index, target_col_index]]
    
    # Convert timestamp column to string format
    buffer_df.iloc[:, 0] = buffer_df.iloc[:, 0].astype(str)
=======
    """
    Saves a buffer of the latest data points to a JSON file.

    :param folder_path: Directory path where the file will be saved.
    :param df: DataFrame from which data will be extracted.
    :param target_column: Column whose data is to be saved.
    :param size: Number of rows to save from the end of the DataFrame.
    :param file_name: Name of the file to save the data in.
    """
    # Select the last rows with the specified column 
    buffer_df = df.iloc[-size:][[target_column]]  # Modified to keep as DataFrame
    
    # Convert the index timestamp column to string format if needed
    buffer_df.index = buffer_df.index.astype(str)
>>>>>>> Stashed changes

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
<<<<<<< Updated upstream
    model = None
    end_index = None
    best_order = None

    try:
        if model_type in ['ARIMA', 'SARIMAX']:
            with open(f"{folder_name}/model.pkl", "rb") as file:
                model = pickle.load(file)

            with open(f"{folder_name}/model_details.txt", "r") as file:
                for line in file:
                    if "End Index" in line:
                        end_index = int(line.split(":")[1].strip())
                    elif "Best Order" in line:
=======
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

            with open(f"{folder_name}/model_details_{model_type}.txt", "r") as file:
                for line in file:
                    if "Best Order" in line:
>>>>>>> Stashed changes
                        best_order_values = line.split(":")[1].strip().strip("()").split(", ")
                        best_order = tuple(map(int, best_order_values)) if best_order_values != [''] else None

    except FileNotFoundError:
        print(f"The folder {folder_name} does not contain a trained model.")
    except Exception as error:
        print(f"Error during model loading: {error}")

<<<<<<< Updated upstream
    return model, end_index, best_order
=======
    return model, best_order
>>>>>>> Stashed changes

