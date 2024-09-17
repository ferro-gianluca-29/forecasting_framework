import argparse
import pandas as pd
import numpy as np
import os
import datetime
import pickle

from tools.data_loader import DataLoader
from tools.utilities import save_data, save_buffer, load_trained_model

model_type = 'ARIMA'
dataset_path = "./data/Dataset/data_spatial_TotalKW.csv"
date_format = "%m/%d/%Y %H:%M"
target_column = "Global_active_power"
model_path = "./data/models/ARIMA_2024-09-17_11-32-49"

# Create current model folder
folder_name = model_type + "_" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
folder_path = f"./data/models/{folder_name}"
os.makedirs(folder_path)

data_loader = DataLoader(dataset_path, date_format, model_type, target_column, 0,  None, None)
df, dates = data_loader.load_data()

# Load a pre-trained model
pre_trained_model, best_order = load_trained_model(model_type, model_path)
last_train_index = pre_trained_model.data.row_labels[-1] + 1

new_data = df[target_column][13932:18299]
new_data_start_index = last_train_index
new_data_end_index = new_data_start_index + len(new_data)
new_data.index = range(new_data_start_index, new_data_end_index)

model = pre_trained_model.append(new_data, refit = False)
save_data("training", None, folder_path, model_type, model, dataset_path, 
    end_index = 0,  valid_metrics = None)