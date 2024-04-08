import pandas as pd
import os

class DataLoader(): 

    def __init__(self,file_path, target_column, time_column_index = 0):
        self.file_path = file_path
        self.format = os.path.splitext(file_path)[1] 
        self.target_column = target_column
        self.time_column_index = time_column_index 

    def load_data(self):

        # load the dataframe with all the columns
        if self.format == '.csv':
            df = pd.read_csv(self.file_path)
        elif self.format == '.txt':
            df = pd.read_csv(self.file_path, delimiter='\t')
        elif self.format == '.xlsx' or self.format == '.xls':
            df = pd.read_excel(self.file_path)
        elif self.format ==  '.json':
            df = pd.read_json(self.file_path)
        else:
            print("File format not supported.")
            return None

        # check if the target column is present
        if self.target_column not in df.columns:
            print(f"{self.target_column} column not found.")
            return None

        # convert the specified time column
        if self.time_column_index is not None:
            if self.time_column_index < len(df.columns):
                # conversion into the datetime format
                time_column = df.columns[self.time_column_index]
                df[time_column] = pd.to_datetime(df[time_column])
                if self.time_column_index != 0:
                   # remove and copy time column
                   time_column = df.pop(df.columns[self.time_column_index])
                   # make the time column the first of the dataframe
                   df = df.insert(0, df.columns[self.time_column_index], time_column)
                   # Sort the dataset by date
                   df = df.sort_values(by=df.columns[0])
            else:
                 print("time column not found.")
            return df

    