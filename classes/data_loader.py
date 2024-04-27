import pandas as pd
import os

class DataLoader(): 

    def __init__(self,file_path, model_type, target_column, time_column_index = 0):
        self.file_path = file_path
        self.model_type = model_type
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
                time_column_name = df.columns[self.time_column_index]  # Get the correct column name
                
                if self.time_column_index != 0:
                    # Remove and copy the time column to the first position
                    time_column_data = df.pop(time_column_name)
                    df.insert(0, 'date', time_column_data)
                else:
                    # Rename if it's already the first column
                    df.rename(columns={time_column_name: 'date'}, inplace=True)

                # Convert the 'date' column to datetime
                df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d %H:%M:%S', utc=True)
                # Sort the dataset by date
                df = df.sort_values(by='date')
                # Set the date column as index for neural network models 
                # (in the case of statistical models it may cause index errors during forecasting)
                if self.model_type == 'LSTM':
                    # Make a copy of date column as set it as last column of the dataframe
                    df['temp_date'] = df['date']
                    date = df.pop('temp_date')
                    df = pd.concat([df, date], 1)
                    # set the date column as index
                    df.set_index('date', inplace=True)
                    # Keep the date column (so the code can use the split_data() method of DataPreprocessor without errors)
                    df.rename(columns={'temp_date': 'date'}, inplace=True)
                else:
                    df.reset_index(drop=True, inplace = True)
            else:
                 print("time column not found.")
            return df

    