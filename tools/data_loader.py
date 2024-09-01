import pandas as pd
import os

class DataLoader(): 
    """
    Class for loading datasets from various file formats and preparing them for machine learning models.

    :param file_path: Path to the dataset file.
    :param date_format: Format of the date in the dataset file, e.g., '%Y-%m-%d'.
    :param model_type: Type of the machine learning model. Supported models are 'LSTM', 'XGB', 'ARIMA', 'SARIMA', 'SARIMAX'.
    :param target_column: Name of the target column in the dataset.
    :param time_column_index: Index of the time column in the dataset (default is 0).
    :param date_list: List of specific dates to be filtered (default is None).
    :param exog: Name or list of exogenous variables (default is None).
    """

    def __init__(self,file_path, date_format, model_type, target_column, time_column_index = 0, date_list = None, exog = None):
        """
        Initialize the DataLoader with the specified parameters.
        """
        self.file_path = file_path
        self.date_format = date_format
        self.model_type = model_type
        self.format = os.path.splitext(file_path)[1] 
        self.target_column = target_column
        self.time_column_index = time_column_index 
        self.date_list = date_list
        self.exog = exog

    def load_data(self):
        """
        Loads data from a file, processes it according to the specified settings,
        and prepares it for machine learning models. This includes formatting date columns,
        filtering specific dates, and adjusting data structure based on the model type.

        :returns: 
            - A tuple containing the dataframe and the indices of the dates if provided in `date_list`.
        """
        # load the dataframe with all the columns
        if self.format == '.csv':
            df = pd.read_csv(self.file_path, sep=None, engine='python')
        elif self.format == '.txt':
            df = pd.read_csv(self.file_path, delimiter='\t')
        elif self.format == '.xlsx' or self.format == '.xls':
            df = pd.read_excel(self.file_path, engine='openpyxl')
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

                # remove columns that will not be used
                useful_columns = [self.target_column, time_column_name]
                if self.exog is not None: 
                    useful_columns.extend(self.exog)
                df = df[useful_columns]

                if self.time_column_index != 0:
                    # Remove and copy the time column to the first position
                    time_column_data = df.pop(time_column_name)
                    df.insert(0, 'date', time_column_data)
                else:
                    # Rename if it's already the first column
                    df.rename(columns={time_column_name: 'date'}, inplace=True)

                # Sort the dataset by date
                #df = df.sort_values(by='date')

                df['temp_date'] = pd.to_datetime(df['date'], format=self.date_format)

                df.sort_values(by='temp_date', inplace=True)
                df.drop('temp_date', axis=1, inplace=True)

                df.reset_index(drop=True, inplace = True)
                # Get the indexes of the sets given by the argument --date_list
                if self.date_list is not None:
                    dates = []
                    for date in self.date_list:     
                        dates.append(df[df['date'] == date].index)
                    # Convert the 'date' column to datetime
                    df['date'] = pd.to_datetime(df['date'], format=self.date_format)
                    
                    
                else:
                    df['date'] = pd.to_datetime(df['date'], format=self.date_format)
                    dates = None

                match self.model_type:
                    case 'LSTM'|'XGB':
                        # Set the date column as index for neural network models 
                        # (in case of statistical models it may cause index errors during forecasting)

                        df.set_index('date', inplace=True)
                        # Keep the date column (so the code can use the split_data() method of DataPreprocessor without errors)
                        df['date'] = df.index
                        
                    case 'ARIMA'|'SARIMA'|'SARIMAX':
                        # Set date as the last column of the dataframe
                        df.insert(df.columns.shape[0] - 1, 'date', df.pop('date'))
            else:
                 print("time column not found.")
            return df, dates