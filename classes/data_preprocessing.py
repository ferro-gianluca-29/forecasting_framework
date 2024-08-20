from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import datetime as datetime
import pickle

class DataPreprocessor():
    """
    A class to handle operations of preprocessing, including tasks such as managing NaN values,
    removing non-numeric columns, splitting datasets, managing outliers, and scaling data.

    :param file_ext: File extension for saving datasets.
    :param run_mode: Mode of operation ('train', 'test', 'train_test', 'fine_tuning').
    :param model_type: Type of machine learning model to prepare data for.
    :param df: DataFrame containing the data.
    :param target_column: Name of the target column in the DataFrame.
    :param dates: Indexes of dates given by command line with --date_list.
    :param scaling: Boolean flag to determine if scaling should be applied.
    :param validation: Boolean flag to determine if a validation set should be created.
    :param train_size: Proportion of data to be used for training.
    :param val_size: Proportion of data to be used for validation.
    :param test_size: Proportion of data to be used for testing.
    :param folder_path: Path to folder for saving data.
    :param model_path: Path to model file for loading or saving the model.
    :param verbose: Boolean flag for verbose output.
    """    
    def __init__(self, file_ext, run_mode, model_type, df: pd.DataFrame, target_column: str, dates = None, 
                 scaling = False, validation = None, train_size = 0.7, val_size = 0.2, test_size = 0.1, 
                 folder_path = None, model_path = None,  verbose = False):
        
        self.file_ext = file_ext
        self.run_mode = run_mode
        self.dates = dates
        self.model_type = model_type
        self.df = df
        self.target_column = target_column
        self.target_column_index = self.df.columns.get_loc(target_column)
        self.scaling = scaling
        self.validation = validation
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.folder_path = folder_path
        self.model_path = model_path
        self.verbose = verbose

    def conditional_print(self, *args, **kwargs):
        """
        Print messages conditionally based on the verbose attribute.

        :param args: Non-keyword arguments to be printed
        :param kwargs: Keyword arguments to be printed
        """
        if self.verbose:
            print(*args, **kwargs)

    def preprocess_data(self):
        """
        Main method to preprocess the dataset according to specified configurations.

        :return: Depending on the mode, returns the splitted dataframe and an exit flag.
        """
        exit = False
        try:
            print('\nData preprocessing in progress...\n')

            ########## NaN MANAGEMENT ##########
            self.df, exit = self.manage_nan(self.df)
      
            if exit:
                raise Exception('The dataset has been modified, please reload the file')
            
            ######## END NaN MANAGEMENT ########


            ########### REMOVING NON-NUMERIC COLUMNS ############
            
            # If there are columns containing non-numeric characters (excluding dates) they are removed
            #non_numeric_cols = self.df.select_dtypes(include=['object']).columns
            # Remove the target column from the list of columns to be deleted, if it is of object type
            #non_numeric_cols = non_numeric_cols.drop(self.target_column, errors='ignore')
            # Deletes the non-numeric columns from the DataFrame
            #self.df.drop(columns=non_numeric_cols, inplace=True)      
            #############################
            
            
            ############## SPLIT DATASET ##############

            train, test, valid = self.split_data(self.df)

            #######################

            ######### OUTLIER MANAGEMENT #########
            if self.run_mode != "test":
                # Removing outliers from the training set
                train = self.replace_outliers(train)

            ######### END OUTLIER MANAGEMENT #########
            
            ############## DATA SCALING ##############
            if self.scaling:

                if self.run_mode == "train" or self.run_mode == "train_test":
                    scaler = MinMaxScaler()
                    # fit the scaler on the training set
                    train = train.applymap(lambda x: x.replace(',', '.') if isinstance(x, str) else x)
                    scaler.fit(train[train.columns[0:train.columns.shape[0] - 1]])
                    # save training scaling data with pickle
                    with open(f"{self.folder_path}/scaler.pkl", "wb") as file:
                        pickle.dump(scaler, file)
                    # scale training data    
                    train[train.columns[0:train.columns.shape[0] - 1]] = scaler.transform(train[train.columns[0:train.columns.shape[0] - 1]])
                    if self.validation: 
                        valid = valid.applymap(lambda x: x.replace(',', '.') if isinstance(x, str) else x)
                        valid[valid.columns[0:valid.columns.shape[0] - 1]] = scaler.transform(valid[valid.columns[0:valid.columns.shape[0] - 1]])
                    if self.run_mode == "train_test":    
                        # scale test data
                        test = test.applymap(lambda x: x.replace(',', '.') if isinstance(x, str) else x)
                        test[test.columns[0:test.columns.shape[0] - 1]] = scaler.transform(test[test.columns[0:test.columns.shape[0] - 1]])
                        

                if self.run_mode == "test":
                    # load scaling data from pkl file
                    with open(f"{self.model_path}/scaler.pkl", "rb") as file:
                        scaler = pickle.load(file)
                    # The last column is the date column, so it is not considered
                    num_features = test.columns.shape[0] - 1
                    test = test.applymap(lambda x: x.replace(',', '.') if isinstance(x, str) else x)
                    test[test.columns[0:num_features]] = scaler.transform(test[test.columns[0:num_features]]) 
                
                if self.run_mode == "fine_tuning": 
                    # load scaling data from pkl file
                    with open(f"{self.model_path}/scaler.pkl", "rb") as file:
                        scaler = pickle.load(file)
                    num_features = train.columns.shape[0] - 1
                    train[train.columns[0:num_features]] = scaler.transform(train[train.columns[0:num_features]])
                    if self.validation: valid[valid.columns[0:num_features]] = scaler.transform(valid[valid.columns[0:num_features]])
                    test[test.columns[0:num_features]] = scaler.transform(test[test.columns[0:num_features]])   

            ############ END DATA SCALING ###########

            print("Data preprocessing complete")
            if self.run_mode == "test":
                 return test, exit
            else:    
                if self.validation:
                    return train, test, valid, exit
                else:
                    return train, test, exit

        except Exception as e:
            print(f"An error occurred during preprocessing: {e}")
            return None
         
    def manage_nan(self, df, max_nan_percentage=50, min_nan_percentage=10, percent_threshold = 40):
        """
        Manage NaN values in the dataset based on defined percentage thresholds and interpolation strategies.

        :param df: Dataframe to analyze
        :param max_nan_percentage: Maximum allowed percentage of NaN values for a column to be interpolated or kept
        :param min_nan_percentage:  Minimum percentage of NaN values for which linear interpolation is applied
        :param percent_threshold: Threshold percentage of NaNs in the target column to decide between interpolation and splitting the dataset
        :return: A tuple (df, exit), where df is the DataFrame after NaN management, and exit is a boolean flag indicating if the dataset needs to be split
        """
        # Save the original index
        original_index = self.df.index
        # Reset the index
        df.reset_index(drop=True, inplace=True)
        # percent_threshold is the percentage threshold of NaNs in the target column to split the file
        exit = False
        # Calculate the percentage of NaNs for each column
        nan_percentages = df.isna().mean() * 100
        # Columns for linear interpolation
        lin_interpol_cols = nan_percentages[(nan_percentages > 0) & (nan_percentages < min_nan_percentage)].index
        # Columns for polynomial interpolation
        pol_interpol_cols = nan_percentages[(nan_percentages >= min_nan_percentage) & (nan_percentages <= max_nan_percentage)].index
        # Apply linear interpolation
        df[lin_interpol_cols] = df[lin_interpol_cols].interpolate(method='linear', limit_direction='both')
        # Apply polynomial interpolation
        df[pol_interpol_cols] = df[pol_interpol_cols].interpolate(method='polynomial', order=2, limit_direction='both')
        # Columns other than target with NaN percentage higher than the maximum allowed
        columns_to_drop = nan_percentages[(nan_percentages > max_nan_percentage) & (nan_percentages.index != self.target_column)].index
        # Remove columns other than target with high percentage of NaNs
        df.drop(columns=columns_to_drop, inplace=True)
        # Operations for the target column if it has a number of NaNs above the maximum threshold 
        if nan_percentages[self.target_column] > max_nan_percentage:         
        # Calculate indices related to NaN holes in all DataFrame columns
            nan_hole = self.detect_nan_hole(df)   
            # If there is a hole in the target column
            if nan_hole[self.target_column][0] is not None:
                # Calculate the size of the NaN hole
                hole_dim = nan_hole[self.target_column][1] - nan_hole[self.target_column][0]
                # If the percentage of the NaN hole is less than a set threshold, fill NaNs with polynomial interpolation
                if hole_dim/len(df) * 100 < percent_threshold:
                    df[self.target_column].interpolate(method='polynomial', inplace=True)
                # Otherwise, split the file into two separate files and return an exit flag 'exit'
                else:
                    self.split_file_at_nanhole(nan_hole)
                    print('\nThe dataset has been divided. Restart and launch with the new dataset.\n')
                    exit = True
                    return df, exit
            # If there is no hole in the target column, fill NaNs with polynomial interpolation
            else:
                df[self.target_column].interpolate(method='polynomial', inplace=True)

        df.index = original_index
        return df, exit
        
    def detect_nan_hole(self, df):
        """
        Detects the largest contiguous NaN hole in the target column.

        :param df: DataFrame in which to find the NaN hole
        :return: A dictionary with the start and end indices of the largest NaN hole in the target column
        """
        target_column = self.target_column
        # Dictionary to store the start and end indices of the consecutive NaN group for the target column
        nan_hole = {}
        # Find the target column in the DataFrame
        target = df[target_column]
        # Find NaN values in the target column
        is_nan = target.isna()
        # Calculate groups of consecutive NaN or non-NaN values
        groups = is_nan.ne(is_nan.shift()).cumsum()
        # Select only the groups containing NaNs
        consecutive_nan_groups = groups[is_nan]
        # If there are no consecutive NaN groups, record None for start and end
        if consecutive_nan_groups.empty:
            nan_hole[target_column] = (None, None)
        else:
            # Calculate the lengths of the groups and find the longest group
            group_lengths = consecutive_nan_groups.value_counts()
            longest_group = group_lengths.idxmax()
            # Find the start and end indices of the longest consecutive NaN group
            group_start = consecutive_nan_groups[consecutive_nan_groups == longest_group].index.min()
            group_end = consecutive_nan_groups[consecutive_nan_groups == longest_group].index.max()
            # Record the start and end indices in the dictionary
            nan_hole[target_column] = (group_start, group_end)
                
        return nan_hole    
    
    # TO BE MODIFIED: HANDLE OTHER EXTENSIONS AS WELL
    def split_file_at_nanhole(self, nan_hole):
        """
        Splits the dataset at a significant NaN hole into two separate files.

        :param nan_hole: Dictionary containing start and end indices of the NaN hole in the target column
        """
        target_column = self.target_column
        # Extract the start and end indices from the target column within nan_hole
        start, end = nan_hole[target_column]
        # Save the time when you are creating the csvs
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M")
        # Create the name for the first CSV file (0 to the first index)    
        first_file_name = f"dataset_part_1_{timestamp}{self.file_ext}"      
        csv1 = self.df.iloc[:start+1]
        csv1.to_csv(first_file_name, index=False)
        # Create the name for the second CSV file (from the second index to the end of the group in the target column)
        second_file_name = f"dataset_part_2_{timestamp}{self.file_ext}"
        csv2 = self.df.iloc[end+1:]
        csv2.to_csv(second_file_name, index=False)
    
    def replace_outliers(self,df):
        """
        Replaces outliers in the dataset based on the Interquartile Range (IQR)
        method. Instead of analyzing the entire dataset at once, this method focuses on a window of data points at a time. 
        The window moves through the data series step by step. For each step, it includes the next data point
        in the sequence while dropping the oldest one, thus maintaining a constant
        window size. For each position of the window, the function calculates the
        first (Q1) and third (Q3) quartiles of the data within the window. These
        quartiles are used to determine the Interquartile Range (IQR), from which
        lower and upper bounds for outliers are derived.


        :param df: DataFrame from which to remove and replace outliers
        :return: DataFrame with outliers replaced
        """
        # Set the window size and k factor
        window_size = 7  # Increase if execution is slow
        k = 1.5  # standard factor for IQR
        # Select only numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        total_outliers = 0
        # Calculate IQR only for numeric columns
        for column in numeric_cols:
            # Calculate IQR only for numeric columns
            Q1 = df[column].rolling(window= window_size).quantile(0.25)
            Q3 = df[column].rolling(window= window_size).quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - (k * IQR)
            upper_bound = Q3 + (k * IQR)
            
            # Count outliers
            outliers_lower = (df[column] < lower_bound).sum()
            outliers_upper = (df[column] > upper_bound).sum()
            # Add up outliers for each column
            total_outliers += outliers_lower + outliers_upper

            # Replace values below the lower limit with the lower limit itself
            df[column] = df[column].mask(df[column] < lower_bound, lower_bound)
            # Replace values above the upper limit with the upper limit itself
            df[column] = df[column].mask(df[column] > upper_bound, upper_bound)
        self.conditional_print("Number of outliers:", total_outliers)
     
        return df

    def print_stats(self, train):
        """
        Print statistics for the selected feature in the training dataset.

        :param train: DataFrame containing the training data
        """
        # Print on the standard output the statistics of the dataset (for the selected feature)
        max_value = train[self.target_column].max()
        min_value = train[self.target_column].min()
        mean_value = train[self.target_column].mean()
        variance_value = train[self.target_column].var()

        # Creating and printing a table with the statistics
        stats_train = pd.DataFrame({
            'MAX': [max_value],
            'MIN': [min_value],
            'MEAN': [mean_value],
            'VARIANCE': [variance_value],
        })
        print(f'Statistics for the target column "{self.target_column}":')
        print(stats_train)
        print('\n')

    def split_data(self, df):
        """
        Split the dataset into training, validation, and test sets.
        If a list with dates is given, each set is created within the respective dates, otherwise the sets are created following 
        the given percentage sizes.

        :param df: DataFrame to split
        :return: Tuple of DataFrames for training, testing, and validation
        """
        # Data splitting for test mode
        if self.run_mode == "test":
            
            if self.dates is not None:
                # Convert into int values the list containing Int64Index elements
                self.dates = [index[0] for index in self.dates]
                test = df[self.dates[0]:self.dates[1]]
            else:
                test = self.df.iloc[:int(len(self.df) * self.test_size)]
            return None, test, None
        
        else:
            # Data splitting for all other run modes
            n = len(df)
            if self.dates is not None:
                # Convert into int values the list containing Int64Index elements
                self.dates = [index[0] for index in self.dates]

                if self.validation:
                    train = df[self.dates[0]:self.dates[1]]
                    valid = df[self.dates[2]:self.dates[3]]
                    test = df[self.dates[4]:self.dates[5]]
                    return train, test, valid
                else:
                    train = df[self.dates[0]:self.dates[1]]
                    test = df[self.dates[2]:self.dates[3]]
                    return train, test, None 
            else:
                if  self.validation:
                    train_end = int(n * self.train_size)
                    valid_end = int(n * (self.train_size + self.val_size))
                                                                            
                    train = df.iloc[:train_end]
                    valid = df.iloc[train_end:valid_end] 
                    test = df.iloc[valid_end:] 
                    print(f"training: {(train.index[0],train.index[-1])} \t valid: {valid.index[0],valid.index[-1]} \t test: {test.index[0],test.index[-1]}")

                    return train, test, valid
                else:
                    train = df[:int(n * self.train_size)]
                    test = df[int(n * self.train_size):]
                    return train, test, None 
        
    def data_windowing(self, train, valid, test, input_len, output_len, seasonal_model=False, set_fourier = False):
        """
        Creates data windows suitable for input into deep learning models, optionally incorporating Fourier features for seasonality.

        :param train: Training dataset.
        :param valid: Validation dataset.
        :param test: Test dataset.
        :param input_len: Length of the input data window.
        :param output_len: Length of the output data window.
        :param seasonal_model: Flag to include Fourier features for seasonal predictions.
        :param set_fourier: Flag to set Fourier transformation features.
        :return: Arrays of input and output data windows for training, validation, and testing.
        """

        stride = input_len
        X_train, y_train, X_valid, y_valid, X_test, y_test = [], [], [], [], [], []
        indices_train, indices_valid, indices_test = [], [], []

        # Definisci le colonne base e le colonne di Fourier
        

        if self.run_mode in ["train", "train_test", "fine_tuning"]:
            # Processa train e valid sets
            for dataset, X, y, indices in [(train, X_train, y_train, indices_train),
                                        (valid, X_valid, y_valid, indices_valid)]:
                
                fourier_columns = [col for col in dataset.columns if col.startswith(('sin', 'cos'))]
                input_columns = [self.target_column] + fourier_columns if set_fourier else [self.target_column]
                first_window = True

                for i in range(0, len(dataset) - input_len - output_len + 1, stride):
                    X.append(dataset[input_columns].iloc[i:i + input_len].values)
                    y.append(dataset[self.target_column].iloc[i + input_len:i + input_len + output_len].values)
                    indices.append(i)
                    if first_window == True and seasonal_model == False:
                        print(f"X first window from {dataset['date'].iloc[i]} to {dataset['date'].iloc[i+input_len-1]}")
                        print(f"y first window from {dataset['date'].iloc[i+input_len]} to {dataset['date'].iloc[i+input_len+output_len-1]}")
                        first_window = False

        # Test set sempre processato
        if len(test) < input_len + output_len:
            print("Test data is too short for creating windows")
            return None
        else:
            
            fourier_columns = [col for col in test.columns if col.startswith(('sin', 'cos'))]
            input_columns = [self.target_column] + fourier_columns if set_fourier else [self.target_column]
            first_window = True

            stride = input_len
            for i in range(0, len(test) - input_len - output_len + 1, stride):
                X_test.append(test[input_columns].iloc[i:i + input_len].values)
                y_test.append(test[self.target_column].iloc[i + input_len:i + input_len + output_len].values)
                indices_test.append(i)
                if first_window == True and seasonal_model == False:
                    print(f"X_test first window from {test['date'].iloc[i]} to {test['date'].iloc[i+input_len-1]}")
                    print(f"y_test first window from {test['date'].iloc[i+input_len]} to {test['date'].iloc[i+input_len+output_len-1]}")
                    first_window = False

        # Conversione in array e ridimensionamento
        X_train, y_train = np.array(X_train), np.array(y_train)
        X_valid, y_valid = np.array(X_valid), np.array(y_valid)
        X_test, y_test = np.array(X_test), np.array(y_test)

        # Reshape dei dati di input per includere tutte le feature nel modello
        if X_train.size > 0:
            X_train = np.reshape(X_train, (X_train.shape[0], input_len, len(input_columns)))
        if X_valid.size > 0:
            X_valid = np.reshape(X_valid, (X_valid.shape[0], input_len, len(input_columns)))
        X_test = np.reshape(X_test, (X_test.shape[0], input_len, len(input_columns)))

        print("Data windowing complete")
        if self.run_mode == "test":
            return [X_test, y_test]
        else:
            return [X_train, y_train, X_valid, y_valid, X_test, y_test]


    def create_time_features(self, df, label=None, seasonal_model = None, lags = [1, 2, 3, 24], rolling_window = 24):
        """
        Create time-based features for a DataFrame, optionally including Fourier features and rolling window statistics.

        :param df: DataFrame to modify with time-based features.
        :param label: Label column name for generating features.
        :param seasonal_model: Boolean indicating whether to add Fourier features for seasonal models.
        :param lags: List of integers representing lag periods to generate features for.
        :param rolling_window: Window size for generating rolling mean and standard deviation.
        :return: Modified DataFrame with new features, optionally including target column labels.
        """

        df['date'] = df.index
        df['hour'] = df['date'].dt.hour
        df['dayofweek'] = df['date'].dt.dayofweek
        df['quarter'] = df['date'].dt.quarter
        df['month'] = df['date'].dt.month
        df['year'] = df['date'].dt.year
        df['dayofyear'] = df['date'].dt.dayofyear
        df['dayofmonth'] = df['date'].dt.day
        df['weekofyear'] = df['date'].dt.isocalendar().week  # Changed liner

        if seasonal_model:
            # Fourier features for daily, weekly, and yearly seasonality
            for period in [24, 7, 365]:
                df[f'sin_{period}'] = np.sin(df.index.dayofyear / period * 2 * np.pi)
                df[f'cos_{period}'] = np.cos(df.index.dayofyear / period * 2 * np.pi)

            # Lagged features
            #for lag in lags:
                #df[f'lag_{lag}'] = df[label].shift(lag)

            # Rolling window features
            #df[f'rolling_mean_{rolling_window}'] = df[label].shift().rolling(window=rolling_window).mean()
            #df[f'rolling_std_{rolling_window}'] = df[label].shift().rolling(window=rolling_window).std()

            df = df.dropna()  # Drop rows with NaN values resulting from lag/rolling operations
            X = df.drop(['date', label], axis=1, errors='ignore')
        else:
            X = df[['hour','dayofweek','quarter','month','year',
                'dayofyear','dayofmonth','weekofyear']]
        if label:
            y = df[label]
            return X, y
        return X

    