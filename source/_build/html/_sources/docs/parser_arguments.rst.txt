Parser Arguments
==============================

The command-line parser manages the framework settings, allowing different implementations and uses of the models. 
It provides many options to customize time series analysis or training and testing of models. 

Parameters
----------

**--verbose**
  Minimizes the additional information provided during the program's execution if specified. Default: ``False``.

**--ts_analysis**
  If ``True``, performs an analysis on the time series. Default: ``False``.

**--run_mode**
  Specifies the running mode, which must be one of 'training', 'testing', 'both', or 'fine tuning'. This parameter is required.

**--dataset_path**
  Specifies the file path to the dataset. This parameter is required.

**--date_list**
  If the "--validation" argument is True, provides start and end dates for training, validation, and test sets, otherwise gives start and end dates for training and test.
  In test-only mode, only the first two dates are used as test set start and end.  

**--seasonal_split**
  If ``True``, adjusts the data split to account for seasonality. Default: ``False``.

**--train_size**
  Sets the proportion of data to be used for training. Default: ``0.7``.

**--val_size**
  Sets the proportion of data to be used for validation. Default: ``0.2``.

**--test_size**
  Sets the proportion of data to be used for testing. Default: ``0.1``.

**--scaling**
  If ``True``, scales the data. This parameter is required.

**--validation**
  If ``True``, includes validation set creation in the data preparation process (not applicable for ARIMA-SARIMAX models). Default: ``False``.

**--target_column**
  Specifies the column name to be used as the target variable for forecasting. This parameter is required.

**--time_column_index**
  Specifies the index of the timestamp column in the dataset. Default: ``0``.

**--model_type**
  Specifies the type of model to be used. Options include 'ARIMA', 'SARIMAX', 'PROPHET', 'CONV', 'LSTM', 'CNN_LSTM'. This parameter is required.

**--forecast_type**
  Specifies the type of forecast; options are 'ol-multi', 'ol-one', 'cl-multi'. Not necessary for 'PROPHET'.

**--steps_ahead**
  Defines the number of time steps to forecast ahead. Default: ``10``.

**--steps_jump**
  Specifies the number of time steps to skip. Default: ``50``.

**--exog**
  Specifies exogenous columns for the SARIMAX model. Accepts multiple values.

**--period**
  Defines the seasonality period for the SARIMAX model. Default: ``24``.

**--seasonal_model**
  If ``True``, performs a seasonal decomposition, and the seasonal component is fed into the LSTM model.

**--model_path**
  Specifies the path of the pre-trained model for fine-tuning. Default: ``None``.

**--ol_refit**
  For ARIMA and SARIMAX models, if specified, the model is retrained for each added observation in open-loop forecasts. Default: ``False``.

Usage Example
-------------
To run the application in training mode with an ARIMA model with validation, use:
``python main_code.py --run_mode training --dataset_path '/path/to/dataset.csv' --target_column 'open' --model_type "ARIMA" --scaling --validation ``
