Parser Arguments
==============================

The command-line parser in the forecasting framework configures settings for various model implementations and applications. It provides multiple options to tailor the time series analysis, model training, and testing processes.

General Arguments
-----------------

**--verbose**
  Minimizes the additional information provided during the program's execution if specified. Default: ``False``.

**--ts_analysis**
  If ``True``, performs an analysis on the time series. Default: ``False``.

**--run_mode**
  Specifies the running mode, which must be one of 'training', 'testing', 'both', or 'fine tuning'. This parameter is required.

Dataset Arguments
-----------------

**--dataset_path**
  Specifies the file path to the dataset. This parameter is required.

**--date_format**
  Specifies the date format in the dataset, crucial for correct datetime parsing. This parameter is required.

**--date_list**
  Provides a list of dates defining the start and end for training, validation, and testing phases, tailored to the model's needs.

**--train_size**
  Sets the proportion of the dataset to be used for training. Default: ``0.7``.

**--val_size**
  Sets the proportion of the dataset to be used for validation. Default: ``0.2``.

**--test_size**
  Sets the proportion of the dataset to be used for testing. Default: ``0.1``.

**--scaling**
  If ``True``, scales the data. This is essential for models sensitive to the magnitude of data.

**--validation**
  If ``True``, includes a validation set in the data preparation process. Default: ``False``.

**--target_column**
  Specifies the column to be forecasted. This parameter is required.

**--time_column_index**
  Specifies the index of the column containing timestamps. Default: ``0``.

Model Arguments
---------------

**--model_type**
  Indicates the type of model to be used. This parameter is required.

**--forecast_type**
  Defines the forecast strategy: 'ol-multi' (open-loop multi-step), 'ol-one' (open loop one-step), or 'cl-multi' (closed-loop multi-step).

**--valid_steps**
  Number of time steps to use during the validation phase. Default: ``10``.

**--steps_jump**
  Specifies the number of time steps to skip during open-loop multi-step predictions. Default: ``50``.

**--exog**
  Defines one or more exogenous variables for models like SARIMAX, enhancing model predictions.

**--period**
  Sets the seasonality period, critical for models handling seasonal variations. Default: ``24``.

**--set_fourier**
  If ``True``, incorporates Fourier terms as exogenous variables, useful for capturing seasonal patterns in data.

Other Arguments
---------------

**--seasonal_model**
  Activates the inclusion of a seasonal component in models like LSTM or XGB.

**--input_len**
  Specifies the number of timesteps for input in models like LSTM. Default: ``24``.

**--output_len**
  Defines the number of timesteps to predict in each window for LSTM models. Default: ``1``.

**--model_path**
  Provides the path to a pre-trained model, facilitating fine-tuning or continued training from a saved state.

**--ol_refit**
  For ARIMA and SARIMA models, allows the model to be retrained for each new observation during open-loop forecasts. Default: ``False``.

**--unscale_predictions**
  If specified, predictions and test data are unscaled, essential for interpreting results in their original scale.

Usage Examples
=================

1. **Training an ARIMA Model with Data Scaling and Validation**

   This example sets up the framework to train an ARIMA model, applying data scaling and including a validation dataset.
   Use the command below to execute the training process:

   .. code-block:: bash

      python main_code.py --run_mode training --dataset_path '/path/to/dataset.csv' --date_format '%Y-%m-%d' --target_column 'sales' --model_type 'ARIMA' --scaling --validation --date_list '2022-01-01' '2022-06-30' '2022-07-01' '2022-08-31' '2022-09-01' '2022-09-30'

2. **Fine-Tuning a Pre-trained LSTM Model for Multi-step Forecasting**

   Demonstrates how to fine-tune an LSTM model for multi-step forecasting, specifying the length of input and output sequences. This setup also includes enabling a seasonal component for the LSTM model:

   .. code-block:: bash

      python main_code.py --run_mode fine_tuning --dataset_path '/path/to/dataset.csv' --date_format '%Y-%m-%d' --target_column 'temperature' --model_type 'LSTM' --model_path '/path/to/pretrained_model' --input_len 24 --output_len 3 --seasonal_model

3. **Testing an XGBoost Model with Feature Engineering**

   This example showcases testing an XGBoost model that incorporates Fourier features as part of its feature engineering process to capture seasonal patterns. 

   .. code-block:: bash

      python main_code.py --run_mode testing --dataset_path '/path/to/dataset.csv' --date_format '%Y-%m-%d' --target_column 'energy_consumption' --model_type 'XGB' --seasonal_model --set_fourier --model_path '/path/to/pretrained_model' --unscale_predictions

