LSTM Model
=================

This module provides functionality for forecasting time series data using Long Short-Term Memory (LSTM) networks. 
It includes the `LSTM_Predictor` class, which employs a Keras (Tensorflow) model to make forecasts.  
The class supports advanced features such as input and output sequence length customization, 
optional Fourier transformation for seasonality, and the capability to handle multi-step forecasts with 
seasonality adjustments.
A data windowing method is also included, in order to give to the neural network the correct input data shape. By correct setting of 
`input_len` and `output_len` command line parameters, both one-step or multi-step ahead predictions can be done.

LSTM Model Structure
--------------------

The model consists of the following sequence of layers:

1. **Input Layer**:

2. **LSTM Layer**:
   - LSTM with 40 units

3. **Dropout Layer**:
   - First dropout layer with rate of 0.15 

4. **LSTM Layer**:
   - Second LSTM layer with 40 units

5. **Dropout Layer**:
   - Second dropout layer with rate of 0.15 

6. **LSTM Layer**:
   - Third LSTM layer with 40 units

7. **Dropout Layer**:
   - Third dropout layer with rate of 0.15 

8. **Dense Layer**:
   - Dense layer for the final output

This structure of the LSTM model is configured to process temporal sequences, 
followed by multiple LSTM and dropout layers to prevent overfitting, concluding with a dense layer for output.

LSTM_Predictor
----------------

.. autoclass:: LSTM_model.LSTM_Predictor
   :members:
   :undoc-members:
   :show-inheritance:


