ARIMA Model
=================

This module is designed to make forecasts on time series through ARIMA (Autoregressive Integrated Moving Average) models, 
encapsulated in the `ARIMA_Predictor` class. The model used comes from the `Statsmodels` library, and the optimization of the hyperparameters
is done through grid search, by finding the model with the best AIC score.  
This class provides methods for predictive analysis of univariate time series, 
and is designed for both one-step ahead or multi-step ahead forecasts.
One-step ahead predictions can be done in open loop mode, i.e. by updating at 
each forecast the model with the present observation, 
in order to make the model suitable for online learning settings.


ARIMA_Predictor
----------------

.. automodule:: ARIMA_model
   :members:
   :undoc-members:
   :show-inheritance:


      
