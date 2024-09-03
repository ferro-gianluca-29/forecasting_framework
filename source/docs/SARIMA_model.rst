SARIMA Model
=================

This module implements Seasonal ARIMA (SARIMA) models for time series forecasting. 
It features the `SARIMA_Predictor` class, which leverages seasonal differencing and potentially exogenous variables, 
like Fourier terms for capturing seasonality. 
The model used comes from the `Statsmodels` library, and the optimization of the hyperparameters
is done through grid search, by finding the model with the best AIC score.  
This model is used for univariate time series, and is designed for both one-step ahead or multi-step ahead forecasts.
Like the ARIMA model, one-step ahead predictions can be done in open loop mode, i.e. by updating at each forecast the 
model with the present observation, in order to make the model suitable for online learning settings.
For datasets with high frequency, like solar panel production with 15 min timesteps, the model may 
not be able to be trained or optimized with daily or weekly seasonality, due to high memory requirements.
For this reason, a setting with additional Fourier terms as exogenous variable is present, and the choice of the hyperparameters could be done
by inspection of the ACF and PACF plots.
An optional rolling window cross validation technique is also implemented in the class.


SARIMA_Predictor
----------------

.. automodule:: SARIMA_model
   :members:
   :undoc-members:
   :show-inheritance:
