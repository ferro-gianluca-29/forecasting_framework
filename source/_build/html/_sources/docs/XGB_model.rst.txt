XGB Model
=================

This module encapsulates the XGB_Predictor class, which leverages the XGBoost machine learning library to forecast 
time series data. 
The class is specifically designed to incorporate extensive 
feature engineering including lag features, rolling window statistics, and optional Fourier transformations to capture 
seasonal patterns. It focuses on using these enhanced datasets to train and evaluate 
XGBoost models for precise predictions, offering functionalities for scaling, plotting, and performance assessment 
tailored to time series forecasting.

XGB_Predictor
----------------

.. autoclass:: XGB_model.XGB_Predictor
   :members:
   :undoc-members:
   :show-inheritance:
