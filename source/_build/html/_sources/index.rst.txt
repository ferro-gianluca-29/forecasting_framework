.. Forewarning framework documentation master file, created by
   sphinx-quickstart on Wed Apr 17 16:23:29 2024.

Forecasting Framework's documentation
======================================================

Introduction
------------
.. rst-class:: justified-text
   

This framework is designed to provide the main blocks for implementing and using many types of machine learning models for time series forecasting, including 
statistical models (ARIMA and SARIMA), Long Short Term Memory (LSTM) neural networks, and Extreme Gradient Boosting (XGB) models. 
Other models can also be integrated, by incorporating the corresponding tools into each respective block of the framework.
The data preprocessing block makes possible to train and test the models on datasets with varying structures and formats, allowing a robust 
support for handling NaN values and outliers. 
The framework comprises a main file that orchestrates the various implementation phases of the models, 
with initial settings provided as 
command-line arguments using a parser (whose parameters are presented in the Appendix).
The code supports four distinct modes of operation: training, testing, combined training and testing, and fine tuning.  
Various configurations of the framework, using different terminal arguments, are present in the JSON files (`launch.json` for debug and `tasks.json` for 
code usage); however, using consistent command line arguments, it is possible to create custom configurations 
by passing parameters directly through the terminal.


Framework Architecture
----------------------

The main blocks of the framework are data loading, data preprocessing, training, testing, and performance measurement.
Once the model is selected, the  file located in the `Predictors` folder corresponding to that model will be used. 
Each of these files contains the classes that implement the training and testing phases of the model.
The blocks of the main code also use classes and functions from a corresponding file located in the `tools` folder,
that includes functionalities as data loading, data preprocessing, optional time series analysis and performance measurement.

Framework Tools
----------------------
Below are reported the classes from the `tools` folder. 

.. toctree::
   :maxdepth: 2

   docs/data_loader

.. toctree::
   :maxdepth: 2

   docs/data_preprocessing

.. toctree::
   :maxdepth: 2

   docs/performance_measurement


.. toctree::
   :maxdepth: 2

   docs/time_series_analysis


.. toctree::
   :maxdepth: 2

   docs/utilities


Framework Models
----------------------
In the following are documented the models used inside the framework.

.. toctree::
   :maxdepth: 2

   docs/ARIMA_model

.. toctree::
   :maxdepth: 2

   docs/SARIMA_model

.. toctree::
   :maxdepth: 2

   docs/XGB_model

.. toctree::
   :maxdepth: 2

   docs/LSTM_model

.. toctree::
   :maxdepth: 2

   docs/NAIVE_model


Appendix
--------

Here are presented all the parameters that can be given to the argument parser, specifying their function.

.. toctree::
   :maxdepth: 2

   docs/parser_arguments


References
----------
https://www.statsmodels.org/stable/examples/notebooks/generated/statespace_forecasting.html#Cross-validation
https://www.statsmodels.org/stable/generated/statsmodels.tsa.ar_model.AutoRegResults.append.html#statsmodels.tsa.ar_model.AutoRegResults.append

Indices
==================
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
