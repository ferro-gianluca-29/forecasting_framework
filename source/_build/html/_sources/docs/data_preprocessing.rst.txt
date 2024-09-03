Data Preprocessing 
=================

This module is equipped with the `DataPreprocessor` class, specifically designed for the preprocessing of time series data 
for machine learning models. 
A method for data splitting is present, that uses processed dates coming from the `DataLoader` object.
The class contains also specific methods for tasks such as managing missing values, 
removing non-numeric columns, managing outliers, and appropriately scaling data.
These methods are designed taking into account the sequential nature of time series data, providing moving windows for outlier detection
and making sure that, if a dataset with a large sequence of NaN is detected, the code stops working due to the lack of 
useful data. Also, data scaling is applied to the test set by using statistics from the training set, avoiding thus data leakage.
The class supports various operational modes like training, testing, and fine-tuning.

.. automodule:: data_preprocessing
   :members:
   :undoc-members:
   :show-inheritance:

