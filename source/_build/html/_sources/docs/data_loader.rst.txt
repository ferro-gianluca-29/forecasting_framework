Data Loader
=================

This module contains the `DataLoader` class, specifically crafted to load and preprocess datasets 
for various types of machine learning models, including LSTM, XGB, ARIMA and SARIMA. 
The class efficiently handles datasets from multiple file formats and prepares them for model-specific requirements. 
It is essential to specify the correct date format present in the dataset using the `--date_format` parser argument 
to avoid errors during data loading. 
If the date column is not the first column in the dataset, its index must be specified using the `--time_column_index` command 
to ensure accurate processing.
The `load_data` method converts date columns to datetime objects and adjusts dataset indices to align with the chosen model type. 
Additionally, the class utilizes specific dates provided through the `--date_list` input, filtering and structuring the data accordingly. 


.. automodule:: data_loader
   :members:
   :undoc-members:
   :show-inheritance:
