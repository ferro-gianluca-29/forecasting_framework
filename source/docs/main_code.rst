Main Code
=================

.. automodule:: main_code
   :members:
   :undoc-members:
   :show-inheritance:

The main code is divided into the following sections: 

* **Data loading**
Loads data from the specified dataset path using the DataLoader class. 

* **Preprocessing and dataset split** 
Handling data preprocessing and dataset splitting based on the specified model type and runtime configurations.
For test-only run mode, only the test set will be created.

* **Optional time series analysis**
Performs additional time series analysis if specified with the parser argument `--ts_analysis`.
Functions used in this section include plotting the ACF and PACF diagrams to have an early estimate of the AR and MA parameters, 
as well as statistic tests for stationarity. For seasonal models, a seasonal-trend decomposition of the series can be performed. 

* **Model loading for test or fine-tuning**
Depending on the run mode, loads a pre-trained model to perform fine-tuning or to handle the test-only mode.
For statistical models, some operations on the index of the training and test set are required, to ensure that the endogenous 
and exogenous variables passed to the model during training have indexes that are contiguous to those of the pre-trained model.

* **Model training**
The training of the model is done using the corresponding method of the `ModelTraining` class.
A buffer containing the last `buffer_size` elements of the training set is created, which can be used for 
further test or to create naive benchmark models. After the training phase data (including the model itself and validation metrics) 
are saved in the model's directory. 

* **Model testing**
The class ModelTest is used in order to generate the predictions of the trained model.

* **Plot of predictions versus real data**

* **Performance measurement and saving**
After computing the metrics of the model, the performance is stored into a text file.
