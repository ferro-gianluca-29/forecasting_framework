import matplotlib.pyplot as plt
from itertools import product
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from tqdm import tqdm
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
<<<<<<< Updated upstream
from statsmodels.tsa.seasonal import MSTL
import matplotlib.pyplot as plt
import pandas as pd

def conditional_print(verbose, *args, **kwargs):
=======
from statsmodels.tsa.seasonal import MSTL, STL, seasonal_decompose
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def conditional_print(verbose, *args, **kwargs):
    """
    Prints messages conditionally based on a verbosity flag.

    :param verbose: Boolean flag indicating whether to print messages.
    :param args: Arguments to be printed.
    :param kwargs: Keyword arguments to be printed.
    """
>>>>>>> Stashed changes
    if verbose:
        print(*args, **kwargs)

def adf_test(df, alpha=0.05, verbose=False):
<<<<<<< Updated upstream
=======
    """
    Performs the Augmented Dickey-Fuller test to determine if a series is stationary and provides detailed output.

    :param df: The time series data as a DataFrame.
    :param alpha: The significance level for the test to determine stationarity.
    :param verbose: Boolean flag that determines whether to print detailed results.
    :return: The number of differences needed to make the series stationary.
    """
>>>>>>> Stashed changes
    d = 0
    adf_result = adfuller(df.dropna())
    p_value = adf_result[1]
    adf_statistic = adf_result[0]
    
    if p_value < alpha and adf_statistic < adf_result[4]['5%']:
        conditional_print(verbose, "The series is stationary.")
        return d

    conditional_print(verbose, 'Stationarity test in progress...\n')
    
    df_diff = df.diff()
    while True:
        adf_result = adfuller(df_diff.dropna())
        
        conditional_print(verbose, f"\nIteration #{d}:\n")
        conditional_print(verbose, "ADF Statistic:", adf_result[0])
        conditional_print(verbose, "p-value:", adf_result[1])
        conditional_print(verbose, "Critical Values:")
        
        for key, value in adf_result[4].items():
            conditional_print(verbose, f"\t{key}: {value}")

        if adf_result[1] < alpha and adf_result[0] < adf_result[4]['5%']:
            conditional_print(verbose, "\nADF test outcome: The series is stationary.\n")
            break
        else:
            d += 1
            df_diff = df_diff.diff()
    
    if verbose:
        print("\n===== ACF and PACF Plots =====")
        plot_acf(df_diff.dropna())
        plt.show()
        
        plot_pacf(df_diff.dropna())
        plt.show()

    return d

def ARIMA_optimizer(train, target_column=None, verbose=False):
<<<<<<< Updated upstream
=======
        """
        Determines the optimal parameters for an ARIMA model based on the Akaike Information Criterion (AIC).

        :param train: The training dataset.
        :param target_column: The target column in the dataset that needs to be forecasted.
        :param verbose: If set to True, prints the process of optimization.
        :return: The best (p, d, q) order for the ARIMA model.
        """
>>>>>>> Stashed changes
        d = adf_test(df=train[target_column], verbose=verbose)

        p = range(0, 5)
        q = range(0, 5)
        griglia_param_ARIMA = list(product(p, [d], q))
        result_df = optimize_ARIMA(train[target_column], griglia_param_ARIMA)
        conditional_print(verbose, result_df)
        best_order = result_df.iloc[0]['(p, d, q)']
        print(f"\nThe optimal parameters for the ARIMA model are: {best_order}\n")
        return best_order

def SARIMAX_optimizer(train, target_column=None, period=None, exog=None, verbose=False):
<<<<<<< Updated upstream
=======
        
        """
        Identifies the optimal parameters for a SARIMAX model.

        :param train: The training dataset.
        :param target_column: The target column in the dataset.
        :param period: The seasonal period of the dataset.
        :param exog: The exogenous variables included in the model.
        :param verbose: Controls the output of the optimization process.
        :return: The best (p, d, q, P, D, Q) parameters for the SARIMAX model.
        """
        
>>>>>>> Stashed changes
        d = adf_test(train[target_column], verbose=verbose)
        D = adf_test(train[target_column].diff(period).dropna(), verbose=verbose)

        p = q = P = Q = range(0, 2)
        griglia_param_SARIMAX = list(product(p, [d], q, P, [D], Q))
<<<<<<< Updated upstream
        result_df = optimize_SARIMAX(train, exog, griglia_param_SARIMAX, period)
=======
        result_df = optimize_SARIMAX(train, griglia_param_SARIMAX, period, exog)
>>>>>>> Stashed changes
        conditional_print(verbose, result_df)
        best_order = result_df.iloc[0]['(p, d, q, P, D, Q)']
        print(f"\nThe optimal parameters for the SARIMAX model are: {best_order}\n")
        return best_order

def optimize_ARIMA(endog, order_list):
<<<<<<< Updated upstream
=======
    """
    Optimizes ARIMA parameters by iterating over a list of (p, d, q) combinations to find the lowest AIC.

    :param endog: The endogenous variable.
    :param order_list: A list of (p, d, q) tuples representing different ARIMA configurations to test.
    :return: A DataFrame containing the AIC scores for each parameter combination.
    """
>>>>>>> Stashed changes
    print("\nOptimizing ARIMA parameters in progress...\n")
    results = []
    
    for order in tqdm(order_list):
        try: 
            model = ARIMA(endog, order=order).fit()
            aic = model.aic
            results.append([order, aic])
        except:
            continue
    result_df = pd.DataFrame(results, columns=['(p, d, q)', 'AIC']).sort_values(by='AIC', ascending=True).reset_index(drop=True)
    return result_df

<<<<<<< Updated upstream
def optimize_SARIMAX(endog, exog, order_list, s):
=======
def optimize_SARIMAX(endog, order_list, s, exog = None):
    """
    Optimizes SARIMAX parameters by testing various combinations and selecting the one with the lowest AIC.

    :param endog: The dependent variable.
    :param order_list: A list of order tuples (p, d, q, P, D, Q) for the SARIMAX.
    :param s: The seasonal period of the model.
    :param exog: Optional exogenous variables.
    :return: A DataFrame with the results of the parameter testing.
    """
>>>>>>> Stashed changes
    print("\nOptimizing SARIMAX parameters in progress...\n")
    results = []
    for order in tqdm(order_list):
        try: 
<<<<<<< Updated upstream
            model = SARIMAX(endog, exog=exog, order=(order[0], order[1], order[2]), seasonal_order=(order[3], order[4], order[5], s)).fit(disp=False)
=======
            if exog is not None:
                model = SARIMAX(endog, exog=exog, order=(order[0], order[1], order[2]), seasonal_order=(order[3], order[4], order[5], s)).fit(disp=False)
            else:
                model = SARIMAX(endog, order=(order[0], order[1], order[2]), seasonal_order=(order[3], order[4], order[5], s)).fit(disp=False)
>>>>>>> Stashed changes
            aic = model.aic
            results.append([order, aic])    
        except:
            continue
    result_df = pd.DataFrame(results, columns=['(p, d, q, P, D, Q)', 'AIC']).sort_values(by='AIC', ascending=True).reset_index(drop=True)
    return result_df

def ljung_box_test(model):
<<<<<<< Updated upstream
=======
        """
        Conducts the Ljung-Box test on the residuals of a fitted time series model to check for autocorrelation.

        :param model: The time series model after fitting to the data.
        """
>>>>>>> Stashed changes
        residuals = model.resid
        lb_test = acorr_ljungbox(residuals, lags=[10], return_df=True)
        lb_pvalue = lb_test['lb_pvalue'].iloc[0]
        if lb_pvalue > 0.05:
<<<<<<< Updated upstream
            return 'Ljung-Box test result:\nNull hypothesis valid: Residuals are uncorrelated\n'
        else:
            return 'Ljung-Box test result:\nNull hypothesis invalid: Residuals are correlated\n'

def multiple_STL(dataframe,target_column):
=======
            print('Ljung-Box test result:\nNull hypothesis valid: Residuals are uncorrelated\n')
        else:
            print('Ljung-Box test result:\nNull hypothesis invalid: Residuals are correlated\n')

def multiple_STL(dataframe,target_column):
    """
    Performs multiple seasonal decomposition using STL on specified periods.

    :param dataframe: The DataFrame containing the time series data.
    :param target_column: The column in the DataFrame to be decomposed.
    """
>>>>>>> Stashed changes
    mstl = MSTL(dataframe[target_column], periods=[24, 24 * 7, 24 * 7 * 4])
    res = mstl.fit()

    fig, ax = plt.subplots(nrows=2, figsize=[10,10])
    res.seasonal["seasonal_24"].iloc[:24*3].plot(ax=ax[0])
    ax[0].set_ylabel(target_column)
    ax[0].set_title("Daily seasonality")

    res.seasonal["seasonal_168"].iloc[:24*7*3].plot(ax=ax[1])
    ax[1].set_ylabel(target_column)
    ax[1].set_title("Weekly seasonality")

    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots(nrows=2, figsize=[10,10])
    res.seasonal["seasonal_168"].iloc[:24*7*3].plot(ax=ax[0])
    ax[0].set_ylabel(target_column)
    ax[0].set_title("Weekly seasonality")

    res.seasonal["seasonal_672"].iloc[:24*7*4*3].plot(ax=ax[1])
    ax[1].set_ylabel(target_column)
    ax[1].set_title("Monthly seasonality")

    plt.tight_layout()
<<<<<<< Updated upstream
    plt.show()
=======
    plt.show()





def prepare_seasonal_sets(train, valid, test, target_column, period):
    """
    Decomposes the datasets into seasonal and residual components based on the specified period.

    :param train: Training dataset.
    :param valid: Validation dataset.
    :param test: Test dataset.
    :param target_column: The target column in the datasets.
    :param period: The period for seasonal decomposition.
    :return: Decomposed training, validation, and test datasets.
    """
    
    # Seasonal and residual components of the training set
    train_seasonal = pd.DataFrame(seasonal_decompose(train[target_column], model='additive', period=period).seasonal) # Assuming daily seasonality.seasonal)
    train_seasonal.rename(columns = {'seasonal': target_column}, inplace = True)
    train_seasonal = train_seasonal.dropna()
    train_residual = pd.DataFrame(seasonal_decompose(train[target_column], model='additive', period=period).resid)
    train_residual.rename(columns = {'resid': target_column}, inplace = True)
    train_residual = train_residual.dropna()
    # Seasonal and residual components of the validation set
    valid_seasonal = pd.DataFrame(seasonal_decompose(valid[target_column], model='additive', period=period).seasonal)
    valid_seasonal.rename(columns = {'seasonal': target_column}, inplace = True)
    valid_seasonal = valid_seasonal.dropna()
    valid_residual = pd.DataFrame(seasonal_decompose(valid[target_column], model='additive', period=period).resid)
    valid_residual.rename(columns = {'resid': target_column}, inplace = True)
    valid_residual = valid_residual.dropna()
    # Seasonal and residual components of the test set
    test_seasonal = pd.DataFrame(seasonal_decompose(test[target_column], model='additive', period=period).seasonal)
    test_seasonal.rename(columns = {'seasonal': target_column}, inplace = True)
    test_seasonal = test_seasonal.dropna()
    test_residual = pd.DataFrame(seasonal_decompose(test[target_column], model='additive', period=period).resid)
    test_residual.rename(columns = {'resid': target_column}, inplace = True)
    test_residual = test_residual.dropna()

    # Merge residual and seasonal components on indices with 'inner' join to keep only matching rows
    train_merge = pd.merge(train_residual, train_seasonal, left_index=True, right_index=True, how='inner')
    valid_merge = pd.merge(valid_residual, valid_seasonal, left_index=True, right_index=True, how='inner')
    test_merge = pd.merge(test_residual, test_seasonal, left_index=True, right_index=True, how='inner')
    
    # Add the residual and seasonal columns
    train_decomposed = pd.DataFrame(train_merge.iloc[:,0] + train_merge.iloc[:,1])
    train_decomposed = train_decomposed.rename(columns = {train_decomposed.columns[0]: target_column})
    valid_decomposed = pd.DataFrame(valid_merge.iloc[:,0] + valid_merge.iloc[:,1])
    valid_decomposed = valid_decomposed.rename(columns = {valid_decomposed.columns[0]: target_column})
    test_decomposed = pd.DataFrame(test_merge.iloc[:,0] + test_merge.iloc[:,1])
    test_decomposed = test_decomposed.rename(columns = {test_decomposed.columns[0]: target_column})

    return train_decomposed, valid_decomposed, test_decomposed

def time_s_analysis(df, target_column, seasonal_period):
    """
    Performs a comprehensive time series analysis including plotting, stationarity testing, and decomposition.

    :param df: The DataFrame containing the time series data.
    :param target_column: The column in the DataFrame representing the time series to analyze.
    :param seasonal_period: The period to consider for seasonal decomposition and autocorrelation analysis.
    """

    # Plot the time series
    plt.plot(df.index, df[target_column], 'b')
    plt.title('Time Series')
    plt.xlabel('Time series index')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

    # ADF test for stationarity
    #df[target_column] = df[target_column].str.replace(',', '.').astype(float)
    adf_result = adfuller(df[target_column].dropna())
    p_value = adf_result[1]
    adf_statistic = adf_result[0]
    alpha = 0.05
   
    if p_value < alpha and adf_statistic < adf_result[4]['5%']:
        print("The series is stationary.")
    else:
        print("The series is not stationary.")

    # ACF and PACF plots
    print("\n===== ACF and PACF Plots =====")
    fig, ax = plt.subplots()
    plot_acf(df[target_column].dropna(), lags = seasonal_period + 4, ax=ax)
    ax.set_xlabel('Lags')  # Imposta l'etichetta dell'asse X
    ax.set_ylabel('Autocorrelation')  # Imposta l'etichetta dell'asse Y
    plt.show()

    fig, ax = plt.subplots()
    plot_pacf(df[target_column].dropna(), lags = seasonal_period + 4, ax=ax)
    ax.set_xlabel('Lags')  # Imposta l'etichetta dell'asse X
    ax.set_ylabel('Partial Autocorrelation')  # Imposta l'etichetta dell'asse Y
    plt.show()

    df_diff = df.diff()

    # ACF and PACF plots od differentiated time series
    print("\n===== ACF and PACF Plots =====")
    fig, ax = plt.subplots()
    plot_acf(df_diff[target_column].dropna(), lags = seasonal_period + 4, ax=ax)
    ax.set_xlabel('Lags')  # Imposta l'etichetta dell'asse X
    ax.set_ylabel('Autocorrelation')  # Imposta l'etichetta dell'asse Y
    ax.set_title('Differentiated Series ACF')  # Imposta il titolo del plot
    plt.show()

    fig, ax = plt.subplots()
    plot_pacf(df_diff[target_column].dropna(), lags = seasonal_period + 4, ax=ax)
    ax.set_xlabel('Lags')  # Imposta l'etichetta dell'asse X
    ax.set_ylabel('Partial Autocorrelation')  # Imposta l'etichetta dell'asse Y
    ax.set_title('Differentiated Series PACF')  # Imposta il titolo del plot
    plt.show()

    """

    # Time series decomposition into its trend, seasonality, and residuals components
    decomposition = STL(df[target_column], period=seasonal_period).fit()
    
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, ncols=1, sharex=True, figsize=(10,8))

    ax1.plot(decomposition.observed)
    ax1.set_ylabel('Observed')

    ax2.plot(decomposition.trend)
    ax2.set_ylabel('Trend')

    ax3.plot(decomposition.seasonal)
    ax3.set_ylabel('Seasonal')

    ax4.plot(decomposition.resid)
    ax4.set_ylabel('Residuals')

    fig.autofmt_xdate()
    plt.tight_layout()
    # Add title
    plt.suptitle(f"Time Series Decomposition with period {seasonal_period}")
    plt.show()
    """
>>>>>>> Stashed changes
