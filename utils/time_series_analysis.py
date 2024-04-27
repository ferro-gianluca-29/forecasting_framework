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
from statsmodels.tsa.seasonal import MSTL, seasonal_decompose
import matplotlib.pyplot as plt
import pandas as pd

def conditional_print(verbose, *args, **kwargs):
    if verbose:
        print(*args, **kwargs)

def adf_test(df, alpha=0.05, verbose=False):
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
        d = adf_test(train[target_column], verbose=verbose)
        D = adf_test(train[target_column].diff(period).dropna(), verbose=verbose)

        p = q = P = Q = range(0, 2)
        griglia_param_SARIMAX = list(product(p, [d], q, P, [D], Q))
        result_df = optimize_SARIMAX(train, exog, griglia_param_SARIMAX, period)
        conditional_print(verbose, result_df)
        best_order = result_df.iloc[0]['(p, d, q, P, D, Q)']
        print(f"\nThe optimal parameters for the SARIMAX model are: {best_order}\n")
        return best_order

def optimize_ARIMA(endog, order_list):
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

def optimize_SARIMAX(endog, exog, order_list, s):
    print("\nOptimizing SARIMAX parameters in progress...\n")
    results = []
    for order in tqdm(order_list):
        try: 
            model = SARIMAX(endog, exog=exog, order=(order[0], order[1], order[2]), seasonal_order=(order[3], order[4], order[5], s)).fit(disp=False)
            aic = model.aic
            results.append([order, aic])    
        except:
            continue
    result_df = pd.DataFrame(results, columns=['(p, d, q, P, D, Q)', 'AIC']).sort_values(by='AIC', ascending=True).reset_index(drop=True)
    return result_df

def ljung_box_test(model):
        residuals = model.resid
        lb_test = acorr_ljungbox(residuals, lags=[10], return_df=True)
        lb_pvalue = lb_test['lb_pvalue'].iloc[0]
        if lb_pvalue > 0.05:
            return 'Ljung-Box test result:\nNull hypothesis valid: Residuals are uncorrelated\n'
        else:
            return 'Ljung-Box test result:\nNull hypothesis invalid: Residuals are correlated\n'

def multiple_STL(dataframe,target_column):
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
    plt.show()

def moving_average_ST(dataframe,target_column):
    "Seasonal-Trend decomposition using moving averages"

    result = seasonal_decompose(dataframe[target_column], model='additive', period=24) # Assuming daily seasonality
    seasonal = result.seasonal
    trend = result.trend
    residual = result.resid
    # Plot the original time series data
    plt.figure(figsize=(16, 8))
    plt.subplot(4, 1, 1)
    plt.plot(dataframe[target_column], label='Original')
    plt.legend(loc='best')
    plt.title('Original Time Series Data')

    # Plot the trend component
    plt.subplot(4, 1, 2)
    plt.plot(result.trend, label='Trend')
    plt.legend(loc='best')
    plt.title('Trend Component')

    # Plot the seasonal component
    plt.subplot(4, 1, 3)
    plt.plot(result.seasonal, label='Seasonal')
    plt.legend(loc='best')
    plt.title('Seasonal Component')

    # Plot the residual component
    plt.subplot(4, 1, 4)
    plt.plot(result.resid, label='Residual')
    plt.legend(loc='best')
    plt.title('Residual Component')

    # Show the plot
    plt.tight_layout()
    plt.show()


