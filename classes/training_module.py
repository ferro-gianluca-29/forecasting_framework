from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from utils.time_series_analysis import ARIMA_optimizer, SARIMAX_optimizer, ljung_box_test
from prophet import Prophet
import warnings
warnings.filterwarnings("ignore")

class ModelTraining():
    def __init__(self, model_type: str, train, target_column = None, 
                 verbose = False):
        
        self.verbose = verbose
        self.model_type = model_type
        self.train = train
        self.target_column = target_column
        self.ARIMA_order = []
        self.SARIMAX_order = []
        
    def train_ARIMA_model(self): 
        
        try:
            best_order = ARIMA_optimizer(self.train, self.target_column, self.verbose)
            self.ARIMA_order = best_order
            print("\nTraining the ARIMA model...")

            # Training the model with the best parameters found
            model = ARIMA(self.train[self.target_column], order=(best_order[0], best_order[1], best_order[2]))                
            model_fit = model.fit()

            # Running the LJUNG-BOX test for residual correlation
            ljung_box_test(model_fit)
            print("Model successfully trained.")
            return model_fit

        except Exception as e:
            print(f"An error occurred during the model training: {e}")
            return None       
        
    def train_SARIMAX_model(self, target_train, exog_train, period): 
        try:        
            target_train = self.train[[self.target_column]]
            best_order = SARIMAX_optimizer(target_train, self.target_column, period, exog_train, self.verbose)
            self.SARIMAX_order = best_order
            print("\nTraining the SARIMAX model...")

            # Training the model with the best parameters found
            model = SARIMAX(target_train, exog_train, order = (best_order[0], best_order[1], best_order[2]),
                                seasonal_order=(best_order[3], best_order[4], best_order[5], period),
                                simple_differencing=False
                                )
            model_fit = model.fit()
        
            # Running the LJUNG-BOX test for residual correlation
            ljung_box_test(model_fit)
            print("Model successfully trained.")
            return model_fit

        except Exception as e:
            print(f"An error occurred during the model training: {e}")
            return None           

