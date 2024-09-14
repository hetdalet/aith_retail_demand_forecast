import os
import re
import json
import joblib
import optuna
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import product

from prophet import Prophet
from sklearn.model_selection import train_test_split
from pandas.tseries.holiday import USFederalHolidayCalendar
from holidays.holiday_base import HolidayBase

from typing import Optional
from catboost import CatBoostRegressor, Pool

from typing import Optional




class suppress_stdout_stderr(object):
    """
    A context manager for doing a "deep suppression" of stdout and stderr in
    Python, i.e. will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function.
       This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).

    """

    def __init__(self):
        # Open a pair of null files
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = (os.dup(1), os.dup(2))

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        # Close the null files
        os.close(self.null_fds[0])
        os.close(self.null_fds[1])

class ProphetsEnsemble:
    """An ensemble of Prophet models with different aggregation functions and frequencies."""

    def __init__(self, freq: str, levels: list, agg_fn: list, holidays_getter: HolidayBase = None):
        """Initializes an ensemble of Prophet models."""
        self.freq = freq
        self.levels = ['_'.join(x) for x in product(levels, agg_fn)]
        self.h_getter = holidays_getter
        self.prophets_ = dict()
        self.is_fitted_ = False
    
    @staticmethod
    def _resample(data: pd.DataFrame, freq: str, how: str) -> pd.DataFrame:
        """Resamples a time series DataFrame."""
        if how not in ['median', 'mean', 'sum']:
            raise NotImplementedError(f'Unknown function {how}. Only [median, mean, sum] are supported.') 
        return data.set_index('ds').resample(freq).agg(how).reset_index(drop=False)

    @staticmethod
    def _merge_key_gen(x, level: str) -> str:
        """Generates a key for merging DataFrames based on the frequency."""
        freq = re.sub('[\d]', '', level.split('_')[0])
        if freq == 'H':
            return f'{x.year}-{x.month}-{x.day}-{x.hour}'
        elif freq in ['D', 'M']:
            return f'{x.year}-{x.month}-{x.day}' if freq == 'D' else f'{x.year}-{x.month}'
        elif freq == 'W':
            return f'{x.isocalendar().year}-{x.isocalendar().week}'
        raise NotImplementedError(f'Only [H, D, W, M] are supported. {freq} was received as input!')
    
    def _get_holidays(self, data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Extracts holidays from the data."""
        if self.h_getter is None:
            return None
        holidays = data[['ds']].copy()
        holidays['holiday'] = holidays['ds'].apply(self.h_getter.get)
        return holidays.dropna()
    
    def _fit_level(self, data: pd.DataFrame, level: str) -> None:
        """Fits a Prophet model for a specific aggregation level."""
        resampled = self._resample(data, *level.split('_')) if level != self.freq else data.copy()
        fb = Prophet(holidays=self._get_holidays(resampled))
        fb.add_regressor('sell_price')
        fb.add_regressor('cashback')
        with suppress_stdout_stderr():
            fb.fit(resampled)
        self.prophets_[level] = fb
        
    def _predict_level(self, periods: int, level: str, future_regressors: Optional[pd.DataFrame]) -> pd.DataFrame:
        """Makes predictions for a specific aggregation level."""
        fb = self.prophets_[level]
        df = fb.make_future_dataframe(periods=periods, freq=level.split('_')[0])
        if future_regressors is not None:
            df = df.merge(future_regressors, on='ds', how='left')
            # Ensure no NaNs in the future data
            df['sell_price'] = df['sell_price'].fillna(df['sell_price'].mean())
            df['cashback'] = df['cashback'].fillna(0)
        forecasts = fb.predict(df)
        forecasts.columns = [f'{x}_{level}' for x in forecasts.columns]
        return forecasts
    
    def _combine_levels(self, base_df: pd.DataFrame, data: pd.DataFrame, level: str) -> pd.DataFrame:
        """Combines predictions from different aggregation levels."""
        key = lambda x: self._merge_key_gen(x, level)
        return (
            base_df.assign(key=base_df['ds'].apply(key))
            .merge(data.assign(key=data[f'ds_{level}'].apply(key)), on='key', how='left')
            .drop(['key', f'ds_{level}'], axis=1)
        )
    
    @staticmethod
    def _drop_redundant(data: pd.DataFrame) -> pd.DataFrame:
        """Drops redundant features from the DataFrame."""
        redundant = [col for col in data.columns if col != 'ds' and 'yhat' not in col and len(data[col].unique()) == 1]
        return data.drop(redundant, axis=1)
    
    def fit(self, data: pd.DataFrame) -> None:
        """Fits the Prophet models for all aggregation levels."""
        for level in tqdm([self.freq] + self.levels, 'Fitting prophets...'):
            self._fit_level(data, level)
        self.is_fitted_ = True
            
    def forecast(self, periods: int, future_regressors: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Makes forecasts for all aggregation levels and combines them."""
        assert self.is_fitted_, 'Model is not fitted'
        forecasts = [
            self._predict_level(periods, level, future_regressors) 
            for level in tqdm([self.freq] + self.levels, 'Forecasting...')
        ]
        
        forecast = forecasts[0].rename(columns={f'ds_{self.freq}': 'ds', f'yhat_{self.freq}': 'yhat'})
        for level, fore in zip(self.levels, forecasts[1:]):
            forecast = self._combine_levels(forecast, fore, level)
            
        return self._drop_redundant(forecast)


class SingletonMeta(type):

    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


class OptunaCatBoostRegressor:
    """
    A wrapper class for the CatBoost Regressor with Optuna for hyperparameters tuning
    """

    def __init__(
            self,
            n_estimators: int,
            learning_rate: float = 0.01,
            metric: str = 'RMSE',
            cat_columns: Optional[list] = None,  # Changed from 'auto' to None as default
            seed: int = 42
    ):
        """
        Initializes a new instance of the OptunaCatBoostRegressor class
        """
        self.params = {
            "iterations": n_estimators,
            "objective": "RMSE",
            "eval_metric": metric,
            "learning_rate": learning_rate,
            "random_seed": seed,
            "logging_level": "Silent",
            "cat_features": cat_columns  # Removed the automatic handling of 'auto'
        }
        self.cat_columns = cat_columns
        self.model = None
        self.features = None
        self.is_fitted_ = False

    def _to_datasets(
            self, x_train: pd.DataFrame, y_train: np.ndarray, x_val: pd.DataFrame, y_val: np.ndarray
    ) -> (Pool, Pool):
        """
        Converts Pandas DataFrames to CatBoost Pools
        """
        # Ensure there are no duplicate features
        self.features = list(x_train.columns)  # Changed from combining with sell_price, cashback
        X_val = x_val[self.features].copy()
        
        # Ensure cat_features is handled correctly
        cat_features = self.cat_columns
        if cat_features is None:  # Automatically detect categorical features if not provided
            cat_features = [col for col in x_train.columns if x_train[col].dtype == 'object' or x_train[col].dtype.name == 'category']

        dtrain = Pool(data=x_train, label=y_train, cat_features=cat_features)
        dval = Pool(data=X_val, label=y_val, cat_features=cat_features)

        return dtrain, dval

    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray, X_val: pd.DataFrame, y_val: np.ndarray) -> None:
        dtrain, dval = self._to_datasets(X_train, y_train, X_val, y_val)

        def objective(trial):
            param = {
                "iterations": self.params["iterations"],
                "learning_rate": trial.suggest_loguniform("learning_rate", 0.01, 0.3),
                "depth": trial.suggest_int("depth", 4, 10),
                "l2_leaf_reg": trial.suggest_loguniform("l2_leaf_reg", 1e-8, 10.0),
                "random_seed": self.params["random_seed"],
                "eval_metric": self.params["eval_metric"],
                "cat_features": self.params["cat_features"],
                "logging_level": "Silent"
            }

            model = CatBoostRegressor(**param)
            model.fit(dtrain, eval_set=dval, early_stopping_rounds=150, verbose=False)
            preds = model.predict(dval)
            return np.sqrt(((y_val - preds) ** 2).mean())

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=100, timeout=600)

        self.model = CatBoostRegressor(**self.params)
        self.model.set_params(**study.best_params)
        self.model.fit(dtrain, eval_set=dval, early_stopping_rounds=150, verbose=False)

        self.is_fitted_ = True

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        assert self.is_fitted_, 'Model is not fitted!'
        return self.model.predict(X_test[self.features])

def get_random_subfolder(store_dir):
    subfolders = [f.name for f in os.scandir(store_dir) if f.is_dir()]
    return random.choice(subfolders) if subfolders else None

class ProphetCatboost:

    name = 'catboost_prophet'

    def __init__(self, 
                 sales_path='ml/catboost_prophet/base_data/shop_sales.csv',
                 dates_path='ml/catboost_prophet/base_data/shop_sales_dates.csv',
                 prices_path='ml/catboost_prophet/base_data/shop_sales_prices.csv',
                 models_path='/Users/dtikhanovskii/Documents/AiTalentHack_RetailDemandForecast/ml/catboost_prophet/models',
                 n_estimators=1000, learning_rate=0.02, metric='RMSE', seed=42):
        self.sales_data_with_prices, self.sales_dates_data = self._preprocess_files(sales_path, dates_path, prices_path)
        self.models_path = models_path
        self.n_estimators=n_estimators
        self.learning_rate=learning_rate
        self.metric=metric
        self.seed=seed
    
    @staticmethod
    def _preprocess_files(sales_path, dates_path, prices_path):
        sales_data = pd.read_csv(sales_path)
        sales_dates_data = pd.read_csv(dates_path)
        sales_prices_data = pd.read_csv(prices_path)
        sales_data['true_item_id'] = sales_data['item_id'].apply(lambda x: '_'.join(x.split('_')[2:]))
        sales_prices_data['true_item_id'] = sales_prices_data['item_id'].apply(lambda x: '_'.join(x.split('_')[2:]))
        sales_data = pd.merge(sales_data, sales_dates_data[['date_id', 'wm_yr_wk']], on='date_id', how='left')
        sales_data_with_prices = pd.merge(sales_data, sales_prices_data, on=['store_id', 'true_item_id', 'wm_yr_wk'], how='left')
        sales_data_with_prices = pd.merge(sales_data_with_prices,
                                      sales_dates_data[['date_id', 'date', 'CASHBACK_STORE_1', 'CASHBACK_STORE_2', 'CASHBACK_STORE_3']],
                                      on='date_id', how='left')

        return sales_data_with_prices, sales_dates_data

    def _get_item_store_data(self, item_id, store_id):
        cashback_column = f'CASHBACK_{store_id}'
        sales_data_with_prices = self.sales_data_with_prices
        sales_data_with_prices['cashback'] = sales_data_with_prices[cashback_column]
        # Filter data for the specific item and store
        item_store_data = sales_data_with_prices[(self.sales_data_with_prices['true_item_id'] == item_id) & (sales_data_with_prices['store_id'] == store_id)]
        item_store_data['date'] = pd.to_datetime(item_store_data['date'])
        item_store_data = item_store_data[['date', 'cnt', 'sell_price', 'cashback']]
    
        # Fill missing values in 'sell_price' using forward fill, then backfill as a safety net
        sales_data_with_prices['sell_price'] = sales_data_with_prices['sell_price'].ffill().bfill()
        # Fill any remaining NaNs in 'sell_price' with the mean value of the entire column
        sales_data_with_prices['sell_price'] = sales_data_with_prices['sell_price'].fillna(sales_data_with_prices['sell_price'].mean())
    
        # Ensure all NaNs in 'cashback' are filled with 0
        sales_data_with_prices['cashback'] = sales_data_with_prices['cashback'].fillna(0)
    
        # Filter data for the specific item and store
        item_store_data = sales_data_with_prices[(sales_data_with_prices['true_item_id'] == item_id) & (sales_data_with_prices['store_id'] == store_id)]
        item_store_data['date'] = pd.to_datetime(item_store_data['date'])
        item_store_data = item_store_data[['date', 'cnt', 'sell_price', 'cashback']]

        # Ensure no NaN values are present after filtering
        item_store_data['sell_price'] = item_store_data['sell_price'].fillna(item_store_data['sell_price'].mean())
        item_store_data['cashback'] = item_store_data['cashback'].fillna(0)

        return item_store_data
    
    def predict(self, item_id, store_id, steps=30, future_df=None):
        data = self._get_item_store_data(item_id, store_id)
        model_dir = os.path.join(self.models_path, store_id, item_id)
        store_dir = os.path.join(self.models_path, store_id)
        model_dir = os.path.join(store_dir, item_id)
    
        if not os.path.exists(model_dir):
            random_item_id = get_random_subfolder(store_dir)
            if random_item_id:
                model_dir = os.path.join(store_dir, random_item_id)
            else:
                raise FileNotFoundError(f"Cold start store -  {store_dir}.")
    
        prophet_model_file = os.path.join(model_dir, f'prophet_model.pkl')
        catboost_model_file = os.path.join(model_dir, f'catboost_model.pkl')

        end_date = data.date.max()

        if future_df is None:
            future_regressor = pd.DataFrame({
                'ds': pd.date_range(start=end_date, periods=steps, freq='D'),
                'sell_price': [data['sell_price'].mean()] * steps,
                'cashback': [0] * steps
            })
        else:
            predict_end = future_df.ds.max()
            predict_start = future_df.ds.min()
            date_range = pd.date_range(start=min(predict_start, end_date), end=predict_end)
            future_regressor = pd.DataFrame(date_range, columns=['date'])
            mean_sell_price = data['sell_price'].mean()
            future_regressor = future_regressor.merge(future_df.rename(columns={'ds': 'date'}), on='date', how='left')
            future_regressor['sell_price'].fillna(mean_sell_price, inplace=True)
            future_regressor['cashback'].fillna(0, inplace=True)
            future_regressor = future_regressor.rename(columns={'date': 'ds'})

        print(future_regressor)
        prophet_model = joblib.load(prophet_model_file)
        future_forecast = prophet_model.forecast(len(future_regressor), future_regressors=future_regressor)
        new_data_with_forecast = future_regressor.merge(future_forecast, left_on='ds', right_on='ds', how='left')
        catboost_model = joblib.load(catboost_model_file)
        predictions = catboost_model.predict(new_data_with_forecast[catboost_model.features])
        if future_df is not None:
            predictions = predictions[-len(future_df):]
        return predictions
    
    def _fit_one_user_store(self, item_id, store_id, models_path):
        data = self._get_item_store_data(item_id, store_id)
        
        # Generate holidays within the date range
        cal = USFederalHolidayCalendar()
        holidays = cal.holidays(start=data['date'].min(), end=data['date'].max())
        
        # Initialize the ProphetsEnsemble with holiday information
        pe = ProphetsEnsemble(
            freq='D', 
            levels=['W', 'M'], 
            agg_fn=['median'], 
            holidays_getter=pd.DataFrame({'ds': holidays, 'holiday': 'us_holiday'})
        )
        
        # Rename columns to fit Prophet's expected input format
        prophet_data = data.rename(columns={'date': 'ds', 'cnt': 'y'})
        
        # Fit the Prophet ensemble model to all data
        pe.fit(prophet_data)
        
        # Prepare future data for forecasting, including regressors
        future_dates = pd.DataFrame({
            'ds': pd.date_range(start=data['date'].max(), periods=30, freq='D'),  # Adjust the period as needed
            'sell_price': [data['sell_price'].mean()] * 30,  # Using mean sell_price for future
            'cashback': [0] * 30  # Setting cashback to 0 for future
        })
    
        # Forecast using the ensemble model
        pe_forecast = pe.forecast(len(future_dates), future_regressors=future_dates)
        
        # Merge forecast with the original data for CatBoost training
        gbt_data = prophet_data.merge(pe_forecast, on='ds', how='left')
    
        # Split a small portion of the data for validation
        train_gbt, val_gbt = train_test_split(gbt_data, test_size=0.1, random_state=42)
        
        # Train CatBoost model using the ensemble forecast as input features
        catboost = OptunaCatBoostRegressor(n_estimators=self.n_estimators, learning_rate=self.learning_rate, metric=self.metric, seed=self.seed)
        catboost.fit(
            X_train=train_gbt.drop(['ds', 'y'], axis=1), 
            y_train=train_gbt['y'].values,
            X_val=val_gbt.drop(['ds', 'y'], axis=1), 
            y_val=val_gbt['y'].values
        )
        
        # Define model directory and save the models
        model_dir = os.path.join(models_path, store_id, item_id)
        os.makedirs(model_dir, exist_ok=True)
        
        prophet_model_file = os.path.join(model_dir, 'prophet_model.pkl')
        catboost_model_file = os.path.join(model_dir, 'catboost_model.pkl')
        
        joblib.dump(pe, prophet_model_file)
        joblib.dump(catboost, catboost_model_file)
        
    def fit(self, models_path=None, sales_path=None, dates_path=None, prices_path=None):
        if sales_path is not None:
            self.sales_data_with_prices, self.sales_dates_data = self._preprocess_files(sales_path, dates_path, prices_path)
        if models_path is not None:
            self.models_path = models_path
        unique_tuples_df = self.sales_data_with_prices[['true_item_id', 'store_id']].drop_duplicates(subset=['true_item_id', 'store_id'])

        unique_tuples = list(unique_tuples_df.itertuples(index=False, name=None))
        for item_id, store_id in unique_tuples:
            self._fit_one_user_store(item_id, store_id, self.models_path)

    def run(self, json_input):
        data = json.loads(json_input)
        item_info = data[0]
        item_id = item_info['item_id']
        store_id = item_info['store_id']

        sales_data = data[1:]
        df = pd.DataFrame(sales_data)
        list_of_strings = [f"{x:.1f}" for x in self.predict(item_id, store_id, future_df=df)]
        return json.dumps(list_of_strings)
        

model = ProphetCatboost()
