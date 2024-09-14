import os
import re
import json
import random
import pandas as pd
import numpy as np
from itertools import product
from pandas.tseries.holiday import USFederalHolidayCalendar
from sklearn.model_selection import train_test_split

# Import the classes from the new modules
from prophets_ensemble import ProphetsEnsemble
from optuna_catboost_regressor import OptunaCatBoostRegressor

def get_random_subfolder(store_dir):
    subfolders = [f.name for f in os.scandir(store_dir) if f.is_dir()]
    return random.choice(subfolders) if subfolders else None

class ProphetCatboost:

    name = 'catboost_prophet'

    def __init__(self, 
                 sales_path='ml/catboost_prophet/base_data/shop_sales.csv',
                 dates_path='ml/catboost_prophet/base_data/shop_sales_dates.csv',
                 prices_path='ml/catboost_prophet/base_data/shop_sales_prices.csv',
                 models_path='ml/catboost_prophet/models',
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
        sales_data_with_prices = self.sales_data_with_prices.copy()
        sales_data_with_prices['cashback'] = sales_data_with_prices[cashback_column]
        # Filter data for the specific item and store
        item_store_data = sales_data_with_prices[(sales_data_with_prices['true_item_id'] == item_id) & (sales_data_with_prices['store_id'] == store_id)]
        item_store_data['date'] = pd.to_datetime(item_store_data['date'])
        item_store_data = item_store_data[['date', 'cnt', 'sell_price', 'cashback']]
    
        # Fill missing values in 'sell_price' using forward fill, then backfill as a safety net
        item_store_data['sell_price'] = item_store_data['sell_price'].ffill().bfill()
        # Fill any remaining NaNs in 'sell_price' with the mean value of the entire column
        item_store_data['sell_price'] = item_store_data['sell_price'].fillna(item_store_data['sell_price'].mean())
    
        # Ensure all NaNs in 'cashback' are filled with 0
        item_store_data['cashback'] = item_store_data['cashback'].fillna(0)
    
        return item_store_data

    def predict(self, item_id, store_id, steps=30, future_df=None):
        data = self._get_item_store_data(item_id, store_id)
        store_dir = os.path.join(self.models_path, store_id)
        model_dir = os.path.join(store_dir, item_id)
    
        if not os.path.exists(model_dir):
            random_item_id = get_random_subfolder(store_dir)
            if random_item_id:
                model_dir = os.path.join(store_dir, random_item_id)
            else:
                raise FileNotFoundError(f"Cold start store - {store_dir}.")
    
        prophet_model_file = os.path.join(model_dir, f'prophet_model.json')
        catboost_model_file = os.path.join(model_dir, f'catboost_model.cbm')
    
        end_date = data.date.max()
    
        if future_df is None:
            future_regressor = pd.DataFrame({
                'ds': pd.date_range(start=end_date + pd.Timedelta(days=1), periods=steps, freq='D'),
                'sell_price': [data['sell_price'].mean()] * steps,
                'cashback': [0] * steps
            })
        else:
            predict_end = future_df.ds.max()
            predict_start = future_df.ds.min()
            date_range = pd.date_range(start=min(predict_start, end_date + pd.Timedelta(days=1)), end=predict_end)
            future_regressor = pd.DataFrame(date_range, columns=['date'])
            mean_sell_price = data['sell_price'].mean()
            future_regressor = future_regressor.merge(future_df.rename(columns={'ds': 'date'}), on='date', how='left')
            future_regressor['sell_price'].fillna(mean_sell_price, inplace=True)
            future_regressor['cashback'].fillna(0, inplace=True)
            future_regressor = future_regressor.rename(columns={'date': 'ds'})
    
        prophet_model = ProphetsEnsemble.load(prophet_model_file)
        future_forecast = prophet_model.forecast(len(future_regressor), future_regressors=future_regressor, fill_data={'sell_price': data['sell_price'].mean()})
        new_data_with_forecast = future_regressor.merge(future_forecast, on='ds', how='left')
        catboost_model = OptunaCatBoostRegressor.load(catboost_model_file)
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
            'ds': pd.date_range(start=data['date'].max() + pd.Timedelta(days=1), periods=30, freq='D'),  # Adjust the period as needed
            'sell_price': [data['sell_price'].mean()] * 30,  # Using mean sell_price for future
            'cashback': [0] * 30  # Setting cashback to 0 for future
        })
        # Forecast using the ensemble model
        pe_forecast = pe.forecast(len(future_dates), future_regressors=future_dates, fill_data={'sell_price': data['sell_price'].mean()})
        
        # Merge forecast with the original data for CatBoost training
        gbt_data = prophet_data.merge(pe_forecast, on='ds', how='left')
    
        # Split a small portion of the data for validation
        train_gbt, val_gbt = train_test_split(gbt_data, test_size=0.1, random_state=42)
        
        # Train CatBoost model using the ensemble forecast as input features
        catboost = OptunaCatBoostRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            metric=self.metric,
            seed=self.seed
        )
        catboost.fit(
            X_train=train_gbt.drop(['ds', 'y'], axis=1), 
            y_train=train_gbt['y'].values,
            X_val=val_gbt.drop(['ds', 'y'], axis=1), 
            y_val=val_gbt['y'].values
        )
        
        # Define model directory and save the models
        model_dir = os.path.join(models_path, store_id, item_id)
        os.makedirs(model_dir, exist_ok=True)
        
        prophet_model_file = os.path.join(model_dir, 'prophet_model.json')
        catboost_model_file = os.path.join(model_dir, 'catboost_model.cbm')
        
        pe.save(prophet_model_file)
        catboost.save(catboost_model_file)
        
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
        df['ds'] = pd.to_datetime(df['ds'])
        df['sell_price'] = pd.to_numeric(df['sell_price']) + 0.0001
        df['cashback'] = pd.to_numeric(df['cashback']) + 0.0000000001
        df = df[['ds', 'sell_price', 'cashback']]
        print(df)
        list_of_strings = [f"{x:.1f}" for x in self.predict(item_id, store_id, future_df=df)]
        return json.dumps(list_of_strings)

model = ProphetCatboost()