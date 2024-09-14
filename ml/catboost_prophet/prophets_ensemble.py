import os
import re
import json
from itertools import product
import pandas as pd
from prophet import Prophet
from prophet.serialize import model_to_json, model_from_json
from tqdm import tqdm
from holidays.holiday_base import HolidayBase

class suppress_stdout_stderr(object):
    """
    A context manager for doing a "deep suppression" of stdout and stderr in Python.
    """

    def __init__(self):
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        self.save_fds = (os.dup(1), os.dup(2))

    def __enter__(self):
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
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

    def _get_holidays(self, data: pd.DataFrame) -> pd.DataFrame:
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

    def _predict_level(self, periods: int, level: str, future_regressors: pd.DataFrame) -> pd.DataFrame:
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
        for level in tqdm([self.freq] + self.levels, desc='Fitting prophets...'):
            self._fit_level(data, level)
        self.is_fitted_ = True

    def forecast(self, periods: int, future_regressors: pd.DataFrame = None) -> pd.DataFrame:
        """Makes forecasts for all aggregation levels and combines them."""
        assert self.is_fitted_, 'Model is not fitted'
        forecasts = [
            self._predict_level(periods, level, future_regressors) 
            for level in tqdm([self.freq] + self.levels, desc='Forecasting...')
        ]

        forecast = forecasts[0].rename(columns={f'ds_{self.freq}': 'ds', f'yhat_{self.freq}': 'yhat'})
        for level, fore in zip(self.levels, forecasts[1:]):
            forecast = self._combine_levels(forecast, fore, level)

        return self._drop_redundant(forecast)

    def save(self, filepath):
        """Save the ensemble to a JSON file."""
        data = {
            'freq': self.freq,
            'levels': self.levels,
            'is_fitted_': self.is_fitted_,
            'prophets_': {level: model_to_json(model) for level, model in self.prophets_.items()}
        }
        with open(filepath, 'w') as f:
            json.dump(data, f)

    @classmethod
    def load(cls, filepath):
        """Load the ensemble from a JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        ensemble = cls(freq=data['freq'], levels=[], agg_fn=[])
        ensemble.levels = data['levels']
        ensemble.is_fitted_ = data['is_fitted_']
        ensemble.prophets_ = {level: model_from_json(model_json) for level, model_json in data['prophets_'].items()}
        return ensemble
