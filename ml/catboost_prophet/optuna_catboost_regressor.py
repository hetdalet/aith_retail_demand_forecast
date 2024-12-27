import json
import optuna
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, Pool

class OptunaCatBoostRegressor:
    """
    A wrapper class for the CatBoost Regressor with Optuna for hyperparameters tuning
    """

    def __init__(
            self,
            n_estimators: int,
            learning_rate: float = 0.01,
            metric: str = 'RMSE',
            cat_columns: list = None,
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
            "cat_features": cat_columns
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
        self.features = list(x_train.columns)
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
            model.fit(dtrain, eval_set=dval, early_stopping_rounds=150)
            preds = model.predict(dval)
            return np.sqrt(((y_val - preds) ** 2).mean())

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=100, timeout=600)

        self.model = CatBoostRegressor(**self.params)
        self.model.set_params(**study.best_params)
        self.model.fit(dtrain, eval_set=dval, early_stopping_rounds=150)

        self.is_fitted_ = True

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        assert self.is_fitted_, 'Model is not fitted!'
        return self.model.predict(X_test[self.features])

    def save(self, filepath):
        """Save the CatBoost model to a file."""
        self.model.save_model(filepath)
        # Save additional attributes if necessary
        metadata = {
            'params': self.params,
            'features': self.features,
            'is_fitted_': self.is_fitted_
        }
        with open(filepath + '_meta.json', 'w') as f:
            json.dump(metadata, f)

    @classmethod
    def load(cls, filepath):
        """Load the CatBoost model from a file."""
        with open(filepath + '_meta.json', 'r') as f:
            metadata = json.load(f)
        obj = cls(
            n_estimators=metadata['params']['iterations'],
            learning_rate=metadata['params']['learning_rate'],
            metric=metadata['params']['eval_metric'],
            cat_columns=metadata['params']['cat_features'],
            seed=metadata['params']['random_seed']
        )
        obj.features = metadata['features']
        obj.is_fitted_ = metadata['is_fitted_']
        obj.model = CatBoostRegressor()
        obj.model.load_model(filepath)
        return obj
