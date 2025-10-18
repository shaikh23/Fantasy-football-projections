"""
Gradient Boosting Machine models with quantile regression for fantasy projections.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import train_test_split

from .config import GBM_PARAMS, GBM_QUANTILES, GBM_NUM_BOOST_ROUND, GBM_EARLY_STOPPING
from .utils import enforce_monotonic_quantiles
from .metrics import mae, rmse, coverage_at


class GBMQuantileEnsemble:
    """
    Ensemble of LightGBM models for quantile regression.

    Trains separate models for different quantiles (P10, P50, P90)
    and enforces monotonicity constraints.
    """

    def __init__(
        self,
        quantiles: List[float] = None,
        params: Dict = None,
        num_boost_round: int = None,
        early_stopping_rounds: int = None,
        verbose: bool = False
    ):
        """
        Initialize GBM quantile ensemble.

        Parameters
        ----------
        quantiles : list of float, optional
            Quantiles to predict (default: [0.1, 0.5, 0.9])
        params : dict, optional
            LightGBM parameters
        num_boost_round : int, optional
            Number of boosting rounds
        early_stopping_rounds : int, optional
            Early stopping rounds
        verbose : bool
            Whether to print training progress
        """
        self.quantiles = quantiles or GBM_QUANTILES
        self.params = params or GBM_PARAMS.copy()
        self.num_boost_round = num_boost_round or GBM_NUM_BOOST_ROUND
        self.early_stopping_rounds = early_stopping_rounds or GBM_EARLY_STOPPING
        self.verbose = verbose

        self.models: Dict[float, lgb.Booster] = {}
        self.feature_names: Optional[List[str]] = None

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: Optional[pd.DataFrame] = None,
        y_valid: Optional[pd.Series] = None,
        categorical_features: Optional[List[str]] = None
    ) -> Dict[float, Dict]:
        """
        Train quantile models.

        Parameters
        ----------
        X_train : pd.DataFrame
            Training features
        y_train : pd.Series
            Training target
        X_valid : pd.DataFrame, optional
            Validation features
        y_valid : pd.Series, optional
            Validation target
        categorical_features : list of str, optional
            Categorical feature names

        Returns
        -------
        dict
            Training history for each quantile
        """
        self.feature_names = list(X_train.columns)

        # If no validation set provided, split from training
        if X_valid is None or y_valid is None:
            X_train, X_valid, y_train, y_valid = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42
            )

        history = {}

        for alpha in self.quantiles:
            if self.verbose:
                print(f"\nTraining quantile {alpha:.2f} model...")

            # Set quantile-specific parameters
            params = self.params.copy()
            params['alpha'] = alpha

            # Create datasets
            dtrain = lgb.Dataset(
                X_train, label=y_train,
                categorical_feature=categorical_features,
                free_raw_data=False
            )
            dvalid = lgb.Dataset(
                X_valid, label=y_valid,
                reference=dtrain,
                categorical_feature=categorical_features,
                free_raw_data=False
            )

            # Train model
            evals_result = {}
            model = lgb.train(
                params,
                dtrain,
                num_boost_round=self.num_boost_round,
                valid_sets=[dtrain, dvalid],
                valid_names=['train', 'valid'],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=self.early_stopping_rounds),
                    lgb.log_evaluation(period=100 if self.verbose else 0),
                    lgb.record_evaluation(evals_result)
                ]
            )

            self.models[alpha] = model
            history[alpha] = evals_result

            if self.verbose:
                # Evaluate on validation set
                y_pred = model.predict(X_valid, num_threads=1)
                val_mae = mae(y_valid, y_pred)
                print(f"Quantile {alpha:.2f} - Validation MAE: {val_mae:.3f}")

        return history

    def predict(
        self,
        X: pd.DataFrame,
        enforce_monotonic: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Predict quantiles for input data.

        Parameters
        ----------
        X : pd.DataFrame
            Features to predict
        enforce_monotonic : bool
            Whether to enforce monotonic quantiles (q10 <= q50 <= q90)

        Returns
        -------
        dict
            Dictionary mapping quantile names to predictions
            Keys: 'q10', 'q50', 'q90'
        """
        if not self.models:
            raise ValueError("Models not trained. Call train() first.")

        predictions = {}
        for alpha in self.quantiles:
            preds = self.models[alpha].predict(X, num_threads=1)
            predictions[alpha] = preds

        # Map to standard names
        result = {}
        if 0.1 in predictions:
            result['q10'] = predictions[0.1]
        if 0.5 in predictions:
            result['q50'] = predictions[0.5]
        if 0.9 in predictions:
            result['q90'] = predictions[0.9]

        # Enforce monotonicity
        if enforce_monotonic and all(k in result for k in ['q10', 'q50', 'q90']):
            result['q10'], result['q50'], result['q90'] = enforce_monotonic_quantiles(
                result['q10'], result['q50'], result['q90']
            )

        return result

    def predict_median(self, X: pd.DataFrame) -> np.ndarray:
        """Predict median (q50) only."""
        if 0.5 not in self.models:
            raise ValueError("Median model (quantile=0.5) not trained.")
        return self.models[0.5].predict(X, num_threads=1)

    def get_feature_importance(
        self,
        quantile: float = 0.5,
        importance_type: str = 'gain'
    ) -> pd.DataFrame:
        """
        Get feature importance for a specific quantile model.

        Parameters
        ----------
        quantile : float
            Which quantile model to use (default: 0.5 for median)
        importance_type : str
            Type of importance: 'gain' or 'split'

        Returns
        -------
        pd.DataFrame
            Feature importance sorted by value
        """
        if quantile not in self.models:
            raise ValueError(f"Model for quantile {quantile} not found.")

        model = self.models[quantile]
        importance = model.feature_importance(importance_type=importance_type)

        df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)

        return df

    def evaluate(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> Dict[str, float]:
        """
        Evaluate models on test data.

        Parameters
        ----------
        X_test : pd.DataFrame
            Test features
        y_test : pd.Series
            Test target

        Returns
        -------
        dict
            Evaluation metrics
        """
        preds = self.predict(X_test)

        metrics = {
            'MAE': mae(y_test, preds['q50']),
            'RMSE': rmse(y_test, preds['q50']),
        }

        if 'q10' in preds and 'q90' in preds:
            metrics['Coverage_80'] = coverage_at(y_test, preds['q10'], preds['q90'])
            metrics['Interval_Width'] = float(np.mean(preds['q90'] - preds['q10']))

        return metrics


def train_gbm_quantiles(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    quantiles: Tuple[float, ...] = (0.1, 0.5, 0.9),
    params: Optional[Dict] = None,
    num_boost_round: int = 500,
    early_stopping: int = 50,
    verbose: bool = False
) -> Dict[float, lgb.Booster]:
    """
    Train multiple LightGBM models for different quantiles.

    This is a simplified functional interface to GBMQuantileEnsemble.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training target
    X_valid : pd.DataFrame
        Validation features
    y_valid : pd.Series
        Validation target
    quantiles : tuple of float
        Quantiles to train (default: 0.1, 0.5, 0.9)
    params : dict, optional
        LightGBM parameters
    num_boost_round : int
        Number of boosting rounds
    early_stopping : int
        Early stopping rounds
    verbose : bool
        Print progress

    Returns
    -------
    dict
        Dictionary mapping quantile -> trained model
    """
    if params is None:
        params = GBM_PARAMS.copy()

    models = {}

    for alpha in quantiles:
        if verbose:
            print(f"\nTraining quantile {alpha:.2f} model...")

        p = params.copy()
        p['alpha'] = alpha

        dtrain = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
        dvalid = lgb.Dataset(X_valid, label=y_valid, reference=dtrain, free_raw_data=False)

        model = lgb.train(
            p,
            dtrain,
            valid_sets=[dvalid],
            num_boost_round=num_boost_round,
            callbacks=[
                lgb.early_stopping(stopping_rounds=early_stopping),
                lgb.log_evaluation(period=100 if verbose else 0)
            ]
        )

        models[alpha] = model

        if verbose:
            y_pred = model.predict(X_valid, num_threads=1)
            val_mae = mae(y_valid, y_pred)
            print(f"Quantile {alpha:.2f} - Validation MAE: {val_mae:.3f}")

    return models


def predict_quantiles(
    models: Dict[float, lgb.Booster],
    X: pd.DataFrame,
    enforce_monotonic: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Predict quantiles using trained models.

    Parameters
    ----------
    models : dict
        Dictionary of quantile -> model
    X : pd.DataFrame
        Features
    enforce_monotonic : bool
        Enforce q10 <= q50 <= q90

    Returns
    -------
    tuple
        (q10, q50, q90) predictions
    """
    q10 = models[0.1].predict(X, num_threads=1) if 0.1 in models else None
    q50 = models[0.5].predict(X, num_threads=1) if 0.5 in models else None
    q90 = models[0.9].predict(X, num_threads=1) if 0.9 in models else None

    if enforce_monotonic and all(q is not None for q in [q10, q50, q90]):
        q10, q50, q90 = enforce_monotonic_quantiles(q10, q50, q90)

    return q10, q50, q90
