"""
Rolling backtest harness for fantasy football projections.
"""
import pandas as pd
import numpy as np
import gc
import multiprocessing as mp
from typing import Dict, List, Tuple, Optional, Callable
from tqdm import tqdm

# Set multiprocessing to spawn mode for macOS stability
try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass  # Already set

from .models_gbm import train_gbm_quantiles, predict_quantiles
from .metrics import compute_all_metrics, mae, rmse, coverage_at
from .config import BACKTEST_TRAIN_WEEKS, BACKTEST_VAL_WEEKS, MIN_TRAIN_SAMPLES, MIN_VAL_SAMPLES


def rolling_backtest_gbm(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = 'fp_ppr',
    weeks_train: int = BACKTEST_TRAIN_WEEKS,
    quantiles: Tuple[float, ...] = (0.1, 0.5, 0.9),
    min_train_samples: int = MIN_TRAIN_SAMPLES,
    min_val_samples: int = MIN_VAL_SAMPLES,
    verbose: bool = True
) -> Tuple[pd.DataFrame, Dict]:
    """
    Perform rolling window backtest using GBM models.

    Trains models on past N weeks and predicts next week,
    rolling forward through the season.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset with features and target
        Must have columns: 'week', 'season', feature_cols, target_col
    feature_cols : list of str
        Feature column names
    target_col : str
        Target column name
    weeks_train : int
        Number of weeks to use for training
    quantiles : tuple
        Quantiles to predict
    min_train_samples : int
        Minimum samples required for training
    min_val_samples : int
        Minimum samples required for validation
    verbose : bool
        Print progress

    Returns
    -------
    tuple
        (predictions_df, metrics_dict)
        predictions_df has columns: week, player_id, y_true, q10, q50, q90
        metrics_dict contains overall evaluation metrics
    """
    # Sort by season and week
    df = df.sort_values(['season', 'week']).reset_index(drop=True)

    # Get unique season-week combinations
    weeks = df[['season', 'week']].drop_duplicates().sort_values(['season', 'week'])

    results = []

    if verbose:
        print(f"Starting rolling backtest with {weeks_train} week training window")
        print(f"Total folds: {len(weeks)}")

    weeks_list = list(weeks.itertuples(index=False))

    for fold_idx, (season, week) in enumerate(weeks_list, 1):
        if verbose and fold_idx % 5 == 0:
            print(f"  Progress: {fold_idx}/{len(weeks_list)} folds completed")
        # Define training window: past N weeks
        train_mask = (
            ((df['season'] == season) & (df['week'] < week)) |
            (df['season'] < season)
        )

        # Take only last weeks_train weeks
        train_df = df[train_mask].groupby('player_id').tail(weeks_train)

        # Validation: current week
        val_mask = (df['season'] == season) & (df['week'] == week)
        val_df = df[val_mask]

        # Check minimum samples
        if len(train_df) < min_train_samples or len(val_df) < min_val_samples:
            continue

        # Prepare data
        X_train = train_df[feature_cols].fillna(0)
        y_train = train_df[target_col]
        X_val = val_df[feature_cols].fillna(0)
        y_val = val_df[target_col]

        # Train models
        try:
            # Force thread-safe parameters for LightGBM
            params = {
                'objective': 'quantile',
                'learning_rate': 0.05,
                'num_leaves': 63,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 1,
                'min_data_in_leaf': 40,
                'verbose': -1,
                'num_threads': 1,  # Single-threaded to avoid segfault
                'force_row_wise': True,  # Stability on macOS
            }

            models = train_gbm_quantiles(
                X_train, y_train, X_val, y_val,
                quantiles=quantiles,
                params=params,
                verbose=False
            )

            # Predict
            q10, q50, q90 = predict_quantiles(models, X_val)

            # Store results
            fold_results = pd.DataFrame({
                'season': season,
                'week': week,
                'player_id': val_df['player_id'].values,
                'player_name': val_df['player_display_name'].values if 'player_display_name' in val_df.columns else None,
                'position': val_df['position'].values if 'position' in val_df.columns else None,
                'team': val_df['team'].values if 'team' in val_df.columns else None,
                'opponent': val_df['opponent'].values if 'opponent' in val_df.columns else None,
                'y_true': y_val.values,
                'q10': q10,
                'q50': q50,
                'q90': q90
            })

            results.append(fold_results)

            # Free memory after each fold
            del X_train, y_train, X_val, y_val, models, q10, q50, q90, fold_results
            gc.collect()

        except Exception as e:
            if verbose:
                print(f"Error at season {season} week {week}: {e}")
            continue

    # Combine all results
    if not results:
        raise ValueError("No valid backtest folds produced. Check data and parameters.")

    predictions_df = pd.concat(results, ignore_index=True)

    # Compute overall metrics
    metrics = {
        'MAE': mae(predictions_df['y_true'], predictions_df['q50']),
        'RMSE': rmse(predictions_df['y_true'], predictions_df['q50']),
        'Coverage_80': coverage_at(
            predictions_df['y_true'],
            predictions_df['q10'],
            predictions_df['q90']
        ),
        'Interval_Width': float(np.mean(predictions_df['q90'] - predictions_df['q10']))
    }

    # Compute metrics by position if available
    if 'position' in predictions_df.columns:
        for pos in predictions_df['position'].unique():
            pos_data = predictions_df[predictions_df['position'] == pos]
            metrics[f'{pos}_MAE'] = mae(pos_data['y_true'], pos_data['q50'])
            metrics[f'{pos}_RMSE'] = rmse(pos_data['y_true'], pos_data['q50'])
            metrics[f'{pos}_Coverage'] = coverage_at(
                pos_data['y_true'], pos_data['q10'], pos_data['q90']
            )

    if verbose:
        print("\n" + "="*60)
        print("Backtest Results")
        print("="*60)
        for metric_name, value in sorted(metrics.items()):
            print(f"{metric_name:<30} {value:>10.4f}")
        print("="*60)

    return predictions_df, metrics


def compare_baselines(
    df: pd.DataFrame,
    target_col: str = 'fp_ppr',
    player_col: str = 'player_id'
) -> pd.DataFrame:
    """
    Create simple baseline predictions for comparison.

    Baselines:
    1. Last game points
    2. 3-game moving average
    3. 5-game moving average
    4. Season average to date

    Parameters
    ----------
    df : pd.DataFrame
        Player data with target column
    target_col : str
        Target column name
    player_col : str
        Player ID column

    Returns
    -------
    pd.DataFrame
        DataFrame with baseline predictions added
    """
    df = df.copy()

    # Last game
    df['baseline_last_game'] = df.groupby(player_col)[target_col].shift(1)

    # 3-game moving average
    df['baseline_3game_avg'] = df.groupby(player_col)[target_col].transform(
        lambda x: x.shift(1).rolling(window=3, min_periods=1).mean()
    )

    # 5-game moving average
    df['baseline_5game_avg'] = df.groupby(player_col)[target_col].transform(
        lambda x: x.shift(1).rolling(window=5, min_periods=1).mean()
    )

    # Season average to date
    df['baseline_season_avg'] = df.groupby([player_col, 'season'])[target_col].transform(
        lambda x: x.shift(1).expanding().mean()
    )

    return df


def evaluate_baselines(
    df: pd.DataFrame,
    target_col: str = 'fp_ppr'
) -> Dict[str, float]:
    """
    Evaluate baseline predictions.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with baseline predictions
    target_col : str
        Target column name

    Returns
    -------
    dict
        MAE for each baseline method
    """
    baseline_cols = [
        'baseline_last_game',
        'baseline_3game_avg',
        'baseline_5game_avg',
        'baseline_season_avg'
    ]

    results = {}

    for col in baseline_cols:
        if col in df.columns:
            valid_mask = df[col].notna() & df[target_col].notna()
            if valid_mask.sum() > 0:
                results[col] = mae(
                    df.loc[valid_mask, target_col],
                    df.loc[valid_mask, col]
                )

    return results


def backtest_start_sit_decisions(
    predictions_df: pd.DataFrame,
    threshold_col: str = 'q50',
    actual_col: str = 'y_true'
) -> Dict[str, float]:
    """
    Evaluate start/sit decisions based on predictions.

    For each position, determine if model would have correctly
    identified top performers.

    Parameters
    ----------
    predictions_df : pd.DataFrame
        Predictions with columns: position, week, q50, y_true
    threshold_col : str
        Column to use for ranking predictions
    actual_col : str
        Column with actual points

    Returns
    -------
    dict
        Precision@K metrics for each position
    """
    from .metrics import precision_at_k

    results = {}

    if 'position' not in predictions_df.columns:
        return results

    for pos in predictions_df['position'].unique():
        pos_data = predictions_df[predictions_df['position'] == pos]

        # Evaluate per week then average
        weekly_precision = []

        for week in pos_data['week'].unique():
            week_data = pos_data[pos_data['week'] == week]

            if len(week_data) < 10:
                continue

            # Top 10 precision
            prec = precision_at_k(
                week_data[actual_col].values,
                week_data[threshold_col].values,
                k=10
            )
            weekly_precision.append(prec)

        if weekly_precision:
            results[f'{pos}_Precision@10'] = np.mean(weekly_precision)

    return results


def save_backtest_results(
    predictions_df: pd.DataFrame,
    metrics: Dict,
    output_path: str
):
    """
    Save backtest results to disk.

    Parameters
    ----------
    predictions_df : pd.DataFrame
        Predictions dataframe
    metrics : dict
        Metrics dictionary
    output_path : str
        Base output path (without extension)
    """
    from .io import write_parquet, save_json

    # Save predictions
    write_parquet(predictions_df, f"{output_path}_predictions.parquet")

    # Save metrics
    save_json(metrics, f"{output_path}_metrics.json")

    print(f"Backtest results saved to {output_path}")
