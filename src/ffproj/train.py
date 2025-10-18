"""
Training script for fantasy football projection models.
"""
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

from .config import (
    PROCESSED_DATA_DIR, EXPERIMENTS_DIR, TRAIN_SEASONS,
    POSITIONS, RANDOM_SEED
)
from .utils import set_random_seed, split_train_val_test
from .data import load_weekly_stats
from .features import build_all_features, select_feature_columns
from .models_gbm import GBMQuantileEnsemble
from .metrics import compute_all_metrics, print_metrics_report
from .backtest import rolling_backtest_gbm, compare_baselines, evaluate_baselines
from .io import write_parquet, save_json, save_model


def prepare_data(
    seasons: list,
    cache: bool = True,
    rebuild_features: bool = False
) -> pd.DataFrame:
    """
    Load and prepare data with features.

    Parameters
    ----------
    seasons : list
        Seasons to load
    cache : bool
        Use cached data if available
    rebuild_features : bool
        Force rebuild features even if cached

    Returns
    -------
    pd.DataFrame
        Prepared data with features
    """
    cache_file = PROCESSED_DATA_DIR / f"features_{min(seasons)}_{max(seasons)}.parquet"

    if not rebuild_features and cache and cache_file.exists():
        print(f"Loading cached features from {cache_file}")
        return pd.read_parquet(cache_file)

    print("Loading raw data...")
    df = load_weekly_stats(seasons, cache=True)

    print(f"Loaded {len(df)} player-weeks")
    print(f"Positions: {df['position'].value_counts().to_dict()}")

    # Sort for time series features
    df = df.sort_values(['player_id', 'season', 'week']).reset_index(drop=True)

    # Build features
    df = build_all_features(df)

    # Remove rows with missing targets
    df = df.dropna(subset=['fp_ppr'])

    if cache:
        write_parquet(df, cache_file)
        print(f"Cached features to {cache_file}")

    return df


def train_gbm_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: list,
    target_col: str = 'fp_ppr',
    model_name: str = 'gbm_quantile',
    save_path: Path = None
) -> GBMQuantileEnsemble:
    """
    Train GBM quantile model.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training data
    val_df : pd.DataFrame
        Validation data
    feature_cols : list
        Feature column names
    target_col : str
        Target column
    model_name : str
        Model name for saving
    save_path : Path, optional
        Path to save model

    Returns
    -------
    GBMQuantileEnsemble
        Trained model
    """
    print("\n" + "="*60)
    print(f"Training {model_name}")
    print("="*60)

    X_train = train_df[feature_cols].fillna(0)
    y_train = train_df[target_col]
    X_val = val_df[feature_cols].fillna(0)
    y_val = val_df[target_col]

    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Features: {len(feature_cols)}")

    # Train model
    model = GBMQuantileEnsemble(verbose=True)
    model.train(X_train, y_train, X_val, y_val)

    # Evaluate
    print("\nEvaluating on validation set...")
    metrics = model.evaluate(X_val, y_val)
    print_metrics_report(metrics, title="Validation Metrics")

    # Feature importance
    print("\nTop 20 Features:")
    importance_df = model.get_feature_importance()
    print(importance_df.head(20).to_string(index=False))

    # Save model
    if save_path:
        save_model(model, save_path / f"{model_name}.pkl")
        # Save feature importance
        importance_df.to_csv(save_path / f"{model_name}_importance.csv", index=False)
        print(f"\nModel saved to {save_path}")

    return model


def main():
    """Main training script."""
    parser = argparse.ArgumentParser(description="Train fantasy football projection models")
    parser.add_argument(
        '--model',
        type=str,
        default='gbm',
        choices=['gbm', 'tcn', 'lstm'],
        help='Model type to train'
    )
    parser.add_argument(
        '--seasons',
        type=int,
        nargs='+',
        default=TRAIN_SEASONS,
        help='Seasons to use for training'
    )
    parser.add_argument(
        '--val-season',
        type=int,
        default=2023,
        help='Season to use for validation'
    )
    parser.add_argument(
        '--backtest',
        action='store_true',
        help='Run rolling backtest instead of single train/val split'
    )
    parser.add_argument(
        '--rebuild-features',
        action='store_true',
        help='Force rebuild features'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for models and results'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=RANDOM_SEED,
        help='Random seed'
    )

    args = parser.parse_args()

    # Set seed
    set_random_seed(args.seed)

    # Create output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = EXPERIMENTS_DIR / f"{args.model}_{timestamp}"

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Save config
    config = vars(args)
    save_json(config, output_dir / "config.json")

    # Load and prepare data
    all_seasons = args.seasons + [args.val_season]
    df = prepare_data(all_seasons, rebuild_features=args.rebuild_features)

    # Select features
    feature_cols = select_feature_columns(df)
    print(f"\nUsing {len(feature_cols)} features")

    if args.backtest:
        # Rolling backtest
        print("\n" + "="*60)
        print("Running Rolling Backtest")
        print("="*60)

        predictions_df, metrics = rolling_backtest_gbm(
            df,
            feature_cols=feature_cols,
            target_col='fp_ppr',
            verbose=True
        )

        # Save results
        write_parquet(predictions_df, output_dir / "backtest_predictions.parquet")
        save_json(metrics, output_dir / "backtest_metrics.json")

        # Evaluate baselines
        print("\nEvaluating baselines...")
        df_with_baselines = compare_baselines(df)
        baseline_metrics = evaluate_baselines(df_with_baselines)
        print("\nBaseline MAE:")
        for name, value in sorted(baseline_metrics.items()):
            print(f"{name:<30} {value:>10.4f}")

        save_json(baseline_metrics, output_dir / "baseline_metrics.json")

    else:
        # Single train/val split
        train_seasons = [s for s in args.seasons if s < args.val_season]
        val_seasons = [args.val_season]

        train_df = df[df['season'].isin(train_seasons)]
        val_df = df[df['season'].isin(val_seasons)]

        print(f"\nTrain seasons: {train_seasons}")
        print(f"Validation season: {val_seasons}")

        if args.model == 'gbm':
            model = train_gbm_model(
                train_df, val_df,
                feature_cols=feature_cols,
                save_path=output_dir
            )

            # Generate predictions for validation season
            print("\nGenerating predictions for validation season...")
            X_val = val_df[feature_cols].fillna(0)
            preds = model.predict(X_val)

            predictions_df = pd.DataFrame({
                'season': val_df['season'].values,
                'week': val_df['week'].values,
                'player_id': val_df['player_id'].values,
                'player_name': val_df['player_name'].values if 'player_name' in val_df else None,
                'position': val_df['position'].values,
                'team': val_df['team'].values if 'team' in val_df else None,
                'y_true': val_df['fp_ppr'].values,
                'q10': preds['q10'],
                'q50': preds['q50'],
                'q90': preds['q90']
            })

            write_parquet(predictions_df, output_dir / "val_predictions.parquet")

        else:
            print(f"Model type '{args.model}' not yet implemented in this script")
            print("Use the GBM model or implement DL training loop")

    print("\n" + "="*60)
    print("Training complete!")
    print(f"Results saved to: {output_dir}")
    print("="*60)


if __name__ == '__main__':
    import multiprocessing as mp
    try:
        mp.set_start_method("spawn", force=True)  # Safer on macOS, prevents segfault
    except RuntimeError:
        pass  # Already set
    main()
