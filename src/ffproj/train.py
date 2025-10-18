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
from .utils import (
    set_random_seed, split_train_val_test,
    get_current_season_and_week, remaining_games
)
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


def predict_adaptive(
    df: pd.DataFrame,
    feature_cols: list,
    output_path: Path = None
) -> pd.DataFrame:
    """
    Adaptive prediction for upcoming NFL week.

    Determines what to predict based on current day and game schedule:
    - If mid-week with games in progress: predict current week, only remaining games
    - If Tuesday after MNF: predict next week, all players

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset with features
    feature_cols : list
        Feature column names
    output_path : Path, optional
        Path to save predictions

    Returns
    -------
    pd.DataFrame
        Predictions for upcoming games
    """
    print("\n" + "="*60)
    print("Adaptive Weekly Prediction")
    print("="*60)

    # Get current NFL week context
    season, latest_done_wk, in_progress_week, reg_sched = get_current_season_and_week()

    print(f"Current season: {season}")
    print(f"Latest completed week: {latest_done_wk}")
    print(f"In-progress week: {in_progress_week}")

    # Determine target week and players
    if in_progress_week is not None:
        # Mid-week: predict current week, only remaining games
        target_week = in_progress_week
        print(f"\n→ Week {target_week} is in progress")
        print("→ Predicting only games that haven't kicked off")

        # Get remaining games
        remaining = remaining_games(season, target_week, reg_sched)
        remaining_teams = set()
        for _, game in remaining.iterrows():
            if 'home_team' in game and pd.notna(game['home_team']):
                remaining_teams.add(game['home_team'])
            if 'away_team' in game and pd.notna(game['away_team']):
                remaining_teams.add(game['away_team'])

        print(f"→ {len(remaining)} games remaining ({len(remaining_teams)} teams)")

    else:
        # Tuesday after MNF: predict next week, all players
        target_week = latest_done_wk + 1
        print(f"\n→ All games completed through week {latest_done_wk}")
        print(f"→ Predicting next week: {target_week}")
        remaining_teams = None  # All teams

    # Anti-leakage: only use data up to latest_done_wk for training
    print(f"\n→ Using data through week {latest_done_wk} for features (anti-leakage)")
    train_df = df[
        (df['season'] < season) |
        ((df['season'] == season) & (df['week'] <= latest_done_wk))
    ].copy()

    # Use recent weeks for validation
    val_cutoff_week = max(1, latest_done_wk - 2)
    val_df = train_df[
        (train_df['season'] == season) &
        (train_df['week'] >= val_cutoff_week)
    ].copy()

    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)} (weeks {val_cutoff_week}-{latest_done_wk})")

    # Train model
    X_train = train_df[feature_cols].fillna(0)
    y_train = train_df['fp_ppr']
    X_val = val_df[feature_cols].fillna(0)
    y_val = val_df['fp_ppr']

    print("\nTraining GBM quantile ensemble...")
    model = GBMQuantileEnsemble(verbose=True)
    model.train(X_train, y_train, X_val, y_val)

    # Get prediction candidates from most recent week's data
    # Use latest_done_wk data as template for player pool
    prediction_df = df[
        (df['season'] == season) &
        (df['week'] == latest_done_wk)
    ].copy()

    # Filter to remaining teams if mid-week
    if remaining_teams is not None:
        prediction_df = prediction_df[prediction_df['team'].isin(remaining_teams)]

    print(f"\n→ Generating predictions for {len(prediction_df)} player-weeks")

    # Generate predictions
    X_pred = prediction_df[feature_cols].fillna(0)
    preds = model.predict(X_pred)

    # Create output dataframe
    output_df = pd.DataFrame({
        'season': season,
        'week': target_week,
        'player_id': prediction_df['player_id'].values,
        'player_name': prediction_df['player_name'].values if 'player_name' in prediction_df else None,
        'position': prediction_df['position'].values,
        'team': prediction_df['team'].values if 'team' in prediction_df else None,
        'opponent': prediction_df['opponent'].values if 'opponent' in prediction_df else None,
        'q10': preds['q10'],
        'q50': preds['q50'],
        'q90': preds['q90']
    })

    # Sort by projection (q50) descending
    output_df = output_df.sort_values('q50', ascending=False).reset_index(drop=True)

    # Save if output path provided
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        write_parquet(output_df, output_path)
        print(f"\n✓ Predictions saved to: {output_path}")

    # Print summary
    print("\n" + "="*60)
    print("Prediction Summary")
    print("="*60)
    print(f"Target: {season} Week {target_week}")
    print(f"Players: {len(output_df)}")
    print(f"Positions: {output_df['position'].value_counts().to_dict()}")
    print("\nTop 10 Projections (PPR):")
    print(output_df[['player_name', 'position', 'team', 'q10', 'q50', 'q90']].head(10).to_string(index=False))
    print("="*60)

    return output_df


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
    parser.add_argument(
        '--predict-adaptive',
        action='store_true',
        help='Generate adaptive predictions for upcoming NFL week'
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

    # Handle adaptive prediction mode
    if args.predict_adaptive:
        # Load all available data for adaptive prediction
        all_seasons = [2022, 2023, 2024, 2025]
        df = prepare_data(all_seasons, rebuild_features=args.rebuild_features)

        # Select features
        feature_cols = select_feature_columns(df)
        print(f"\nUsing {len(feature_cols)} features")

        # Generate adaptive predictions
        output_file = PROCESSED_DATA_DIR / "preds_upcoming_week.parquet"
        predictions_df = predict_adaptive(df, feature_cols, output_path=output_file)

        print("\n" + "="*60)
        print("Adaptive prediction complete!")
        print(f"Results saved to: {output_file}")
        print("="*60)
        return

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
