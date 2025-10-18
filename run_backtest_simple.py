"""
Simple backtest runner that avoids multiprocessing issues.
"""
import sys
sys.path.insert(0, 'src')

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

from ffproj.data import load_weekly_stats
from ffproj.features import build_all_features, select_feature_columns
from ffproj.backtest import rolling_backtest_gbm
from ffproj.io import write_parquet, save_json
from ffproj.utils import set_random_seed

# Set random seed
set_random_seed(42)

print("="*70)
print("Fantasy Football Projections - Simple Backtest")
print("="*70)

# Load data
seasons = [2022, 2023, 2024, 2025]
print(f"\nLoading data for seasons: {seasons}")
df = load_weekly_stats(seasons, cache=True)
print(f"Loaded {len(df):,} player-weeks")

# Sort for time series
df = df.sort_values(['player_id', 'season', 'week']).reset_index(drop=True)

# Build features
print("\nBuilding features...")
df = build_all_features(df)
print(f"Features built: {df.shape[1]} columns")

# Select features
feature_cols = select_feature_columns(df)
print(f"Using {len(feature_cols)} features for modeling")

# Run backtest
print("\n" + "="*70)
print("Running Rolling Backtest")
print("="*70)

predictions_df, metrics = rolling_backtest_gbm(
    df,
    feature_cols=feature_cols,
    target_col='fp_ppr',
    verbose=True
)

# Create output directory
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = Path(f"experiments/gbm_{timestamp}")
output_dir.mkdir(parents=True, exist_ok=True)

# Save results
write_parquet(predictions_df, output_dir / "backtest_predictions.parquet")
save_json(metrics, output_dir / "backtest_metrics.json")

# Print summary
print("\n" + "="*70)
print("Backtest Complete!")
print("="*70)
print(f"\nResults saved to: {output_dir}")
print(f"Predictions shape: {predictions_df.shape}")
print("\nMetrics:")
for key, value in sorted(metrics.items()):
    print(f"  {key:<25} {value:>10.4f}")

print("\n" + "="*70)
