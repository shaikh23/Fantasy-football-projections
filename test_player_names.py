"""
Test that player names are preserved in backtest results.
"""
import sys
sys.path.insert(0, 'src')

import pandas as pd
from ffproj.data import load_weekly_stats
from ffproj.features import build_all_features, select_feature_columns
from ffproj.backtest import rolling_backtest_gbm

print("="*70)
print("Testing Player Names in Backtest")
print("="*70)

# Load small dataset
print("\nLoading 2025 data...")
df = load_weekly_stats([2025], cache=True)
df = df.sort_values(['player_id', 'season', 'week']).reset_index(drop=True)

print(f"Columns before features: {len(df.columns)}")
print(f"Has player_name: {'player_name' in df.columns}")
print(f"Has player_display_name: {'player_display_name' in df.columns}")

# Build features
print("\nBuilding features...")
df = build_all_features(df)
feature_cols = select_feature_columns(df)

print(f"Columns after features: {len(df.columns)}")
print(f"Has player_name: {'player_name' in df.columns}")
print(f"Has player_display_name: {'player_display_name' in df.columns}")

# Run backtest on just 3 weeks to test quickly
print("\nRunning mini-backtest (first 3 weeks)...")
df_mini = df[df['week'] <= 3].copy()

predictions_df, metrics = rolling_backtest_gbm(
    df_mini,
    feature_cols=feature_cols,
    target_col='fp_ppr',
    verbose=False
)

print(f"\n✓ Backtest complete!")
print(f"Predictions shape: {predictions_df.shape}")
print(f"\nColumns in predictions:")
print(predictions_df.columns.tolist())

print(f"\nSample predictions:")
print(predictions_df.head(10)[['player_name', 'player_id', 'position', 'team', 'opponent', 'q50']])

if 'player_name' in predictions_df.columns:
    non_null = predictions_df['player_name'].notna().sum()
    print(f"\n✅ SUCCESS! Player names preserved: {non_null}/{len(predictions_df)} rows")
else:
    print("\n❌ FAILED: player_name column missing")

print("\n" + "="*70)
