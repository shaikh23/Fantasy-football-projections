"""
Prepare prediction data for the Streamlit app.
Copies backtest predictions to the format the app expects.
"""
import sys
sys.path.insert(0, 'src')

import pandas as pd
from pathlib import Path
from ffproj.io import read_parquet

print("="*70)
print("Preparing Data for Streamlit App")
print("="*70)

# Find the most recent backtest results
experiments_dir = Path("experiments")
backtest_dirs = sorted([d for d in experiments_dir.glob("gbm_*") if d.is_dir()], reverse=True)

if not backtest_dirs:
    print("\n❌ No backtest results found!")
    print("Run: python run_backtest_simple.py")
    sys.exit(1)

latest_dir = backtest_dirs[0]
pred_file = latest_dir / "backtest_predictions.parquet"

if not pred_file.exists():
    print(f"\n❌ Predictions not found in {latest_dir}")
    sys.exit(1)

print(f"\nLoading predictions from: {latest_dir.name}")
df = read_parquet(pred_file)

print(f"Found {len(df):,} predictions")
print(f"Seasons: {sorted(df['season'].unique())}")
print(f"Weeks: {sorted(df['week'].unique())}")

# Save to processed data directory for app
processed_dir = Path("data/processed")
processed_dir.mkdir(parents=True, exist_ok=True)

# Save overall file
output_file = processed_dir / "backtest_predictions.parquet"
df.to_parquet(output_file)
print(f"\n✓ Saved to: {output_file}")

# Also save individual week files for faster loading
print("\nSaving individual week files...")
for season in df['season'].unique():
    for week in df[df['season'] == season]['week'].unique():
        week_df = df[(df['season'] == season) & (df['week'] == week)]
        week_file = processed_dir / f"preds_week_{season}_{week}.parquet"
        week_df.to_parquet(week_file)
        print(f"  ✓ Season {season}, Week {week}: {len(week_df)} players")

print("\n" + "="*70)
print("✓ App data ready!")
print("="*70)
print("\nTo launch the app:")
print("  streamlit run src/app/streamlit_app.py")
print("="*70)
