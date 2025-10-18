"""
Simple example script demonstrating fantasy football projections.

Usage:
    python example.py
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

import pandas as pd
from ffproj.data import load_weekly_stats
from ffproj.features import build_all_features, select_feature_columns
from ffproj.models_gbm import GBMQuantileEnsemble
from ffproj.metrics import mae, rmse, coverage_at
from ffproj.utils import set_random_seed


def main():
    """Run simple train/predict example."""

    print("="*70)
    print("Fantasy Football Projections - Simple Example")
    print("="*70)

    # Set seed
    set_random_seed(42)

    # 1. Load data
    print("\n1. Loading NFL data (this may take a few minutes on first run)...")
    try:
        df = load_weekly_stats([2024, 2025], cache=True)
        print(f"   ✓ Loaded {len(df):,} player-weeks")
    except Exception as e:
        print(f"   ✗ Error loading data: {e}")
        print("\n   Make sure you have internet connection and nflreadpy installed:")
        print("   pip install nflreadpy")
        return

    # 2. Build features
    print("\n2. Building features...")
    df = df.sort_values(['player_id', 'season', 'week']).reset_index(drop=True)
    df = build_all_features(df)
    df = df.dropna(subset=['fp_ppr'])

    feature_cols = select_feature_columns(df)
    print(f"   ✓ Created {len(feature_cols)} features")

    # 3. Train/val split
    print("\n3. Splitting data...")
    train_df = df[df['season'] == 2024]
    val_df = df[df['season'] == 2025]

    X_train = train_df[feature_cols].fillna(0)
    y_train = train_df['fp_ppr']
    X_val = val_df[feature_cols].fillna(0)
    y_val = val_df['fp_ppr']

    print(f"   ✓ Training samples: {len(X_train):,}")
    print(f"   ✓ Validation samples: {len(X_val):,}")

    # 4. Train model
    print("\n4. Training GBM model (this will take 1-2 minutes)...")
    model = GBMQuantileEnsemble(verbose=False)
    model.train(X_train, y_train, X_val, y_val)
    print("   ✓ Model trained!")

    # 5. Evaluate
    print("\n5. Evaluating model...")
    preds = model.predict(X_val)

    metrics = {
        'MAE': mae(y_val, preds['q50']),
        'RMSE': rmse(y_val, preds['q50']),
        'Coverage_80%': coverage_at(y_val, preds['q10'], preds['q90']),
    }

    print("\n   Validation Metrics:")
    print("   " + "-"*50)
    for name, value in metrics.items():
        print(f"   {name:<20} {value:>10.3f}")
    print("   " + "-"*50)

    # 6. Show example predictions
    print("\n6. Example predictions for Week 1 of 2025:")
    print("   " + "-"*70)

    week1_df = val_df[val_df['week'] == 1].copy()
    week1_X = week1_df[feature_cols].fillna(0)
    week1_preds = model.predict(week1_X)

    # Create results dataframe
    results = pd.DataFrame({
        'Player': week1_df['player_name'].values if 'player_name' in week1_df.columns else week1_df['player_id'].values,
        'Pos': week1_df['position'].values,
        'P10': week1_preds['q10'].round(1),
        'Proj': week1_preds['q50'].round(1),
        'P90': week1_preds['q90'].round(1),
        'Actual': week1_df['fp_ppr'].values.round(1)
    })

    # Show top 15
    results_sorted = results.sort_values('Proj', ascending=False)
    print(results_sorted.head(15).to_string(index=False))
    print("   " + "-"*70)

    # 7. Feature importance
    print("\n7. Top 10 most important features:")
    importance = model.get_feature_importance()
    print("   " + "-"*50)
    for idx, row in importance.head(10).iterrows():
        print(f"   {row['feature']:<40} {row['importance']:>8.0f}")
    print("   " + "-"*50)

    print("\n" + "="*70)
    print("✓ Example complete!")
    print("="*70)
    print("\nNext steps:")
    print("  • Run full backtest: python -m src.ffproj.train --backtest")
    print("  • Launch web app: streamlit run src/app/streamlit_app.py")
    print("  • Explore notebook: jupyter notebook notebooks/00_quickstart.ipynb")
    print("="*70)


if __name__ == '__main__':
    main()
