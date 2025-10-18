"""
Test if the issue is specifically in the predict/evaluate step.
"""
import sys
sys.path.insert(0, 'src')

import pandas as pd
import numpy as np
import lightgbm as lgb

from ffproj.data import load_weekly_stats
from ffproj.features import build_all_features, select_feature_columns

print("="*70)
print("Testing Prediction Step")
print("="*70)

# Load and prepare data
print("\n1. Loading data...")
df = load_weekly_stats([2024, 2025], cache=True)
df = df.sort_values(['player_id', 'season', 'week']).reset_index(drop=True)
df = build_all_features(df)
feature_cols = select_feature_columns(df)

# Split
train_df = df[df['season'] == 2024].copy()
val_df = df[df['season'] == 2025].copy()

X_train = train_df[feature_cols].fillna(0)
y_train = train_df['fp_ppr']
X_val = val_df[feature_cols].fillna(0)
y_val = val_df['fp_ppr']

print(f"   Train: {len(X_train)}, Val: {len(X_val)}")

# Train single model
print("\n2. Training single model...")
params = {
    'objective': 'quantile',
    'alpha': 0.5,
    'learning_rate': 0.05,
    'num_leaves': 31,
    'verbose': -1,
    'num_threads': 1,
    'force_row_wise': True
}

dtrain = lgb.Dataset(X_train, label=y_train)
dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

model = lgb.train(
    params,
    dtrain,
    valid_sets=[dval],
    num_boost_round=100,
    callbacks=[lgb.early_stopping(stopping_rounds=10)]
)

print(f"   ✓ Training complete (iteration {model.best_iteration})")

# Test prediction in different ways
print("\n3. Testing predictions...")

try:
    print("   a) Predicting on small batch (10 rows)...")
    preds_small = model.predict(X_val.iloc[:10].values, num_threads=1)
    print(f"      ✓ Success: {preds_small[:3]}")
except Exception as e:
    print(f"      ✗ Failed: {e}")

try:
    print("   b) Predicting on full validation set...")
    preds_full = model.predict(X_val.values, num_threads=1)
    print(f"      ✓ Success: mean={preds_full.mean():.2f}")
except Exception as e:
    print(f"      ✗ Failed: {e}")

try:
    print("   c) Computing MAE manually...")
    mae = np.abs(y_val.values - preds_full).mean()
    print(f"      ✓ MAE: {mae:.2f}")
except Exception as e:
    print(f"      ✗ Failed: {e}")

print("\n4. Checking if issue is in pandas operations...")
try:
    print("   a) Creating results DataFrame...")
    results_df = pd.DataFrame({
        'actual': y_val.values[:100],
        'pred': preds_full[:100]
    })
    print(f"      ✓ DataFrame created: {results_df.shape}")
except Exception as e:
    print(f"      ✗ Failed: {e}")

print("\n" + "="*70)
print("Test complete - if you see this, prediction works!")
print("="*70)
