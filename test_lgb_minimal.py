"""
Minimal test to see if LightGBM training works at all.
"""
import sys
sys.path.insert(0, 'src')

import pandas as pd
import lightgbm as lgb
from ffproj.data import load_weekly_stats
from ffproj.features import build_all_features, select_feature_columns

print("Loading data...")
df = load_weekly_stats([2025], cache=True)
df = df.sort_values(['player_id', 'season', 'week']).reset_index(drop=True)

print("Building features...")
df = build_all_features(df)
feature_cols = select_feature_columns(df)

print(f"Data shape: {df.shape}")
print(f"Features: {len(feature_cols)}")

# Simple train/test split
train_df = df[df['week'] < 5].copy()
test_df = df[df['week'] == 5].copy()

X_train = train_df[feature_cols].fillna(0)
y_train = train_df['fp_ppr']
X_test = test_df[feature_cols].fillna(0)
y_test = test_df['fp_ppr']

print(f"\nTrain samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# Train single model with thread-safe params
print("\nTraining LightGBM model...")
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
dtest = lgb.Dataset(X_test, label=y_test, reference=dtrain)

model = lgb.train(
    params,
    dtrain,
    valid_sets=[dtest],
    num_boost_round=100,
    callbacks=[lgb.early_stopping(stopping_rounds=10)]
)

print("\n✓ SUCCESS! LightGBM training completed")
print(f"Best iteration: {model.best_iteration}")

# Make predictions
preds = model.predict(X_test)
mae = abs(y_test.values - preds).mean()
print(f"Test MAE: {mae:.2f}")
