"""
Debug script to isolate memory issues in feature engineering.
"""
import sys
sys.path.insert(0, 'src')

from ffproj.data import load_weekly_stats
from ffproj.features import build_all_features
from ffproj.utils import set_random_seed

set_random_seed(42)

print("="*70)
print("DEBUG: Feature Engineering Memory Test")
print("="*70)

# Test with just 2025 (smallest dataset)
print("\n1. Loading 2025 data only...")
df = load_weekly_stats([2025], cache=True)
print(f"   Loaded: {len(df):,} player-weeks")
print(f"   Columns: {len(df.columns)}")
print(f"   Memory: {df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")

# Check for missing columns
print("\n2. Checking for required columns...")
required = ['targets', 'receptions', 'receiving_yards', 'carries', 'rushing_yards',
            'passing_yards', 'passing_td', 'fp_ppr']
missing = [c for c in required if c not in df.columns]
if missing:
    print(f"   ❌ MISSING: {missing}")
    print(f"   Available columns: {sorted(df.columns)}")
    sys.exit(1)
else:
    print(f"   ✓ All required columns present")

# Sort for time series
print("\n3. Sorting data...")
df = df.sort_values(['player_id', 'season', 'week']).reset_index(drop=True)
print(f"   ✓ Sorted")

# Try building features
print("\n4. Building features with debug logging...")
try:
    df = build_all_features(df)
    print("\n✓ SUCCESS! Features built without errors")
    print(f"   Final shape: {df.shape}")
    print(f"   Final memory: {df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")
except Exception as e:
    print(f"\n❌ ERROR during feature engineering:")
    print(f"   {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*70)
print("✓ Debug test passed!")
print("="*70)
