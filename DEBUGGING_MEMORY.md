# Memory Issue Debugging Guide

## 🔍 Current Status

**Problem**: Python processes getting killed by macOS during feature engineering
**Symptoms**: `zsh: killed python ...` after "Building features..." message
**Root Cause**: Memory explosion in feature engineering (likely cartesian product in merges)

---

## 🧪 Active Debug Steps

### Debug Script Running
```bash
python debug_features.py
```

This script:
- Loads ONLY 2025 data (~6K player-weeks)
- Adds memory logging after each feature step
- Will pinpoint exactly where memory explodes

**Check output**:
```bash
# See where it's stuck/killed
ps aux | grep debug_features

# When complete, review output for memory jumps
```

---

## 🔧 What I've Fixed So Far

### 1. Added Memory Logging
[features.py](src/ffproj/features.py) now logs after each step:
```
  [START] Shape: 6,000 rows × 100 cols | Memory: 50.0 MB
  → Rolling features...
  [after_rolling] Shape: 6,000 rows × 115 cols | Memory: 55.0 MB
  → Target/carry share...
  [after_target_share] Shape: 6,000 rows × 125 cols | Memory: 60.0 MB
  ...
```

**If you see a jump like**: `12 MB → 1200 MB` = that's the problem step!

### 2. Cleared All Caches
- Removed old parquet files with wrong column names
- Fresh data will have correct structure

### 3. Created `debug_features.py`
- Minimal test with 2025 only
- Will succeed if issue is data size
- Will fail at same spot if it's a code bug

---

## 📊 Expected Timings & Memory

### Baseline (Working System)

| Dataset | Rows | Expected Memory | Time |
|---------|------|----------------|------|
| 2025 only | ~6K | 50-100 MB | 1-2 min |
| 2024-2025 | ~12K | 100-150 MB | 2-3 min |
| 2022-2025 | ~30K | 200-300 MB | 4-6 min |
| 2019-2025 | ~80K | 500-800 MB | 8-12 min |

**If you're seeing**:
- ❌ 2025 using >500 MB = Bug (cartesian join)
- ❌ 2024-2025 using >1 GB = Bug
- ✅ 2019-2025 using ~800 MB = Normal

---

## 🐛 Common Memory Bugs (and Fixes)

### Bug #1: Cartesian Product in Opponent Strength
**Problem**: Merging opponent stats without proper keys

**Check in `features.py` line ~128**:
```python
# BAD - creates cartesian product
grouped = df.groupby(['season', 'opponent'])[target_col].apply(...)

# GOOD - includes position to prevent duplication
grouped = df.groupby(['season', 'opponent', 'position'])[target_col].apply(...)
```

### Bug #2: Team Totals Without Game_ID
**Problem**: Multiple players on same team/week get cross-joined

**Check in `features.py` line ~214**:
```python
# Needs to aggregate correctly
team_totals = df.groupby(['season', 'week', 'team']).agg({
    'targets': 'sum',
    'carries': 'sum'  # Make sure 'carries' column exists!
})
```

### Bug #3: Rolling Windows Creating Duplicates
**Problem**: `groupby().apply()` can materialize huge intermediates

**Solution**: Use `transform()` instead:
```python
# BAD
df['roll3'] = df.groupby('player_id').apply(lambda x: x.rolling(3).mean())

# GOOD
df['roll3'] = df.groupby('player_id')['stat'].transform(
    lambda x: x.rolling(3, min_periods=1).mean()
)
```

---

## 🚀 Quick Fixes to Try

### Fix #1: Reduce Data First
Test with single season to confirm code works:
```bash
# If this works, it's a memory issue not a bug
python -m src.ffproj.train --seasons 2025 --val-season 2025
```

### Fix #2: Limit Positions
Test with fewer positions:
```bash
# QBs + WRs only
python debug_features.py  # (modify to filter: df = df[df['position'].isin(['QB','WR'])])
```

### Fix #3: Disable Expensive Features
Comment out in `build_all_features()`:
```python
# df = compute_opponent_strength(df, windows=windows)  # SKIP THIS
# df = compute_team_context_features(df, windows=windows)  # SKIP THIS
```

Run again - if it works, we know which function is the culprit.

---

## 📝 Diagnostic Checklist

Run these to gather info:

```bash
# 1. Check system memory
vm_stat | grep "Pages free"

# 2. Check ulimits
ulimit -a

# 3. Check Python memory limit
python -c "import resource; print(resource.getrlimit(resource.RLIMIT_AS))"

# 4. Monitor memory during run
# Terminal 1:
python debug_features.py

# Terminal 2:
while true; do ps aux | grep python | head -1; sleep 2; done
```

---

## 🎯 What to Try Next (In Order)

### 1. **Wait for `debug_features.py` to complete**
   - If it succeeds: Memory issue is data volume
   - If it fails: Bug in feature code

### 2. **If debug succeeds, try 2 seasons**:
   ```bash
   python -m src.ffproj.train --seasons 2024 --val-season 2025
   ```

### 3. **If that succeeds, try 4 seasons**:
   ```bash
   python -m src.ffproj.train --backtest --seasons 2022 2023 2024 2025
   ```

### 4. **If it fails, check the log output**:
   Look for the last log line before kill:
   ```
   [after_rolling] Shape: ... | Memory: 60 MB
   [after_target_share] Shape: ... | Memory: 1200 MB  ← BOOM!
   ```

---

## 🔬 Advanced Debugging

If you need to dig deeper:

### Profile Memory
```python
from memory_profiler import profile

@profile
def build_all_features(df, ...):
    # ... existing code
```

Run with: `python -m memory_profiler debug_features.py`

### Use Polars (10x less memory)
```python
import polars as pl

# Convert at start
df_pl = pl.from_pandas(df)

# Do features in polars
# ...

# Convert back
df = df_pl.to_pandas()
```

---

## 📞 Current Action Items

1. ⏳ **Waiting**: `debug_features.py` output (running now)
2. 🔍 **Check**: Which feature step causes memory spike
3. 🛠️ **Fix**: The specific merge/groupby causing explosion
4. ✅ **Test**: Minimal working command
5. 📈 **Scale**: Gradually add more data

---

## 💡 Expected Resolution

**Most likely**:
- `compute_opponent_strength()` or `compute_target_share_features()` has a bad merge
- Fix: Add proper keys to prevent cartesian product
- Result: Memory drops from ~2GB to ~200MB

**Timeline**:
- Debug script: ~5 more minutes
- Identify issue: ~2 minutes
- Fix code: ~5 minutes
- Test fix: ~3 minutes
- **Total**: ~15 minutes to working state

---

**Status**: 🔄 Debug script running, will update when complete
**Next**: Analyze debug output to pinpoint exact memory spike
