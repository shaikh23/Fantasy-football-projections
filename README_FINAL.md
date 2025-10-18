# Fantasy Football Projections - Ready to Use! 🏈

## ✅ Status: ALL ISSUES FIXED

The project is now **fully functional** with all critical bugs resolved:

1. ✅ **Memory explosion fixed** (80TB → 3MB)
2. ✅ **Threading issues resolved** (no more segfaults)
3. ✅ **Feature engineering working** (all 43 features)
4. ✅ **Model training stable** (LightGBM single-threaded)
5. ✅ **Backtest completed** (28 folds, 3 seasons)
6. ✅ **Streamlit app ready** (interactive projections)

---

## 🚀 Quick Start

### 1. View Projections in the App

```bash
# Launch the Streamlit web app
./launch_app.sh

# Or manually:
streamlit run src/app/streamlit_app.py
```

The app will open at `http://localhost:8501` and show:
- Weekly projections with uncertainty (P10, P50, P90)
- Filter by position, scoring system
- Compare predictions vs actuals
- Download CSV for your fantasy platform

### 2. Run a Complete Example

```bash
# See the full ML pipeline in action
python working_example.py
```

This will:
- Load 2024-2025 NFL data
- Build 43 features (11.7 MB memory)
- Train quantile models (P10, P50, P90)
- Evaluate on 2025 season
- Show top predictions and metrics

**Expected Output:**
- MAE: ~4.37 fantasy points
- RMSE: ~6.19 points
- 80% Coverage: ~75%

### 3. Run Full Backtest

```bash
# Rolling window backtest on multiple seasons
python run_backtest_simple.py
```

This runs a 28-fold backtest on 2023-2025 data and saves results to `experiments/`.

---

## 📊 Performance Metrics

### Overall Backtest Results (28 Folds)
| Metric | Value |
|--------|-------|
| MAE | 4.52 pts |
| RMSE | 6.45 pts |
| Coverage (80%) | 74.5% |

### By Position
| Position | MAE | RMSE | Coverage |
|----------|-----|------|----------|
| QB | 6.54 | 8.31 | 72.2% |
| RB | 4.43 | 6.44 | 75.9% |
| WR | 4.51 | 6.49 | 74.5% |
| TE | 3.55 | 5.08 | 74.3% |

---

## 🔧 What Was Fixed

### Critical Bug #1: Memory Explosion
**Problem:** Feature engineering tried to allocate 80.8 TiB of memory
**Cause:** Cartesian product in `compute_opponent_strength()`
**Fix:** Changed from `.apply()` + `.merge()` to `.transform()`
**Result:** 80TB → 2.9MB (30 million times improvement!)

**File:** [`src/ffproj/features.py`](src/ffproj/features.py)

### Critical Bug #2: Segmentation Faults
**Problem:** Python crashed during model training/prediction
**Cause:** LightGBM + macOS + multiprocessing threading conflicts
**Fix:** Added `num_threads=1` to all training AND prediction calls
**Result:** No more crashes!

**Files Modified:**
- [`src/ffproj/config.py`](src/ffproj/config.py) - Default params
- [`src/ffproj/models_gbm.py`](src/ffproj/models_gbm.py) - All predict() calls
- [`src/ffproj/backtest.py`](src/ffproj/backtest.py) - Spawn multiprocessing
- [`src/ffproj/train.py`](src/ffproj/train.py) - Entry point

---

## 📁 Project Structure

```
ff_ppr/
├── src/
│   ├── ffproj/
│   │   ├── data.py              # NFL data loading (nflreadpy)
│   │   ├── features.py          # Feature engineering (FIXED!)
│   │   ├── models_gbm.py        # LightGBM quantile models (FIXED!)
│   │   ├── backtest.py          # Rolling window backtest (FIXED!)
│   │   ├── train.py             # Training script
│   │   ├── metrics.py           # Evaluation metrics
│   │   ├── scoring.py           # PPR/Half-PPR scoring
│   │   └── config.py            # Configuration
│   └── app/
│       └── streamlit_app.py     # Web dashboard
│
├── data/
│   ├── interim/                 # Cached raw data
│   └── processed/               # Feature-engineered data & predictions
│
├── experiments/                 # Backtest results
│
├── working_example.py           # ✅ Complete working demo
├── run_backtest_simple.py       # ✅ Full backtest runner
├── prepare_app_data.py          # Prepare data for Streamlit
├── launch_app.sh                # Launch web app
├── test_lgb_minimal.py          # Test single model
└── debug_features.py            # Test feature engineering
```

---

## 🎯 How to Use

### Option 1: Use Existing Predictions (Fastest)

```bash
# Prepare app data from backtest results
python prepare_app_data.py

# Launch the app
./launch_app.sh
```

### Option 2: Train Your Own Models

```python
import sys
sys.path.insert(0, 'src')

from ffproj.data import load_weekly_stats
from ffproj.features import build_all_features, select_feature_columns
from ffproj.models_gbm import GBMQuantileEnsemble

# 1. Load data
df = load_weekly_stats([2024, 2025], cache=True)
df = df.sort_values(['player_id', 'season', 'week']).reset_index(drop=True)

# 2. Build features
df = build_all_features(df)
feature_cols = select_feature_columns(df)

# 3. Split
train = df[df['season'] == 2024]
val = df[df['season'] == 2025]

X_train = train[feature_cols].fillna(0)
y_train = train['fp_ppr']
X_val = val[feature_cols].fillna(0)
y_val = val['fp_ppr']

# 4. Train
model = GBMQuantileEnsemble(verbose=True)
model.train(X_train, y_train, X_val, y_val)

# 5. Predict
preds = model.predict(X_val)
print(preds['q50'][:20])  # Top 20 predictions
```

### Option 3: Run Full Backtest

```bash
# Simple version (recommended)
python run_backtest_simple.py

# Or using the train module
python -m src.ffproj.train --backtest --seasons 2023 2024 2025
```

---

## 📈 Feature Importance

The model found these features most predictive:

| Feature | Description | Importance |
|---------|-------------|------------|
| `f_fp_ppr_roll5` | 5-game rolling average fantasy points | 4918 |
| `f_fp_ppr_roll3` | 3-game rolling average | 465 |
| `f_target_share_5w` | 5-week target share | 434 |
| `f_opp_fp_allowed_pos_5w` | Opponent strength vs position | 347 |
| `f_fp_ppr_roll1` | Last game fantasy points | 321 |

---

## 🔍 Validation

### Test Scripts (All Pass ✅)

```bash
# Test feature engineering only
python debug_features.py
# ✅ Output: 2,061 rows, 163 cols, 2.9 MB

# Test single model training
python test_lgb_minimal.py
# ✅ Output: MAE 4.57, no crashes

# Test full pipeline
python working_example.py
# ✅ Output: Complete end-to-end success

# Test backtest
python run_backtest_simple.py
# ✅ Output: 28 folds, all metrics computed
```

---

## ⚙️ Configuration

Edit [`src/ffproj/config.py`](src/ffproj/config.py) to customize:

```python
# Seasons to train on
TRAIN_SEASONS = list(range(2019, 2026))  # 2019-2025

# Rolling windows for features
ROLLING_WINDOWS = [1, 3, 5]

# Backtest settings
BACKTEST_TRAIN_WEEKS = 8  # Train on last 8 weeks

# Model hyperparameters
GBM_PARAMS = {
    "learning_rate": 0.05,
    "num_leaves": 63,
    "num_threads": 1,  # IMPORTANT: Keep at 1 for macOS stability
}
```

---

## 📚 Data Source

The project uses **nflreadpy** (modern replacement for deprecated nfl-data-py):

```python
from nflreadpy import load_player_stats

# Load weekly player stats
df = load_player_stats([2024, 2025]).to_pandas()
```

**Available Data:**
- Weekly player stats (2019-2025)
- Fantasy points (PPR, Half-PPR, Standard)
- Volume metrics (targets, carries, snaps, routes)
- Efficiency metrics (yards per route, YAC, etc.)
- Team/opponent context

---

## 🐛 Troubleshooting

### App shows "No predictions available"
**Solution:** Run `python prepare_app_data.py` to copy backtest results

### Python crashes with "segmentation fault"
**Solution:** Make sure you've cleared the Python cache:
```bash
rm -rf src/ffproj/__pycache__
```

### "MemoryError: Unable to allocate 80TB"
**Solution:** You have an old cached version of features.py. Clear cache:
```bash
rm -rf src/ffproj/__pycache__
```

### Feature engineering takes too long
**Solution:** Use cached data:
```python
df = load_weekly_stats([2024, 2025], cache=True)  # Set cache=True
```

---

## 🎓 Understanding the Model

### Quantile Regression
Instead of predicting a single value, we predict three quantiles:
- **P10**: 10th percentile (low-end outcome)
- **P50**: Median (most likely outcome)
- **P90**: 90th percentile (high-end outcome)

This gives you **uncertainty bands** for risk assessment.

### Example Prediction
```
Player: Patrick Mahomes
P10: 14.2 pts  ← Floor
P50: 22.3 pts  ← Projection (use this)
P90: 31.8 pts  ← Ceiling
```

**Interpretation:**
- Start Mahomes with confidence (high floor + ceiling)
- 80% chance he scores between 14-32 points
- Use P50 (22.3) for expected value calculations

---

## 📝 Next Steps

1. **Integrate with your fantasy platform**: Export CSVs from the app
2. **Combine with other data**: Add injury reports, weather, etc.
3. **Train on more seasons**: Extend `TRAIN_SEASONS` in config
4. **Add deep learning models**: TCN/LSTM models are stubbed in `models_dl.py`
5. **Deploy to cloud**: Containerize with Docker and host on AWS/GCP

---

## 🙏 Credits

**Data:** nflreadpy (https://github.com/nflreadpy/nflreadpy)
**Models:** LightGBM (https://github.com/microsoft/LightGBM)
**App:** Streamlit (https://streamlit.io)

---

## ⚠️ Disclaimer

These projections are for **informational and educational purposes only**. Always combine with:
- Your own research and analysis
- Latest injury reports and news
- Expert consensus rankings
- Game script and matchup considerations

Fantasy football involves luck and uncertainty. No model is perfect!

---

## 📞 Support

**Files to check if something goes wrong:**
- [`FIXES_COMPLETED.md`](FIXES_COMPLETED.md) - Detailed fix documentation
- [`working_example.py`](working_example.py) - Reference implementation
- [`test_lgb_minimal.py`](test_lgb_minimal.py) - Minimal test case

**Common issues and solutions are in the Troubleshooting section above.**

---

**Enjoy your fantasy football projections! 🏈🎉**
