# Training Status & Progress

## 🔄 Currently Running

**Command**:
```bash
python -m src.ffproj.train --backtest --seasons 2019 2020 2021 2022 2023 2024 2025
```

**Started**: October 16, 2024 ~11:50 PM

**Expected Duration**: 20-30 minutes (first run with data download)

---

## 📊 What's Happening

### Phase 1: Data Download (5-10 min)
- Downloading 7 seasons (2019-2025) from nflreadpy
- ~80,000+ player-weeks
- ~100MB total download
- **Status**: In Progress... ⏳

### Phase 2: Feature Engineering (5-7 min)
- Building rolling windows (1/3/5 games)
- Computing opponent strength
- Creating efficiency metrics
- **Status**: Waiting...

### Phase 3: Rolling Backtest (10-15 min)
- Training models week-by-week
- ~50+ individual model trainings
- Each week: train on past 8 weeks → predict next week
- **Status**: Waiting...

### Phase 4: Results (< 1 min)
- Save predictions to parquet
- Save metrics to JSON
- Display summary
- **Status**: Waiting...

---

## 🔍 How to Monitor Progress

### Check if still running
```bash
ps aux | grep "python -m src.ffproj.train"
```

### View output (when available)
The script is running in the background. Output will be available when it completes or you can check the background process.

### Check data cache (shows download progress)
```bash
ls -lh data/interim/
```

You should see:
- `weekly_stats_2019_2025.parquet` (~50-80 MB when complete)

### Check for results
```bash
ls -la experiments/
```

When complete, you'll see a directory like:
- `experiments/gbm_20241016_235000/`

---

## ⏱️ Timeline Estimate

| Time | Milestone |
|------|-----------|
| **0-10 min** | Downloading data (7 seasons) |
| **10-15 min** | Feature engineering |
| **15-25 min** | Rolling backtest training |
| **25-30 min** | Saving results & summary |
| **30 min** | ✅ COMPLETE |

Current time: ~1-2 minutes in (data downloading)

---

## 📁 Expected Output

When complete, you'll find:

```
experiments/gbm_YYYYMMDD_HHMMSS/
├── config.json                      # Run configuration
├── backtest_predictions.parquet     # All weekly predictions
├── backtest_metrics.json            # Overall metrics
└── baseline_metrics.json            # Baseline comparison
```

### Metrics You'll See

```json
{
  "MAE": 6.4,                 // Mean Absolute Error
  "RMSE": 9.2,                // Root Mean Squared Error
  "Coverage_80": 0.79,        // 80% prediction interval coverage
  "Interval_Width": 12.3,     // Average uncertainty width
  "QB_MAE": 5.8,              // Position-specific metrics
  "RB_MAE": 6.7,
  "WR_MAE": 6.2,
  "TE_MAE": 6.9
}
```

---

## 🚨 If Something Goes Wrong

### Training fails
```bash
# Check error output
python -m src.ffproj.train --backtest --seasons 2019 2020 2021 2022 2023 2024 2025
```

### Out of memory
```bash
# Use fewer seasons
python -m src.ffproj.train --backtest --seasons 2022 2023 2024 2025
```

### Slow download
- Normal for first run
- Subsequent runs use cached data (~10x faster)

### Feature engineering error
```bash
# Clear cache and retry
rm -f data/interim/*.parquet
python -m src.ffproj.train --backtest --rebuild-features --seasons 2019 2020 2021 2022 2023 2024 2025
```

---

## ✅ When Complete

You'll see output like:
```
============================================================
Backtest Results
============================================================
MAE                                  6.4
RMSE                                 9.2
Coverage_80                          0.79
Interval_Width                       12.3
QB_MAE                              5.8
RB_MAE                              6.7
WR_MAE                              6.2
TE_MAE                              6.9
============================================================

Results saved to: experiments/gbm_20241016_235000
```

### Next Steps After Completion

1. **View Results**
   ```bash
   python -c "
   import pandas as pd
   preds = pd.read_parquet('experiments/gbm_*/backtest_predictions.parquet')
   print(preds.head(20))
   "
   ```

2. **Launch Streamlit App**
   ```bash
   streamlit run src/app/streamlit_app.py
   ```

3. **Analyze Top Predictions**
   ```python
   import pandas as pd
   preds = pd.read_parquet('experiments/gbm_*/backtest_predictions.parquet')

   # Week 1 of 2025 top projections
   week1 = preds[(preds['season']==2025) & (preds['week']==1)]
   week1_sorted = week1.sort_values('q50', ascending=False)
   print(week1_sorted[['player_id', 'position', 'q50', 'y_true']].head(20))
   ```

---

## 📊 Training Progress Checklist

- [x] Project setup complete
- [x] Dependencies installed
- [x] nflreadpy configured
- [ ] Data download (2019-2025) - **IN PROGRESS**
- [ ] Feature engineering
- [ ] Rolling backtest training
- [ ] Results saved
- [ ] Ready for predictions

---

**Status**: 🔄 Training in progress...
**Estimated completion**: ~25-30 minutes from start
**Check back at**: ~12:15-12:20 AM

*This file will be updated when training completes.*
