# Backtest Status - 2022-2025 Seasons

## Current Run

**Status**: ⏳ In Progress (15/72 folds completed - 21%)

**Dataset**:
- **Seasons**: 2022, 2023, 2024, 2025 (4 seasons)
- **Total player-weeks**: 20,357
- **Memory usage**: 29.0 MB ✅
- **Features**: 43

**Backtest Configuration**:
- **Total folds**: 72 (rolling window across all weeks)
- **Training window**: 8 weeks
- **Models per fold**: 3 (P10, P50, P90)
- **Total models to train**: 216 (72 folds × 3 quantiles)

**Estimated Time**:
- Based on ~2 minutes per fold
- Total time: ~2.5 hours
- Started: ~9:11 PM
- Expected completion: ~11:40 PM

## Expected Results

Once complete, you'll have predictions for:

| Season | Expected Player-Weeks | Weeks |
|--------|----------------------|-------|
| 2022 | ~6,092 | 1-22 |
| 2023 | ~6,081 | 2-22 |
| 2024 | ~6,123 | 1-22 |
| 2025 | ~2,061 | 1-6 |
| **Total** | **~20,357** | **88 weeks** |

## When Complete

The backtest will automatically:
1. Save predictions to `experiments/gbm_[timestamp]/backtest_predictions.parquet`
2. Save metrics to `experiments/gbm_[timestamp]/backtest_metrics.json`
3. Display final performance metrics by position

Then run:
```bash
python prepare_app_data.py
```

This will copy the predictions to the app data directory and create individual week files.

## Monitoring Progress

Check the log file:
```bash
tail -20 backtest_2022_2025.log
```

Or monitor progress updates (every 5 folds):
```bash
grep "Progress:" backtest_2022_2025.log
```

## Performance So Far

The feature engineering completed successfully:
- ✅ No memory explosion (29 MB for 20K rows)
- ✅ No segmentation faults
- ✅ All threading fixes working
- ✅ Models training normally

---

**You can safely let this run in the background!** The script will complete on its own and save all results.
