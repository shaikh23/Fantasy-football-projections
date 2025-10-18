# Quick Start Guide

Get up and running with fantasy football projections in minutes.

## Installation

```bash
# 1. Navigate to project directory
cd ff_ppr

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

## Usage

### Option 1: Interactive Notebook (Recommended for First Time)

```bash
# Launch Jupyter
jupyter notebook

# Open notebooks/00_quickstart.ipynb
# Run cells to see the full workflow
```

### Option 2: Command Line Training

```bash
# Train model with rolling backtest (most realistic evaluation)
python -m src.ffproj.train --model gbm --backtest --seasons 2019 2020 2021 2022 2023

# This will:
# - Download NFL data using nfl-data-py
# - Build features (rolling stats, opponent strength, game context)
# - Run rolling window backtest
# - Save results to experiments/ directory
```

**Note**: First run will download ~500MB of NFL data and take 5-10 minutes.

### Option 3: Quick Train/Val Split

```bash
# Train on 2019-2022, validate on 2023
python -m src.ffproj.train --model gbm --seasons 2019 2020 2021 2022 --val-season 2023

# Faster than backtest, but less realistic
```

### Option 4: Streamlit App

```bash
# Launch web app
streamlit run src/app/streamlit_app.py

# Open browser to http://localhost:8501
# Note: Requires predictions to be generated first (Option 2 or 3)
```

## Expected Output

After running a backtest, you'll see:

```
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
```

Results saved to:
- `experiments/gbm_YYYYMMDD_HHMMSS/backtest_predictions.parquet`
- `experiments/gbm_YYYYMMDD_HHMMSS/backtest_metrics.json`

## Project Structure

```
ff_ppr/
├── README.md              # Full documentation
├── QUICKSTART.md         # This file
├── requirements.txt      # Dependencies
├── .gitignore
├── data/                 # Data (created on first run)
│   ├── raw/             # Downloaded NFL data
│   ├── interim/         # Cleaned data
│   └── processed/       # Features + predictions
├── notebooks/
│   └── 00_quickstart.ipynb   # Interactive tutorial
├── src/
│   ├── ffproj/          # Main package
│   │   ├── config.py         # Settings
│   │   ├── data.py           # Data loading
│   │   ├── features.py       # Feature engineering
│   │   ├── scoring.py        # PPR/Half-PPR scoring
│   │   ├── models_gbm.py     # LightGBM models
│   │   ├── models_dl.py      # Deep learning (TCN/LSTM)
│   │   ├── backtest.py       # Rolling backtest
│   │   ├── metrics.py        # Evaluation metrics
│   │   ├── train.py          # Training CLI
│   │   ├── utils.py          # Utilities
│   │   └── io.py             # I/O functions
│   └── app/
│       └── streamlit_app.py  # Web dashboard
└── experiments/          # Saved models/results (created on train)
```

## Common Issues

### 1. "nfl_data_py not found"
```bash
pip install nfl-data-py
```

### 2. "No module named ffproj"
```bash
# Make sure you're in the ff_ppr directory
cd ff_ppr

# Use module syntax with src prefix
python -m src.ffproj.train --help
```

### 3. "No predictions available" in Streamlit
You need to run training first to generate predictions:
```bash
python -m src.ffproj.train --model gbm --backtest
```

### 4. Out of memory during training
Reduce data size or increase swap:
```bash
# Use fewer seasons
python -m src.ffproj.train --model gbm --seasons 2022 2023 --backtest
```

## What's Next?

1. **View Results**: Check `experiments/` directory for model outputs
2. **Customize**: Edit `src/ffproj/config.py` to change hyperparameters
3. **Try Deep Learning**: Train TCN model (requires more data/time)
   ```bash
   python -m src.ffproj.train --model tcn --seasons 2019 2020 2021 2022 --val-season 2023
   ```
4. **Build Applications**:
   - Waiver wire recommender
   - DFS lineup optimizer
   - Trade value calculator

## Performance Benchmarks

On a typical laptop (4 cores, 16GB RAM):

| Task | Time |
|------|------|
| First data download | 5-10 min |
| Feature engineering (5 seasons) | 2-3 min |
| GBM training (single split) | 1-2 min |
| GBM backtest (rolling) | 10-15 min |
| TCN training | 15-30 min |

## Need Help?

- Check the full [README.md](README.md) for detailed documentation
- Review [notebooks/00_quickstart.ipynb](notebooks/00_quickstart.ipynb) for examples
- Open an issue on GitHub

## Key CLI Options

```bash
# Training options
python -m src.ffproj.train \
  --model {gbm,tcn,lstm}           # Model type
  --seasons 2019 2020 2021         # Training seasons
  --val-season 2023                # Validation season
  --backtest                       # Use rolling backtest
  --rebuild-features               # Force rebuild features
  --output-dir path/to/output      # Custom output directory
  --seed 42                        # Random seed

# Examples
python -m src.ffproj.train --model gbm --backtest
python -m src.ffproj.train --model gbm --seasons 2020 2021 2022 --val-season 2023
python -m src.ffproj.train --model tcn --backtest --rebuild-features
```

---

**Ready to start?** Run Option 1 (notebook) or Option 2 (backtest) above!
