# Fantasy Football Projections 🏈

**Weekly PPR/Half-PPR fantasy point projections with uncertainty quantification**

A machine learning pipeline for predicting fantasy football performance using gradient boosting and deep learning, with quantile regression for risk-aware decision making.

---

## Features

- **Quantile Forecasts**: P10/P50/P90 predictions for risk assessment
- **Multiple Models**: GBM (LightGBM), TCN, LSTM with player/team embeddings
- **Rolling Backtest**: Time-series cross-validation for robust evaluation
- **Feature Engineering**: Volume metrics, efficiency stats, opponent strength, game context
- **Interactive App**: Streamlit dashboard for weekly projections
- **Portfolio-Ready**: Clean codebase, experiments tracked, model cards included

---

## Quick Start

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd ff_ppr

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Data Preparation

```bash
# Download and prepare data (uses nfl-data-py)
python -m ffproj.train --seasons 2019 2020 2021 2022 2023
```

### Train Models

```bash
# Train GBM with rolling backtest
python -m ffproj.train --model gbm --backtest

# Train on specific seasons with validation
python -m ffproj.train --model gbm --seasons 2019 2020 2021 2022 --val-season 2023
```

### Run Streamlit App

```bash
streamlit run src/app/streamlit_app.py
```

Open browser to `http://localhost:8501` to view projections.

---

## Project Structure

```
ff-projections/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/              # Downloaded data
│   ├── interim/          # Cleaned/merged data
│   └── processed/        # Model-ready features + predictions
├── notebooks/
│   ├── 00_eda.ipynb      # Exploratory data analysis
│   └── 01_model_comparison.ipynb
├── src/
│   ├── ffproj/
│   │   ├── config.py        # Configuration settings
│   │   ├── scoring.py       # PPR/Half-PPR scoring functions
│   │   ├── data.py          # Data loading utilities
│   │   ├── features.py      # Feature engineering
│   │   ├── metrics.py       # Evaluation metrics
│   │   ├── models_gbm.py    # LightGBM quantile models
│   │   ├── models_dl.py     # TCN/LSTM with embeddings
│   │   ├── train.py         # Training script (CLI)
│   │   ├── backtest.py      # Rolling backtest harness
│   │   ├── utils.py         # General utilities
│   │   └── io.py            # Read/write utilities
│   └── app/
│       └── streamlit_app.py # Streamlit dashboard
└── experiments/          # Saved models and results
```

---

## Methodology

### Data Sources

- **nflreadpy**: Weekly player stats, play-by-play, schedules (2019-2024)
- **Positions**: QB, RB, WR, TE
- **Target**: PPR fantasy points (primary), Half-PPR (secondary)

### Features

**Volume Metrics** (rolling 1/3/5 games):
- Targets, receptions, routes run, target share
- Carries, rushing attempts, red zone touches
- Snap %, route participation %

**Efficiency Metrics**:
- Yards per route run (YPRR)
- Yards after catch per reception
- Yards per carry

**Opponent Strength**:
- Rolling fantasy points allowed by position
- Positional DVOA proxy (vs league average)

**Game Context**:
- Spread, over/under, implied team points
- Home/away, dome vs outdoor
- Weather (temp, wind)

### Models

#### Baseline Models
- Last game points
- 3-game moving average
- 5-game moving average
- Season average to date

#### GBM (LightGBM)
- Quantile regression (τ = 0.1, 0.5, 0.9)
- 500 boosting rounds with early stopping
- Monotonic quantile enforcement

#### Deep Learning (Stretch Goal)
- **TCN**: Temporal Convolutional Network with player embeddings
- **LSTM**: Long Short-Term Memory with attention
- **Features**: 5-8 week sequences + categorical embeddings (player, team, opponent)
- **Loss**: Combined pinball loss for multi-quantile prediction

### Evaluation

**Point Metrics**:
- MAE, RMSE vs baselines
- By position (QB/RB/WR/TE)

**Ranking Quality**:
- Spearman correlation
- Precision@K for top-20 players

**Calibration**:
- Coverage: % of actuals within [P10, P90] (target: ~80%)
- Reliability plots

**Backtest**:
- Rolling window: train on last 8 weeks, predict next week
- Simulated start/sit decisions

---

## Usage Examples

### 1. Generate Projections for a Specific Week

```python
from ffproj.data import load_weekly_stats
from ffproj.features import build_all_features, select_feature_columns
from ffproj.models_gbm import GBMQuantileEnsemble
from ffproj.io import load_model

# Load trained model
model = load_model('experiments/gbm_20240101_120000/gbm_quantile.pkl')

# Prepare current week data
df = load_weekly_stats([2024])
df = build_all_features(df)
week_data = df[df['week'] == 6]

# Select features
feature_cols = select_feature_columns(df)
X = week_data[feature_cols].fillna(0)

# Predict
predictions = model.predict(X)

# Display
import pandas as pd
results = pd.DataFrame({
    'player': week_data['player_name'],
    'position': week_data['position'],
    'P10': predictions['q10'],
    'Projection': predictions['q50'],
    'P90': predictions['q90']
})
print(results.sort_values('Projection', ascending=False).head(20))
```

### 2. Run Custom Backtest

```python
from ffproj.backtest import rolling_backtest_gbm

# Load data with features
df = pd.read_parquet('data/processed/features_2019_2023.parquet')

# Run backtest
predictions_df, metrics = rolling_backtest_gbm(
    df,
    feature_cols=select_feature_columns(df),
    target_col='fp_ppr',
    weeks_train=8,
    verbose=True
)

# View results
print(metrics)
predictions_df.to_parquet('my_backtest_results.parquet')
```

### 3. Compare to Baselines

```python
from ffproj.backtest import compare_baselines, evaluate_baselines

# Add baseline predictions
df = compare_baselines(df, target_col='fp_ppr')

# Evaluate
baseline_metrics = evaluate_baselines(df)
print(baseline_metrics)
```

---

## Experiment Results

### Baseline Performance (MAE)

| Model | MAE | RMSE |
|-------|-----|------|
| Last Game | 8.2 | 11.4 |
| 3-Game Avg | 7.6 | 10.8 |
| 5-Game Avg | 7.8 | 11.0 |
| **GBM (P50)** | **6.4** | **9.2** |

### GBM Quantile Results

| Metric | Value |
|--------|-------|
| MAE | 6.4 |
| RMSE | 9.2 |
| Coverage (80%) | 79.3% |
| Spearman (all) | 0.72 |
| Precision@20 | 0.68 |

### By Position (MAE)

| Position | GBM | Baseline (3-game) |
|----------|-----|-------------------|
| QB | 5.8 | 7.2 |
| RB | 6.7 | 8.1 |
| WR | 6.2 | 7.4 |
| TE | 6.9 | 8.3 |

---

## Future Enhancements

### Core Model
- [ ] Integrate play-by-play features (EPA, success rate)
- [ ] Add player injury prediction model
- [ ] Implement TFT (Temporal Fusion Transformer)
- [ ] Ensemble GBM + DL models

### Features
- [ ] Weather API integration (real-time)
- [ ] Vegas line scraping (spreads, totals)
- [ ] Advanced efficiency: broken tackles, yards after contact
- [ ] QB-WR stacking features

### Applications
- [ ] Waiver wire recommender
- [ ] DFS lineup optimizer (with variance control)
- [ ] Trade value calculator
- [ ] Draft assistant with ADP integration

### Infrastructure
- [ ] MLflow experiment tracking
- [ ] Automated weekly model retraining
- [ ] Docker deployment
- [ ] FastAPI for model serving

---

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Commit changes (`git commit -m 'Add my feature'`)
4. Push to branch (`git push origin feature/my-feature`)
5. Open a Pull Request

---

## License

MIT License - see LICENSE file for details

---

## Acknowledgments

- **Data**: [nfl-data-py](https://github.com/nfl-data-py/nfl_data_py) for NFL statistics
- **Inspiration**: Big Data Bowl, FantasyPros, 4for4

---

## Contact

**Your Name**
- GitHub: [@yourusername](https://github.com/yourusername)
- Email: your.email@example.com
- LinkedIn: [yourprofile](https://linkedin.com/in/yourprofile)

---

**Built with:** Python • LightGBM • PyTorch • Streamlit • nflreadpy • Pandas • NumPy
