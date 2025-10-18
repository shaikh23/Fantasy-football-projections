"""
Configuration settings for the fantasy football projections project.
"""
from pathlib import Path
from typing import List, Dict, Any

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"

# Ensure directories exist
for d in [RAW_DATA_DIR, INTERIM_DATA_DIR, PROCESSED_DATA_DIR, EXPERIMENTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Scoring systems
SCORING_SYSTEMS = ["PPR", "HALF_PPR", "STANDARD"]
DEFAULT_SCORING = "PPR"

# Positions to model
POSITIONS = ["QB", "RB", "WR", "TE"]

# Season range
TRAIN_SEASONS = list(range(2019, 2026))  # 2019-2025 for training
TEST_SEASON = 2025  # Current season (can be included in training)

# Feature configuration
ROLLING_WINDOWS = [1, 3, 5]  # Game windows for rolling features

# Volume features to compute rolling stats
VOLUME_FEATURES = [
    "rush_att",
    "targets",
    "rec",
    "routes",
    "snaps",
    "snap_pct",
    "route_pct",
    "target_share",
    "air_yards",
    "rz_touches"
]

# Efficiency features to compute rolling stats
EFFICIENCY_FEATURES = [
    "yprr",  # yards per route run
    "yac_per_rec",  # yards after catch per reception
    "yco_per_att",  # yards after contact per attempt
    "broken_tackles_per_touch"
]

# Team context features
TEAM_CONTEXT_FEATURES = [
    "team_pass_rate",
    "pace_sec_play",
    "implied_points",
    "spread",
    "dome",
    "temp",
    "wind"
]

# Model hyperparameters
GBM_PARAMS = {
    "objective": "quantile",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "min_data_in_leaf": 40,
    "verbose": -1,
    "num_threads": 1,  # Single-threaded to avoid segfault on macOS
    "force_row_wise": True  # Stability on macOS with multiprocessing
}

GBM_QUANTILES = [0.1, 0.5, 0.9]  # P10, P50 (median), P90
GBM_NUM_BOOST_ROUND = 500
GBM_EARLY_STOPPING = 50

# Deep learning hyperparameters
DL_EMBEDDING_DIM = 16
DL_HIDDEN_DIM = 128
DL_TCN_KERNEL = 3
DL_TCN_LEVELS = 3
DL_SEQUENCE_LENGTH = 8  # weeks of history
DL_BATCH_SIZE = 64
DL_LEARNING_RATE = 0.001
DL_NUM_EPOCHS = 100
DL_PATIENCE = 10

# Backtest configuration
BACKTEST_TRAIN_WEEKS = 8
BACKTEST_VAL_WEEKS = 1
MIN_TRAIN_SAMPLES = 100
MIN_VAL_SAMPLES = 10

# Random seed
RANDOM_SEED = 42
