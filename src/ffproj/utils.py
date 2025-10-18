"""
General utility functions.
"""
import numpy as np
import pandas as pd
import random
from typing import Optional
import torch
from datetime import datetime
import pytz
import nflreadpy as nfl


def set_random_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility.

    Parameters
    ----------
    seed : int
        Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Get PyTorch device (cuda if available, else cpu)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def enforce_monotonic_quantiles(
    q10: np.ndarray,
    q50: np.ndarray,
    q90: np.ndarray
) -> tuple:
    """
    Enforce monotonic ordering: q10 <= q50 <= q90.

    Parameters
    ----------
    q10 : np.ndarray
        10th percentile predictions
    q50 : np.ndarray
        50th percentile (median) predictions
    q90 : np.ndarray
        90th percentile predictions

    Returns
    -------
    tuple
        (q10, q50, q90) with enforced monotonicity
    """
    q10 = np.asarray(q10)
    q50 = np.asarray(q50)
    q90 = np.asarray(q90)

    # Ensure q50 >= q10
    q50 = np.maximum(q50, q10)
    # Ensure q90 >= q50
    q90 = np.maximum(q90, q50)

    return q10, q50, q90


def rolling_mean(
    df: pd.DataFrame,
    columns: list,
    window: int,
    groupby: list,
    min_periods: int = 1
) -> pd.DataFrame:
    """
    Compute rolling mean for specified columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    columns : list
        Columns to compute rolling stats
    window : int
        Rolling window size
    groupby : list
        Columns to group by (e.g., ['player_id'])
    min_periods : int
        Minimum periods for rolling calculation

    Returns
    -------
    pd.DataFrame
        DataFrame with added rolling mean columns
    """
    df = df.copy()

    for col in columns:
        if col in df.columns:
            new_col = f"{col}_roll{window}"
            df[new_col] = df.groupby(groupby)[col].transform(
                lambda x: x.rolling(window=window, min_periods=min_periods).mean()
            )

    return df


def lag_features(
    df: pd.DataFrame,
    columns: list,
    lags: list,
    groupby: list
) -> pd.DataFrame:
    """
    Create lagged features.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    columns : list
        Columns to lag
    lags : list
        List of lag periods (e.g., [1, 2, 3])
    groupby : list
        Columns to group by

    Returns
    -------
    pd.DataFrame
        DataFrame with lagged features
    """
    df = df.copy()

    for col in columns:
        if col in df.columns:
            for lag in lags:
                new_col = f"{col}_lag{lag}"
                df[new_col] = df.groupby(groupby)[col].shift(lag)

    return df


def exponential_moving_average(
    series: pd.Series,
    span: int
) -> pd.Series:
    """
    Compute exponential moving average.

    Parameters
    ----------
    series : pd.Series
        Input series
    span : int
        EMA span

    Returns
    -------
    pd.Series
        Exponential moving average
    """
    return series.ewm(span=span, adjust=False).mean()


def normalize_player_name(name: str) -> str:
    """
    Normalize player names for consistent matching.

    Parameters
    ----------
    name : str
        Player name

    Returns
    -------
    str
        Normalized name (lowercase, stripped, no punctuation)
    """
    if pd.isna(name):
        return ""

    # Remove punctuation and convert to lowercase
    name = str(name).lower()
    for char in ['.', ',', "'", '"', '-']:
        name = name.replace(char, '')

    return name.strip()


def get_week_year_from_game_id(game_id: str) -> tuple:
    """
    Extract week and year from NFL game ID.

    Parameters
    ----------
    game_id : str
        NFL game ID (format: YYYY_WW_AWAY_HOME)

    Returns
    -------
    tuple
        (season, week)
    """
    try:
        parts = game_id.split('_')
        season = int(parts[0])
        week = int(parts[1])
        return season, week
    except:
        return None, None


def split_train_val_test(
    df: pd.DataFrame,
    train_seasons: list,
    val_seasons: list,
    test_seasons: list,
    season_col: str = 'season'
) -> tuple:
    """
    Split data by season.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    train_seasons : list
        Training season years
    val_seasons : list
        Validation season years
    test_seasons : list
        Test season years
    season_col : str
        Name of season column

    Returns
    -------
    tuple
        (train_df, val_df, test_df)
    """
    train_df = df[df[season_col].isin(train_seasons)].copy()
    val_df = df[df[season_col].isin(val_seasons)].copy()
    test_df = df[df[season_col].isin(test_seasons)].copy()

    return train_df, val_df, test_df


def get_feature_columns(df: pd.DataFrame, prefix: str = "f_") -> list:
    """
    Get feature columns by prefix.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    prefix : str
        Feature column prefix

    Returns
    -------
    list
        List of feature column names
    """
    return [col for col in df.columns if col.startswith(prefix)]


def clip_outliers(
    series: pd.Series,
    lower_percentile: float = 0.01,
    upper_percentile: float = 0.99
) -> pd.Series:
    """
    Clip outliers to percentile bounds.

    Parameters
    ----------
    series : pd.Series
        Input series
    lower_percentile : float
        Lower percentile bound (0-1)
    upper_percentile : float
        Upper percentile bound (0-1)

    Returns
    -------
    pd.Series
        Series with outliers clipped
    """
    lower = series.quantile(lower_percentile)
    upper = series.quantile(upper_percentile)
    return series.clip(lower, upper)


# ==============================================================================
# Adaptive Weekly Prediction Utilities
# ==============================================================================

ET = pytz.timezone("America/New_York")


def nfl_now_et() -> datetime:
    """
    Get current time in Eastern Time (NFL's timezone).

    Returns
    -------
    datetime
        Current datetime in ET with timezone info
    """
    return datetime.now(tz=ET)


def get_current_season_and_week() -> tuple:
    """
    Determine current NFL season, latest completed week, and in-progress week.

    Uses game schedule to determine:
    - Current season
    - Latest completed week (all games have results)
    - In-progress week (some games started but not all finished)

    Returns
    -------
    tuple
        (season, latest_done_week, in_progress_week, schedule_df)
        - season: int, current NFL season year
        - latest_done_week: int, last week where all games finished
        - in_progress_week: int or None, week currently being played (if any)
        - schedule_df: pd.DataFrame, full regular season schedule
    """
    # Load current season schedule (use 2024 as current season)
    # nflreadpy requires integer seasons, not "current"
    current_season = 2024
    sched = nfl.load_schedules(seasons=[current_season])

    # Convert to pandas for easier manipulation
    reg = sched.filter(sched["game_type"] == "REG").to_pandas()

    # Get latest season
    season = int(reg["season"].max())

    # Find latest completed week (all games have results)
    finished = reg[reg["result"].notna()]
    latest_done_wk = int(finished["week"].max()) if len(finished) > 0 else 0

    # Check if current week is in progress
    now = nfl_now_et()
    in_progress_week = None

    # Check if any games in the next week have started
    candidate_week = latest_done_wk + 1
    if candidate_week <= 18:  # Regular season is weeks 1-18
        week_games = reg[reg["week"] == candidate_week].copy()
        if len(week_games) > 0:
            # Convert game times to ET
            week_games["start_et"] = pd.to_datetime(
                week_games["gameday"], errors="coerce"
            ).dt.tz_localize(ET)

            # Check if any game has started
            started = week_games[week_games["start_et"] <= now]
            # Check if all games are finished
            all_finished = week_games["result"].notna().all()

            if len(started) > 0 and not all_finished:
                in_progress_week = candidate_week

    return season, latest_done_wk, in_progress_week, reg


def remaining_games(season: int, week: int, reg_sched: pd.DataFrame) -> pd.DataFrame:
    """
    Get games that haven't kicked off yet for a given week.

    Parameters
    ----------
    season : int
        NFL season year
    week : int
        Week number
    reg_sched : pd.DataFrame
        Regular season schedule from nflreadpy

    Returns
    -------
    pd.DataFrame
        Games in the specified week that haven't kicked off yet
    """
    now = nfl_now_et()

    # Filter to specified week
    wk = reg_sched[
        (reg_sched["season"] == season) &
        (reg_sched["week"] == week)
    ].copy()

    # Convert game times to ET
    wk["start_et"] = pd.to_datetime(wk["gameday"], errors="coerce").dt.tz_localize(ET)

    # Return only games that haven't started
    return wk[wk["start_et"] > now]
