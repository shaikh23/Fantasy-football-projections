"""
Feature engineering for fantasy football projections.
"""
import pandas as pd
import numpy as np
from typing import List, Optional
from .config import ROLLING_WINDOWS, POSITIONS
from .utils import rolling_mean, lag_features


def create_rolling_features(
    df: pd.DataFrame,
    feature_cols: List[str],
    windows: List[int] = None,
    groupby_cols: List[str] = None
) -> pd.DataFrame:
    """
    Create rolling window features for volume and efficiency metrics.

    Parameters
    ----------
    df : pd.DataFrame
        Player data sorted by player_id and week
    feature_cols : list of str
        Columns to compute rolling stats
    windows : list of int, optional
        Rolling window sizes (default: [1, 3, 5])
    groupby_cols : list of str, optional
        Grouping columns (default: ['player_id'])

    Returns
    -------
    pd.DataFrame
        DataFrame with rolling features added (prefixed with f_)
    """
    if windows is None:
        windows = ROLLING_WINDOWS
    if groupby_cols is None:
        groupby_cols = ['player_id']

    df = df.copy()

    for window in windows:
        for col in feature_cols:
            if col in df.columns:
                new_col = f"f_{col}_roll{window}"
                df[new_col] = df.groupby(groupby_cols)[col].transform(
                    lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
                )

    return df


def create_lag_features(
    df: pd.DataFrame,
    feature_cols: List[str],
    lags: List[int] = None,
    groupby_cols: List[str] = None
) -> pd.DataFrame:
    """
    Create lagged features (previous week values).

    Parameters
    ----------
    df : pd.DataFrame
        Player data
    feature_cols : list of str
        Columns to lag
    lags : list of int, optional
        Lag periods (default: [1])
    groupby_cols : list of str, optional
        Grouping columns (default: ['player_id'])

    Returns
    -------
    pd.DataFrame
        DataFrame with lagged features
    """
    if lags is None:
        lags = [1]
    if groupby_cols is None:
        groupby_cols = ['player_id']

    df = df.copy()

    for lag in lags:
        for col in feature_cols:
            if col in df.columns:
                new_col = f"f_{col}_lag{lag}"
                df[new_col] = df.groupby(groupby_cols)[col].shift(lag)

    return df


def compute_opponent_strength(
    df: pd.DataFrame,
    target_col: str = 'fp_ppr',
    windows: List[int] = None
) -> pd.DataFrame:
    """
    Compute opponent positional strength (fantasy points allowed).

    For each team-week, calculate rolling average of fantasy points
    allowed to each position.

    Parameters
    ----------
    df : pd.DataFrame
        Player data with columns: season, week, opponent, position, fp_ppr
    target_col : str
        Target column (default: 'fp_ppr')
    windows : list of int, optional
        Rolling windows for opponent strength (default: [3, 5])

    Returns
    -------
    pd.DataFrame
        DataFrame with opponent strength features
    """
    if windows is None:
        windows = [3, 5]

    df = df.copy()

    # Group by opponent (defense), position, and compute rolling allowed points
    # Use transform to avoid cartesian product
    for window in windows:
        # Create a feature for points allowed by this opponent to this position
        df[f'f_opp_fp_allowed_pos_{window}w'] = (
            df.groupby(['season', 'opponent', 'position'])[target_col]
            .transform(lambda x: x.shift(1).rolling(window=window, min_periods=1).mean())
        )

    # League average by position (baseline comparison)
    pos_avg = df.groupby(['season', 'position'])[target_col].transform('mean')
    for window in windows:
        col = f'f_opp_fp_allowed_pos_{window}w'
        if col in df.columns:
            df[f'f_opp_strength_vs_avg_{window}w'] = df[col] - pos_avg

    return df


def compute_team_context_features(
    df: pd.DataFrame,
    windows: List[int] = None
) -> pd.DataFrame:
    """
    Compute team-level context features (pace, pass rate, etc.).

    Parameters
    ----------
    df : pd.DataFrame
        Player data with team column
    windows : list of int, optional
        Rolling windows (default: [3, 5])

    Returns
    -------
    pd.DataFrame
        DataFrame with team context features
    """
    if windows is None:
        windows = [3, 5]

    df = df.copy()

    # Team pass rate (if available)
    if 'pass_att' in df.columns and 'rush_att' in df.columns:
        df['team_plays'] = df['pass_att'] + df['rush_att']
        df['team_pass_rate'] = df['pass_att'] / (df['team_plays'] + 1e-6)

        for window in windows:
            df[f'f_team_pass_rate_{window}w'] = df.groupby(['season', 'team'])['team_pass_rate'].transform(
                lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
            )

    # Team pace (plays per game) - would need play-by-play data
    # Placeholder for when PBP data is integrated

    return df


def compute_target_share_features(
    df: pd.DataFrame,
    windows: List[int] = None
) -> pd.DataFrame:
    """
    Compute target share and touch share for skill position players.

    Parameters
    ----------
    df : pd.DataFrame
        Player data with targets and carries
    windows : list of int, optional
        Rolling windows (default: [3, 5])

    Returns
    -------
    pd.DataFrame
        DataFrame with share features
    """
    if windows is None:
        windows = [3, 5]

    df = df.copy()

    # Compute team totals for targets and carries
    team_totals = df.groupby(['season', 'week', 'team']).agg({
        'targets': 'sum',
        'carries': 'sum'
    }).rename(columns={
        'targets': 'team_targets',
        'carries': 'team_carries'
    })

    df = df.merge(team_totals, on=['season', 'week', 'team'], how='left')

    # Compute shares
    df['target_share'] = df['targets'] / (df['team_targets'] + 1e-6)
    df['carry_share'] = df['carries'] / (df['team_carries'] + 1e-6)

    # Rolling shares
    for window in windows:
        df[f'f_target_share_{window}w'] = df.groupby('player_id')['target_share'].transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
        )
        df[f'f_carry_share_{window}w'] = df.groupby('player_id')['carry_share'].transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
        )

    return df


def compute_efficiency_features(
    df: pd.DataFrame,
    windows: List[int] = None
) -> pd.DataFrame:
    """
    Compute efficiency metrics (yards per route, yards after catch, etc.).

    Parameters
    ----------
    df : pd.DataFrame
        Player data
    windows : list of int, optional
        Rolling windows

    Returns
    -------
    pd.DataFrame
        DataFrame with efficiency features
    """
    if windows is None:
        windows = [3, 5]

    df = df.copy()

    # Yards per route run (if routes column exists)
    if 'routes' in df.columns and 'receiving_yards' in df.columns:
        df['yprr'] = df['receiving_yards'] / (df['routes'] + 1e-6)

        for window in windows:
            df[f'f_yprr_{window}w'] = df.groupby('player_id')['yprr'].transform(
                lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
            )

    # Yards after catch per reception
    if 'receiving_yards_after_catch' in df.columns and 'receptions' in df.columns:
        df['yac_per_rec'] = df['receiving_yards_after_catch'] / (df['receptions'] + 1e-6)

        for window in windows:
            df[f'f_yac_per_rec_{window}w'] = df.groupby('player_id')['yac_per_rec'].transform(
                lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
            )

    # Yards per carry
    if 'rushing_yards' in df.columns and 'carries' in df.columns:
        df['ypc'] = df['rushing_yards'] / (df['carries'] + 1e-6)

        for window in windows:
            df[f'f_ypc_{window}w'] = df.groupby('player_id')['ypc'].transform(
                lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
            )

    return df


def add_game_context_features(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Add game context features (spread, total, weather, dome, etc.).

    Parameters
    ----------
    df : pd.DataFrame
        Player data with merged schedule info

    Returns
    -------
    pd.DataFrame
        DataFrame with game context features (prefixed with f_)
    """
    df = df.copy()

    # Spread and total as features
    if 'spread' in df.columns:
        df['f_spread'] = df['spread']

    if 'total' in df.columns:
        df['f_total'] = df['total']
        # Implied team points = (total / 2) + (spread / 2)
        df['f_implied_points'] = (df['total'] / 2) + (df['spread'] / 2)

    # Dome indicator
    if 'roof' in df.columns:
        df['f_dome'] = (df['roof'].isin(['dome', 'closed'])).astype(int)

    # Weather
    if 'temp' in df.columns:
        df['f_temp'] = df['temp'].fillna(70)  # Fill missing with neutral temp

    if 'wind' in df.columns:
        df['f_wind'] = df['wind'].fillna(0)

    # Home/away
    if 'home_away' in df.columns:
        df['f_is_home'] = (df['home_away'] == 'home').astype(int)

    return df


def add_positional_dummies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add position dummy variables.

    Parameters
    ----------
    df : pd.DataFrame
        Player data with position column

    Returns
    -------
    pd.DataFrame
        DataFrame with position dummies (f_pos_QB, etc.)
    """
    df = df.copy()

    for pos in POSITIONS:
        df[f'f_pos_{pos}'] = (df['position'] == pos).astype(int)

    return df


def build_all_features(
    df: pd.DataFrame,
    volume_cols: List[str] = None,
    windows: List[int] = None
) -> pd.DataFrame:
    """
    Build all features for modeling.

    This is the main feature engineering pipeline that combines all
    feature creation functions.

    Parameters
    ----------
    df : pd.DataFrame
        Raw player weekly data (must be sorted by player_id, season, week)
    volume_cols : list of str, optional
        Volume columns to use for rolling features
    windows : list of int, optional
        Rolling windows to use

    Returns
    -------
    pd.DataFrame
        DataFrame with all features (columns prefixed with f_)
    """
    if volume_cols is None:
        volume_cols = [
            'targets', 'receptions', 'receiving_yards',
            'carries', 'rushing_yards',
            'passing_yards', 'passing_tds',
            'fp_ppr'  # Include past fantasy points
        ]

    if windows is None:
        windows = ROLLING_WINDOWS

    def log_shape(tag):
        """Helper to log dataframe shape and memory"""
        mem_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
        print(f"  [{tag}] Shape: {df.shape[0]:,} rows × {df.shape[1]:,} cols | Memory: {mem_mb:.1f} MB")

    print("Building features...")
    log_shape("START")

    # 1. Rolling features for volume stats
    print("  → Rolling features...")
    df = create_rolling_features(df, volume_cols, windows=windows)
    log_shape("after_rolling")

    # 2. Target/carry share
    print("  → Target/carry share...")
    df = compute_target_share_features(df, windows=windows)
    log_shape("after_target_share")

    # 3. Efficiency metrics
    print("  → Efficiency metrics...")
    df = compute_efficiency_features(df, windows=windows)
    log_shape("after_efficiency")

    # 4. Opponent strength
    print("  → Opponent strength...")
    df = compute_opponent_strength(df, windows=windows)
    log_shape("after_opponent")

    # 5. Team context
    print("  → Team context...")
    df = compute_team_context_features(df, windows=windows)
    log_shape("after_team_context")

    # 6. Game context (spread, total, weather)
    print("  → Game context...")
    df = add_game_context_features(df)
    log_shape("after_game_context")

    # 7. Position dummies
    print("  → Position dummies...")
    df = add_positional_dummies(df)
    log_shape("FINAL")

    feature_cols = [c for c in df.columns if c.startswith('f_')]
    print(f"✓ Feature engineering complete: {len(feature_cols)} features created")

    return df


def select_feature_columns(df: pd.DataFrame, prefix: str = 'f_') -> List[str]:
    """
    Select feature columns by prefix.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with features
    prefix : str
        Feature prefix (default: 'f_')

    Returns
    -------
    list of str
        Feature column names
    """
    feature_cols = [c for c in df.columns if c.startswith(prefix)]
    return sorted(feature_cols)
