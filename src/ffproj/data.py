"""
Data loading and preprocessing functions.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple
import warnings

try:
    import nflreadpy as nfl
    NFL_DATA_AVAILABLE = True
except ImportError:
    NFL_DATA_AVAILABLE = False
    warnings.warn("nflreadpy not installed. Some data loading features may be unavailable.")

from .config import RAW_DATA_DIR, INTERIM_DATA_DIR, POSITIONS
from .scoring import add_fantasy_points_columns
from .io import write_parquet, read_parquet


def load_weekly_stats(
    seasons: List[int],
    cache: bool = True
) -> pd.DataFrame:
    """
    Load NFL weekly player statistics.

    Parameters
    ----------
    seasons : list of int
        List of seasons to load (e.g., [2019, 2020, 2021])
    cache : bool
        Whether to cache data to disk

    Returns
    -------
    pd.DataFrame
        Weekly player statistics with columns:
        - season, week, player_id, player_name, position, team
        - rushing/passing/receiving stats
        - fantasy points (ppr, half_ppr, standard)
    """
    cache_file = INTERIM_DATA_DIR / f"weekly_stats_{min(seasons)}_{max(seasons)}.parquet"

    if cache and cache_file.exists():
        print(f"Loading cached data from {cache_file}")
        return read_parquet(cache_file)

    if not NFL_DATA_AVAILABLE:
        raise ImportError(
            "nflreadpy is required to load data. "
            "Install it with: pip install nflreadpy"
        )

    print(f"Downloading weekly stats for seasons {seasons}...")
    # nflreadpy uses load_player_stats for weekly data
    # Returns Polars DataFrame - convert to Pandas
    df = nfl.load_player_stats(seasons).to_pandas()

    # Filter to relevant positions
    df = df[df['position'].isin(POSITIONS)].copy()

    # Clean column names first (nflreadpy uses lowercase already, but be consistent)
    df.columns = df.columns.str.lower().str.replace(' ', '_')

    # nflreadpy already includes fantasy_points and fantasy_points_ppr
    # Rename to match our convention (after lowercasing)
    df = df.rename(columns={
        'fantasy_points': 'fp_standard',
        'fantasy_points_ppr': 'fp_ppr',
        'opponent_team': 'opponent',
        'passing_interceptions': 'interceptions',
        'receiving_tds': 'receiving_td',
        'passing_tds': 'passing_td',
        'rushing_tds': 'rushing_td',
        'attempts': 'pass_att'
        # 'carries' stays as 'carries' - nflreadpy already uses this name
    })

    # Calculate half-PPR (average of standard and PPR)
    df['fp_half_ppr'] = (df['fp_standard'] + df['fp_ppr']) / 2

    # Sort by player and week
    df = df.sort_values(['player_id', 'season', 'week']).reset_index(drop=True)

    if cache:
        write_parquet(df, cache_file)
        print(f"Cached data to {cache_file}")

    return df


def load_pbp_data(
    seasons: List[int],
    cache: bool = True
) -> pd.DataFrame:
    """
    Load NFL play-by-play data for advanced features.

    Parameters
    ----------
    seasons : list of int
        List of seasons to load
    cache : bool
        Whether to cache data to disk

    Returns
    -------
    pd.DataFrame
        Play-by-play data
    """
    cache_file = INTERIM_DATA_DIR / f"pbp_{min(seasons)}_{max(seasons)}.parquet"

    if cache and cache_file.exists():
        print(f"Loading cached PBP data from {cache_file}")
        return read_parquet(cache_file)

    if not NFL_DATA_AVAILABLE:
        raise ImportError(
            "nflreadpy is required to load data. "
            "Install it with: pip install nflreadpy"
        )

    print(f"Downloading play-by-play data for seasons {seasons}...")
    # nflreadpy uses load_pbp for play-by-play data
    # Returns Polars DataFrame - convert to Pandas
    df = nfl.load_pbp(seasons).to_pandas()

    if cache:
        write_parquet(df, cache_file)
        print(f"Cached PBP data to {cache_file}")

    return df


def load_rosters(
    seasons: List[int],
    cache: bool = True
) -> pd.DataFrame:
    """
    Load NFL roster data.

    Parameters
    ----------
    seasons : list of int
        List of seasons to load
    cache : bool
        Whether to cache data to disk

    Returns
    -------
    pd.DataFrame
        Roster data with player info
    """
    cache_file = INTERIM_DATA_DIR / f"rosters_{min(seasons)}_{max(seasons)}.parquet"

    if cache and cache_file.exists():
        print(f"Loading cached roster data from {cache_file}")
        return read_parquet(cache_file)

    if not NFL_DATA_AVAILABLE:
        raise ImportError(
            "nflreadpy is required to load data. "
            "Install it with: pip install nflreadpy"
        )

    print(f"Downloading roster data for seasons {seasons}...")
    # nflreadpy uses load_rosters
    # Returns Polars DataFrame - convert to Pandas
    df = nfl.load_rosters(seasons).to_pandas()

    if cache:
        write_parquet(df, cache_file)
        print(f"Cached roster data to {cache_file}")

    return df


def load_schedules(
    seasons: List[int],
    cache: bool = True
) -> pd.DataFrame:
    """
    Load NFL schedule data for opponent matching and game context.

    Parameters
    ----------
    seasons : list of int
        List of seasons to load
    cache : bool
        Whether to cache data to disk

    Returns
    -------
    pd.DataFrame
        Schedule with game_id, teams, spreads, over/under, etc.
    """
    cache_file = INTERIM_DATA_DIR / f"schedules_{min(seasons)}_{max(seasons)}.parquet"

    if cache and cache_file.exists():
        print(f"Loading cached schedule data from {cache_file}")
        return read_parquet(cache_file)

    if not NFL_DATA_AVAILABLE:
        raise ImportError(
            "nflreadpy is required to load data. "
            "Install it with: pip install nflreadpy"
        )

    print(f"Downloading schedule data for seasons {seasons}...")
    # nflreadpy uses load_schedules
    # Returns Polars DataFrame - convert to Pandas
    df = nfl.load_schedules(seasons).to_pandas()

    if cache:
        write_parquet(df, cache_file)
        print(f"Cached schedule data to {cache_file}")

    return df


def merge_game_context(
    player_df: pd.DataFrame,
    schedule_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge game context (spread, over/under, weather) to player data.

    Parameters
    ----------
    player_df : pd.DataFrame
        Player weekly stats
    schedule_df : pd.DataFrame
        Schedule data

    Returns
    -------
    pd.DataFrame
        Player data with game context features
    """
    # Create home and away team views
    schedule_home = schedule_df.rename(columns={
        'home_team': 'team',
        'away_team': 'opponent',
        'spread_line': 'spread',
        'total_line': 'total'
    })[['game_id', 'season', 'week', 'team', 'opponent', 'spread', 'total',
        'roof', 'temp', 'wind']].copy()
    schedule_home['home_away'] = 'home'

    schedule_away = schedule_df.rename(columns={
        'away_team': 'team',
        'home_team': 'opponent',
    })[['game_id', 'season', 'week', 'team', 'opponent', 'roof',
        'temp', 'wind']].copy()
    schedule_away['home_away'] = 'away'
    # Flip spread for away team
    schedule_away['spread'] = -schedule_df['spread_line']
    schedule_away['total'] = schedule_df['total_line']

    # Combine home and away
    schedule_merged = pd.concat([schedule_home, schedule_away], ignore_index=True)

    # Merge to player data
    merged = player_df.merge(
        schedule_merged,
        on=['season', 'week', 'team'],
        how='left'
    )

    return merged


def compute_target_encoding(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    categorical_col: str,
    target_col: str,
    smoothing: float = 10.0
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Compute target encoding for a categorical variable.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training data
    val_df : pd.DataFrame
        Validation data
    test_df : pd.DataFrame
        Test data
    categorical_col : str
        Column to encode
    target_col : str
        Target column
    smoothing : float
        Smoothing parameter (higher = more regularization)

    Returns
    -------
    tuple
        (train_df, val_df, test_df) with encoded column added
    """
    # Compute mean target per category
    global_mean = train_df[target_col].mean()

    category_stats = train_df.groupby(categorical_col)[target_col].agg(['mean', 'count'])
    category_stats['smooth_mean'] = (
        (category_stats['mean'] * category_stats['count'] + global_mean * smoothing) /
        (category_stats['count'] + smoothing)
    )

    encoding_map = category_stats['smooth_mean'].to_dict()

    # Apply encoding
    encoded_col = f"{categorical_col}_encoded"
    train_df[encoded_col] = train_df[categorical_col].map(encoding_map).fillna(global_mean)
    val_df[encoded_col] = val_df[categorical_col].map(encoding_map).fillna(global_mean)
    test_df[encoded_col] = test_df[categorical_col].map(encoding_map).fillna(global_mean)

    return train_df, val_df, test_df


def remove_low_volume_players(
    df: pd.DataFrame,
    min_games: int = 4,
    min_snaps_per_game: float = 5.0
) -> pd.DataFrame:
    """
    Filter out low-volume players who are unlikely to be fantasy-relevant.

    Parameters
    ----------
    df : pd.DataFrame
        Player data
    min_games : int
        Minimum games played
    min_snaps_per_game : float
        Minimum average snaps per game

    Returns
    -------
    pd.DataFrame
        Filtered dataframe
    """
    # Count games per player
    player_games = df.groupby('player_id').size()
    valid_players = player_games[player_games >= min_games].index

    # Filter by snaps if available
    if 'snaps' in df.columns:
        player_avg_snaps = df.groupby('player_id')['snaps'].mean()
        valid_players_snaps = player_avg_snaps[
            player_avg_snaps >= min_snaps_per_game
        ].index
        valid_players = valid_players.intersection(valid_players_snaps)

    return df[df['player_id'].isin(valid_players)].copy()


def create_player_lookup(df: pd.DataFrame) -> dict:
    """
    Create player_id to player_name lookup dictionary.

    Parameters
    ----------
    df : pd.DataFrame
        Player data

    Returns
    -------
    dict
        Mapping of player_id -> player_name
    """
    return df.groupby('player_id')['player_name'].first().to_dict()
