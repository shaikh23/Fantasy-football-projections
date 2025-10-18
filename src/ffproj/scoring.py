"""
Fantasy football scoring functions for different formats.
"""
import pandas as pd
import numpy as np


def fantasy_points_ppr(
    passing_yds: float = 0,
    passing_td: float = 0,
    interceptions: float = 0,
    rushing_yds: float = 0,
    rushing_td: float = 0,
    receptions: float = 0,
    receiving_yds: float = 0,
    receiving_td: float = 0,
    fumbles_lost: float = 0,
    two_pt_conversions: float = 0
) -> float:
    """
    Calculate PPR (Point Per Reception) fantasy points.

    Scoring rules:
    - Passing: 1 pt per 25 yards, 4 pts per TD, -2 pts per INT
    - Rushing: 1 pt per 10 yards, 6 pts per TD
    - Receiving: 1 pt per reception, 1 pt per 10 yards, 6 pts per TD
    - Fumbles: -2 pts per lost fumble
    - 2-pt conversions: 2 pts

    Parameters
    ----------
    passing_yds : float
        Passing yards
    passing_td : float
        Passing touchdowns
    interceptions : float
        Interceptions thrown
    rushing_yds : float
        Rushing yards
    rushing_td : float
        Rushing touchdowns
    receptions : float
        Number of receptions
    receiving_yds : float
        Receiving yards
    receiving_td : float
        Receiving touchdowns
    fumbles_lost : float
        Fumbles lost
    two_pt_conversions : float
        Two-point conversions

    Returns
    -------
    float
        Total fantasy points in PPR format
    """
    return (
        (passing_yds / 25) +
        (4 * passing_td) -
        (2 * interceptions) +
        (rushing_yds / 10) +
        (6 * rushing_td) +
        (1.0 * receptions) +
        (receiving_yds / 10) +
        (6 * receiving_td) -
        (2 * fumbles_lost) +
        (2 * two_pt_conversions)
    )


def fantasy_points_half_ppr(
    passing_yds: float = 0,
    passing_td: float = 0,
    interceptions: float = 0,
    rushing_yds: float = 0,
    rushing_td: float = 0,
    receptions: float = 0,
    receiving_yds: float = 0,
    receiving_td: float = 0,
    fumbles_lost: float = 0,
    two_pt_conversions: float = 0
) -> float:
    """
    Calculate Half-PPR fantasy points.
    Half-PPR awards 0.5 points per reception instead of 1.0.

    Parameters are the same as fantasy_points_ppr.

    Returns
    -------
    float
        Total fantasy points in Half-PPR format
    """
    ppr_score = fantasy_points_ppr(
        passing_yds, passing_td, interceptions,
        rushing_yds, rushing_td, receptions,
        receiving_yds, receiving_td, fumbles_lost,
        two_pt_conversions
    )
    # Subtract 0.5 per reception to convert from PPR to half-PPR
    return ppr_score - (0.5 * receptions)


def fantasy_points_standard(
    passing_yds: float = 0,
    passing_td: float = 0,
    interceptions: float = 0,
    rushing_yds: float = 0,
    rushing_td: float = 0,
    receptions: float = 0,
    receiving_yds: float = 0,
    receiving_td: float = 0,
    fumbles_lost: float = 0,
    two_pt_conversions: float = 0
) -> float:
    """
    Calculate Standard (non-PPR) fantasy points.
    Standard scoring awards 0 points per reception.

    Parameters are the same as fantasy_points_ppr.

    Returns
    -------
    float
        Total fantasy points in Standard format
    """
    ppr_score = fantasy_points_ppr(
        passing_yds, passing_td, interceptions,
        rushing_yds, rushing_td, receptions,
        receiving_yds, receiving_td, fumbles_lost,
        two_pt_conversions
    )
    # Subtract 1.0 per reception to convert from PPR to standard
    return ppr_score - (1.0 * receptions)


def add_fantasy_points_columns(
    df: pd.DataFrame,
    scoring_systems: list = None
) -> pd.DataFrame:
    """
    Add fantasy points columns to a DataFrame with player stats.

    Expected columns in df:
    - passing_yards, passing_tds, interceptions
    - rushing_yards, rushing_tds
    - receptions, receiving_yards, receiving_tds
    - fumbles_lost, two_pt_conversions

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with player weekly stats
    scoring_systems : list, optional
        List of scoring systems to compute. Options: ['ppr', 'half_ppr', 'standard']
        If None, computes all three.

    Returns
    -------
    pd.DataFrame
        DataFrame with added fantasy points columns
    """
    if scoring_systems is None:
        scoring_systems = ['ppr', 'half_ppr', 'standard']

    df = df.copy()

    # Normalize column names
    col_map = {
        'passing_yards': 'passing_yds',
        'passing_tds': 'passing_td',
        'ints': 'interceptions',
        'rushing_yards': 'rushing_yds',
        'rushing_tds': 'rushing_td',
        'rec': 'receptions',
        'receiving_yards': 'receiving_yds',
        'receiving_tds': 'receiving_td',
        'fum_lost': 'fumbles_lost',
        'two_pt': 'two_pt_conversions'
    }

    # Fill missing columns with 0
    stat_cols = [
        'passing_yds', 'passing_td', 'interceptions',
        'rushing_yds', 'rushing_td', 'receptions',
        'receiving_yds', 'receiving_td', 'fumbles_lost',
        'two_pt_conversions'
    ]

    for old_name, new_name in col_map.items():
        if old_name in df.columns and new_name not in df.columns:
            df[new_name] = df[old_name]

    for col in stat_cols:
        if col not in df.columns:
            df[col] = 0.0

    # Fill NaN with 0
    df[stat_cols] = df[stat_cols].fillna(0.0)

    # Compute fantasy points
    if 'ppr' in scoring_systems:
        df['fp_ppr'] = df.apply(
            lambda row: fantasy_points_ppr(
                row['passing_yds'], row['passing_td'], row['interceptions'],
                row['rushing_yds'], row['rushing_td'], row['receptions'],
                row['receiving_yds'], row['receiving_td'], row['fumbles_lost'],
                row['two_pt_conversions']
            ),
            axis=1
        )

    if 'half_ppr' in scoring_systems:
        df['fp_half_ppr'] = df.apply(
            lambda row: fantasy_points_half_ppr(
                row['passing_yds'], row['passing_td'], row['interceptions'],
                row['rushing_yds'], row['rushing_td'], row['receptions'],
                row['receiving_yds'], row['receiving_td'], row['fumbles_lost'],
                row['two_pt_conversions']
            ),
            axis=1
        )

    if 'standard' in scoring_systems:
        df['fp_standard'] = df.apply(
            lambda row: fantasy_points_standard(
                row['passing_yds'], row['passing_td'], row['interceptions'],
                row['rushing_yds'], row['rushing_td'], row['receptions'],
                row['receiving_yds'], row['receiving_td'], row['fumbles_lost'],
                row['two_pt_conversions']
            ),
            axis=1
        )

    return df
