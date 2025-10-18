"""
I/O utilities for reading and writing data files.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, Optional
import pickle
import json


def read_parquet(
    file_path: Union[str, Path],
    columns: Optional[list] = None
) -> pd.DataFrame:
    """
    Read parquet file with error handling.

    Parameters
    ----------
    file_path : str or Path
        Path to parquet file
    columns : list, optional
        Specific columns to read

    Returns
    -------
    pd.DataFrame
        Loaded dataframe
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    return pd.read_parquet(file_path, columns=columns)


def write_parquet(
    df: pd.DataFrame,
    file_path: Union[str, Path],
    compression: str = "snappy"
) -> None:
    """
    Write dataframe to parquet file.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to save
    file_path : str or Path
        Output file path
    compression : str
        Compression algorithm (default: snappy)
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(file_path, compression=compression, index=False)


def read_csv(
    file_path: Union[str, Path],
    **kwargs
) -> pd.DataFrame:
    """
    Read CSV file with common defaults.

    Parameters
    ----------
    file_path : str or Path
        Path to CSV file
    **kwargs
        Additional arguments to pd.read_csv

    Returns
    -------
    pd.DataFrame
        Loaded dataframe
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    return pd.read_csv(file_path, **kwargs)


def write_csv(
    df: pd.DataFrame,
    file_path: Union[str, Path],
    index: bool = False,
    **kwargs
) -> None:
    """
    Write dataframe to CSV file.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to save
    file_path : str or Path
        Output file path
    index : bool
        Whether to write index (default: False)
    **kwargs
        Additional arguments to df.to_csv
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(file_path, index=index, **kwargs)


def save_model(
    model: object,
    file_path: Union[str, Path]
) -> None:
    """
    Save model to pickle file.

    Parameters
    ----------
    model : object
        Model object to save
    file_path : str or Path
        Output file path
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, 'wb') as f:
        pickle.dump(model, f)


def load_model(file_path: Union[str, Path]) -> object:
    """
    Load model from pickle file.

    Parameters
    ----------
    file_path : str or Path
        Path to pickled model

    Returns
    -------
    object
        Loaded model
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Model file not found: {file_path}")

    with open(file_path, 'rb') as f:
        return pickle.load(f)


def save_json(
    data: dict,
    file_path: Union[str, Path],
    indent: int = 2
) -> None:
    """
    Save dictionary to JSON file.

    Parameters
    ----------
    data : dict
        Dictionary to save
    file_path : str or Path
        Output file path
    indent : int
        JSON indentation (default: 2)
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert numpy types to native Python types
    def convert(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    data_clean = {k: convert(v) for k, v in data.items()}

    with open(file_path, 'w') as f:
        json.dump(data_clean, f, indent=indent)


def load_json(file_path: Union[str, Path]) -> dict:
    """
    Load dictionary from JSON file.

    Parameters
    ----------
    file_path : str or Path
        Path to JSON file

    Returns
    -------
    dict
        Loaded dictionary
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"JSON file not found: {file_path}")

    with open(file_path, 'r') as f:
        return json.load(f)


def ensure_directory(dir_path: Union[str, Path]) -> Path:
    """
    Create directory if it doesn't exist.

    Parameters
    ----------
    dir_path : str or Path
        Directory path

    Returns
    -------
    Path
        Path object to directory
    """
    dir_path = Path(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path
