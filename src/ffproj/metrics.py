"""
Evaluation metrics for fantasy football projections.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error
from typing import Tuple, Optional
import matplotlib.pyplot as plt


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(mean_absolute_error(y_true, y_pred))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-10) -> float:
    """
    Mean Absolute Percentage Error.
    Adds epsilon to avoid division by zero.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100)


def coverage_at(
    y_true: np.ndarray,
    q_low: np.ndarray,
    q_high: np.ndarray
) -> float:
    """
    Calculate coverage: proportion of true values within [q_low, q_high].

    Parameters
    ----------
    y_true : array-like
        True values
    q_low : array-like
        Lower quantile predictions
    q_high : array-like
        Upper quantile predictions

    Returns
    -------
    float
        Coverage rate (should be ~0.8 for 10th-90th percentile intervals)
    """
    y_true = np.asarray(y_true)
    q_low = np.asarray(q_low)
    q_high = np.asarray(q_high)
    return float(np.mean((y_true >= q_low) & (y_true <= q_high)))


def interval_width(q_low: np.ndarray, q_high: np.ndarray) -> float:
    """Average width of prediction intervals."""
    return float(np.mean(q_high - q_low))


def spearman_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Spearman rank correlation.
    Useful for evaluating ranking quality.
    """
    if len(y_true) < 2:
        return np.nan
    corr, _ = spearmanr(y_true, y_pred)
    return float(corr)


def precision_at_k(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    k: int,
    threshold: Optional[float] = None
) -> float:
    """
    Precision@K for start/sit decisions.

    Parameters
    ----------
    y_true : array-like
        True fantasy points
    y_pred : array-like
        Predicted fantasy points
    k : int
        Number of top players to consider
    threshold : float, optional
        Points threshold for "startable". If None, uses top-k from y_true.

    Returns
    -------
    float
        Precision@K
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if threshold is None:
        # Top-K approach: players who actually scored in top-K
        top_k_true_idx = np.argsort(y_true)[-k:]
        relevant = set(top_k_true_idx)
    else:
        # Threshold approach: players who scored above threshold
        relevant = set(np.where(y_true >= threshold)[0])

    # Top-K predicted
    top_k_pred_idx = np.argsort(y_pred)[-k:]
    predicted = set(top_k_pred_idx)

    if len(predicted) == 0:
        return 0.0

    return float(len(relevant & predicted) / len(predicted))


def calibration_curve(
    y_true: np.ndarray,
    q_low: np.ndarray,
    q_high: np.ndarray,
    n_bins: int = 10
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute calibration curve for prediction intervals.

    Returns expected vs observed coverage across different prediction subsets.

    Parameters
    ----------
    y_true : array-like
        True values
    q_low : array-like
        Lower quantile predictions
    q_high : array-like
        Upper quantile predictions
    n_bins : int
        Number of bins for calibration

    Returns
    -------
    expected_coverage : np.ndarray
        Expected coverage (always 0.8 for 80% intervals)
    observed_coverage : np.ndarray
        Observed coverage in each bin
    """
    y_true = np.asarray(y_true)
    q_low = np.asarray(q_low)
    q_high = np.asarray(q_high)

    # Sort by interval width (uncertainty)
    widths = q_high - q_low
    sorted_idx = np.argsort(widths)

    bin_size = len(y_true) // n_bins
    observed = []

    for i in range(n_bins):
        start = i * bin_size
        end = start + bin_size if i < n_bins - 1 else len(y_true)
        bin_idx = sorted_idx[start:end]

        coverage = coverage_at(
            y_true[bin_idx],
            q_low[bin_idx],
            q_high[bin_idx]
        )
        observed.append(coverage)

    expected = np.full(n_bins, 0.8)  # For 80% intervals
    return expected, np.array(observed)


def plot_calibration_curve(
    y_true: np.ndarray,
    q_low: np.ndarray,
    q_high: np.ndarray,
    n_bins: int = 10,
    title: str = "Calibration Curve"
) -> plt.Figure:
    """
    Plot calibration curve.

    Parameters
    ----------
    y_true : array-like
        True values
    q_low : array-like
        Lower quantile predictions
    q_high : array-like
        Upper quantile predictions
    n_bins : int
        Number of bins
    title : str
        Plot title

    Returns
    -------
    matplotlib.figure.Figure
        Calibration plot
    """
    expected, observed = calibration_curve(y_true, q_low, q_high, n_bins)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    ax.plot(expected, observed, 'o-', label='Model')
    ax.set_xlabel('Expected Coverage')
    ax.set_ylabel('Observed Coverage')
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)

    return fig


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    q_low: Optional[np.ndarray] = None,
    q_high: Optional[np.ndarray] = None,
    position: Optional[str] = None
) -> dict:
    """
    Compute all evaluation metrics.

    Parameters
    ----------
    y_true : array-like
        True fantasy points
    y_pred : array-like
        Predicted fantasy points (median)
    q_low : array-like, optional
        Lower quantile (e.g., 10th percentile)
    q_high : array-like, optional
        Upper quantile (e.g., 90th percentile)
    position : str, optional
        Position label for reporting

    Returns
    -------
    dict
        Dictionary of metric name -> value
    """
    metrics = {
        'MAE': mae(y_true, y_pred),
        'RMSE': rmse(y_true, y_pred),
        'MAPE': mape(y_true, y_pred),
        'Spearman': spearman_correlation(y_true, y_pred),
        'Precision@10': precision_at_k(y_true, y_pred, k=10),
        'Precision@20': precision_at_k(y_true, y_pred, k=20),
    }

    if q_low is not None and q_high is not None:
        metrics['Coverage_80'] = coverage_at(y_true, q_low, q_high)
        metrics['Interval_Width'] = interval_width(q_low, q_high)

    if position:
        metrics = {f"{position}_{k}": v for k, v in metrics.items()}

    return metrics


def print_metrics_report(metrics: dict, title: str = "Evaluation Metrics"):
    """Pretty print metrics report."""
    print(f"\n{'='*60}")
    print(f"{title:^60}")
    print(f"{'='*60}")

    for metric_name, value in sorted(metrics.items()):
        if isinstance(value, (int, float)):
            print(f"{metric_name:<40} {value:>10.4f}")
        else:
            print(f"{metric_name:<40} {value:>10}")

    print(f"{'='*60}\n")


def compare_models(
    y_true: np.ndarray,
    predictions_dict: dict,
    metric_fn=mae,
    metric_name: str = "MAE"
) -> pd.DataFrame:
    """
    Compare multiple models on a single metric.

    Parameters
    ----------
    y_true : array-like
        True values
    predictions_dict : dict
        Dictionary mapping model name -> predictions
    metric_fn : callable
        Metric function (default: mae)
    metric_name : str
        Name of the metric for display

    Returns
    -------
    pd.DataFrame
        Comparison table sorted by metric value
    """
    results = []
    for model_name, y_pred in predictions_dict.items():
        score = metric_fn(y_true, y_pred)
        results.append({'Model': model_name, metric_name: score})

    df = pd.DataFrame(results).sort_values(metric_name)
    return df
