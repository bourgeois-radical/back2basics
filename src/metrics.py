"""
Metrics module for MLE and Loss Functions presentation.

This module provides functions to calculate and compare metrics
for regression and classification tasks.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    explained_variance_score,
    f1_score,
    max_error,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


def calculate_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:
    """
    Calculate comprehensive regression metrics.

    Parameters
    ----------
    y_true : np.ndarray
        True values
    y_pred : np.ndarray
        Predicted values

    Returns
    -------
    metrics : dict
        Dictionary containing:
        - mse: Mean Squared Error
        - rmse: Root MSE
        - mae: Mean Absolute Error
        - r2: R^2 score
        - median_ae: Median Absolute Error
        - max_error: Maximum error
        - explained_variance: Explained variance score
    """
    mse = mean_squared_error(y_true, y_pred)

    return {
        "mse": mse,
        "rmse": np.sqrt(mse),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
        "median_ae": median_absolute_error(y_true, y_pred),
        "max_error": max_error(y_true, y_pred),
        "explained_variance": explained_variance_score(y_true, y_pred),
    }


def calculate_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: np.ndarray | None = None,
) -> dict:
    """
    Calculate comprehensive classification metrics.

    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    y_pred_proba : np.ndarray, optional
        Predicted probabilities

    Returns
    -------
    metrics : dict
        Dictionary containing:
        - accuracy: Overall accuracy
        - precision: Precision for positive class
        - recall: Recall for positive class
        - f1: F1-score
        - auc_roc: AUC-ROC (if proba provided)
        - auc_pr: AUC-PR (if proba provided)
        - confusion_matrix: 2x2 array
        - classification_report: sklearn report string
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
        "classification_report": classification_report(y_true, y_pred),
    }

    if y_pred_proba is not None:
        try:
            metrics["auc_roc"] = roc_auc_score(y_true, y_pred_proba)
        except ValueError:
            metrics["auc_roc"] = float("nan")

        # AUC-PR
        precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_pred_proba)
        metrics["auc_pr"] = auc(recall_curve, precision_curve)

    return metrics


def calculate_metrics_at_thresholds(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> pd.DataFrame:
    """
    Calculate metrics across different classification thresholds.

    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_pred_proba : np.ndarray
        Predicted probabilities
    thresholds : np.ndarray, optional
        Array of thresholds to try (default: 0.0 to 1.0 in 0.05 steps)

    Returns
    -------
    df : pd.DataFrame
        DataFrame with columns: threshold, precision, recall, f1, accuracy
        One row per threshold

    Notes
    -----
    Useful for finding optimal threshold for business metric.
    """
    if thresholds is None:
        thresholds = np.arange(0.0, 1.05, 0.05)

    results = []
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        results.append(
            {
                "threshold": threshold,
                "precision": precision_score(y_true, y_pred, zero_division=0),
                "recall": recall_score(y_true, y_pred, zero_division=0),
                "f1": f1_score(y_true, y_pred, zero_division=0),
                "accuracy": accuracy_score(y_true, y_pred),
            }
        )

    return pd.DataFrame(results)


def compare_model_metrics(
    models_results: dict,
    metric_type: str = "regression",
) -> pd.DataFrame:
    """
    Create comparison table of metrics across models.

    Parameters
    ----------
    models_results : dict
        Dictionary with structure:
        {'model_name': {'test_metrics': {...}}} or
        {'model_name': {'metrics': {...}}}
    metric_type : str
        'regression' or 'classification'

    Returns
    -------
    df : pd.DataFrame
        Index: model names (MSE, MAE, Huber, etc.)
        Columns: metrics
        Values: metric values

    Notes
    -----
    Makes side-by-side comparison easy.
    Can highlight best values per column.
    """
    if metric_type == "regression":
        metric_keys = ["mse", "rmse", "mae", "r2", "median_ae"]
    else:
        metric_keys = ["accuracy", "precision", "recall", "f1", "auc_roc"]

    data = {}
    for model_name, results in models_results.items():
        metrics = results.get("test_metrics", results.get("metrics", {}))
        data[model_name] = {k: metrics.get(k, float("nan")) for k in metric_keys}

    df = pd.DataFrame(data).T
    return df.round(4)


def analyze_residual_distribution(
    residuals: np.ndarray,
    assumed_distribution: str = "gaussian",
) -> dict:
    """
    Statistical tests for residual distribution.

    Parameters
    ----------
    residuals : np.ndarray
        Model residuals
    assumed_distribution : str
        'gaussian' or 'laplace'

    Returns
    -------
    analysis : dict
        Dictionary containing:
        - mean: Residual mean (should be ~0)
        - std: Standard deviation
        - skewness: Skewness (should be ~0 for symmetric)
        - kurtosis: Kurtosis (excess, should be ~0 for Gaussian)
        - shapiro_test: Shapiro-Wilk test for normality (stat, p-value)
        - ks_test: Kolmogorov-Smirnov test (stat, p-value)
        - jarque_bera: Jarque-Bera test (stat, p-value)
        - q25: 25th percentile
        - q50: Median
        - q75: 75th percentile

    Notes
    -----
    Quantifies how well residuals match assumed distribution.
    Low p-values indicate mismatch.
    """
    residuals = np.asarray(residuals).flatten()

    analysis = {
        "mean": np.mean(residuals),
        "std": np.std(residuals),
        "skewness": stats.skew(residuals),
        "kurtosis": stats.kurtosis(residuals),  # excess kurtosis
        "q25": np.percentile(residuals, 25),
        "q50": np.percentile(residuals, 50),
        "q75": np.percentile(residuals, 75),
    }

    # Shapiro-Wilk test for normality (use subset if too many samples)
    sample_for_test = residuals if len(residuals) <= 5000 else residuals[:5000]
    try:
        shapiro_stat, shapiro_p = stats.shapiro(sample_for_test)
        analysis["shapiro_test"] = {"statistic": shapiro_stat, "p_value": shapiro_p}
    except Exception:
        analysis["shapiro_test"] = {"statistic": float("nan"), "p_value": float("nan")}

    # Kolmogorov-Smirnov test
    if assumed_distribution == "gaussian":
        standardized = (residuals - np.mean(residuals)) / np.std(residuals)
        ks_stat, ks_p = stats.kstest(standardized, "norm")
    else:  # laplace
        scale = np.std(residuals) / np.sqrt(2)  # MLE scale for Laplace
        ks_stat, ks_p = stats.kstest(
            residuals, "laplace", args=(np.median(residuals), scale)
        )
    analysis["ks_test"] = {"statistic": ks_stat, "p_value": ks_p}

    # Jarque-Bera test (tests for normality based on skewness and kurtosis)
    try:
        jb_stat, jb_p = stats.jarque_bera(residuals)
        analysis["jarque_bera"] = {"statistic": jb_stat, "p_value": jb_p}
    except Exception:
        analysis["jarque_bera"] = {"statistic": float("nan"), "p_value": float("nan")}

    return analysis


def print_residual_analysis(analysis: dict) -> None:
    """
    Print formatted residual analysis results.

    Parameters
    ----------
    analysis : dict
        Output from analyze_residual_distribution()
    """
    print("\n--- Residual Distribution Analysis ---")
    print(f"Mean:     {analysis['mean']:.4f} (should be ~0)")
    print(f"Std Dev:  {analysis['std']:.4f}")
    print(f"Skewness: {analysis['skewness']:.4f} (should be ~0 for symmetric)")
    print(f"Kurtosis: {analysis['kurtosis']:.4f} (excess, ~0 for Gaussian, ~3 for Laplace)")
    print(f"\nQuantiles: Q25={analysis['q25']:.4f}, Median={analysis['q50']:.4f}, Q75={analysis['q75']:.4f}")

    print("\n--- Statistical Tests ---")
    if "shapiro_test" in analysis:
        sw = analysis["shapiro_test"]
        print(f"Shapiro-Wilk:  stat={sw['statistic']:.4f}, p={sw['p_value']:.4f}")
    if "ks_test" in analysis:
        ks = analysis["ks_test"]
        print(f"K-S Test:      stat={ks['statistic']:.4f}, p={ks['p_value']:.4f}")
    if "jarque_bera" in analysis:
        jb = analysis["jarque_bera"]
        print(f"Jarque-Bera:   stat={jb['statistic']:.4f}, p={jb['p_value']:.4f}")

    print("\n(Low p-values indicate residuals don't match assumed distribution)")


def calculate_loss_values(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    loss_type: str = "mse",
) -> np.ndarray:
    """
    Calculate per-sample loss values.

    Parameters
    ----------
    y_true : np.ndarray
        True values
    y_pred : np.ndarray
        Predicted values
    loss_type : str
        'mse', 'mae', or 'huber'

    Returns
    -------
    losses : np.ndarray
        Per-sample loss values
    """
    residuals = y_true - y_pred

    if loss_type == "mse":
        return residuals**2
    elif loss_type == "mae":
        return np.abs(residuals)
    elif loss_type == "huber":
        delta = 1.0
        abs_residuals = np.abs(residuals)
        quadratic = 0.5 * residuals**2
        linear = delta * (abs_residuals - 0.5 * delta)
        return np.where(abs_residuals <= delta, quadratic, linear)
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")
