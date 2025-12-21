"""
Models module for MLE and Loss Functions presentation.

This module provides functions to train models with different loss functions,
demonstrating how the choice of loss affects model behavior.
"""

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.linear_model import (
    HuberRegressor,
    LinearRegression,
    LogisticRegression,
    QuantileRegressor,
)

from metrics import calculate_classification_metrics, calculate_regression_metrics


def train_linear_regression_mse(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> BaseEstimator:
    """
    Train linear regression with MSE (standard sklearn).

    Parameters
    ----------
    X_train, y_train : np.ndarray
        Training data

    Returns
    -------
    model : Fitted LinearRegression

    Notes
    -----
    This is standard OLS: minimizes sum of squared residuals.
    Equivalent to MLE under Gaussian noise assumption.
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def train_linear_regression_mae(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> BaseEstimator:
    """
    Train linear regression with MAE (L1 loss).

    Parameters
    ----------
    X_train, y_train : np.ndarray
        Training data

    Returns
    -------
    model : Fitted model (QuantileRegressor)

    Notes
    -----
    sklearn's QuantileRegressor with quantile=0.5 minimizes MAE.
    Equivalent to MLE under Laplace noise assumption.
    """
    model = QuantileRegressor(quantile=0.5, alpha=0.0, solver="highs")
    model.fit(X_train, y_train)
    return model


def train_huber_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    epsilon: float = 1.35,
) -> HuberRegressor:
    """
    Train regression with Huber loss (robust to outliers).

    Parameters
    ----------
    X_train, y_train : np.ndarray
        Training data
    epsilon : float
        Threshold where loss transitions from quadratic to linear

    Returns
    -------
    model : Fitted HuberRegressor

    Notes
    -----
    Huber loss:
    - L = 0.5 * error^2 if |error| <= epsilon
    - L = epsilon * (|error| - 0.5*epsilon) if |error| > epsilon

    Combines MSE (small errors) and MAE (large errors).
    Default epsilon=1.35 is commonly used in robust statistics.
    """
    model = HuberRegressor(epsilon=epsilon, max_iter=1000)
    model.fit(X_train, y_train)
    return model


def train_models_with_different_losses(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_type: str = "linear",
) -> dict:
    """
    Train multiple models with different loss functions.

    Parameters
    ----------
    X_train, y_train : np.ndarray
        Training data
    X_test, y_test : np.ndarray
        Test data
    model_type : str
        'linear' (default), 'xgboost' (if available)

    Returns
    -------
    results : dict
        Dictionary with keys 'MSE', 'MAE', 'Huber'
        Each value is a dict containing:
        - 'model': trained model object
        - 'train_preds': predictions on train set
        - 'test_preds': predictions on test set
        - 'train_metrics': dict of metrics
        - 'test_metrics': dict of metrics
        - 'residuals_train': train residuals
        - 'residuals_test': test residuals
    """
    results = {}

    # MSE (LinearRegression)
    model_mse = train_linear_regression_mse(X_train, y_train)
    train_preds_mse = model_mse.predict(X_train)
    test_preds_mse = model_mse.predict(X_test)

    results["MSE"] = {
        "model": model_mse,
        "train_preds": train_preds_mse,
        "test_preds": test_preds_mse,
        "train_metrics": calculate_regression_metrics(y_train, train_preds_mse),
        "test_metrics": calculate_regression_metrics(y_test, test_preds_mse),
        "residuals_train": y_train - train_preds_mse,
        "residuals_test": y_test - test_preds_mse,
    }

    # MAE (QuantileRegressor)
    model_mae = train_linear_regression_mae(X_train, y_train)
    train_preds_mae = model_mae.predict(X_train)
    test_preds_mae = model_mae.predict(X_test)

    results["MAE"] = {
        "model": model_mae,
        "train_preds": train_preds_mae,
        "test_preds": test_preds_mae,
        "train_metrics": calculate_regression_metrics(y_train, train_preds_mae),
        "test_metrics": calculate_regression_metrics(y_test, test_preds_mae),
        "residuals_train": y_train - train_preds_mae,
        "residuals_test": y_test - test_preds_mae,
    }

    # Huber
    model_huber = train_huber_regression(X_train, y_train)
    train_preds_huber = model_huber.predict(X_train)
    test_preds_huber = model_huber.predict(X_test)

    results["Huber"] = {
        "model": model_huber,
        "train_preds": train_preds_huber,
        "test_preds": test_preds_huber,
        "train_metrics": calculate_regression_metrics(y_train, train_preds_huber),
        "test_metrics": calculate_regression_metrics(y_test, test_preds_huber),
        "residuals_train": y_train - train_preds_huber,
        "residuals_test": y_test - test_preds_huber,
    }

    # Try XGBoost if requested and available
    if model_type == "xgboost":
        try:
            xgb_results = train_xgboost_models(X_train, y_train, X_test, y_test)
            results.update(xgb_results)
        except ImportError:
            print("XGBoost not available, skipping XGBoost models")

    return results


def train_xgboost_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """
    Train XGBoost models with different objectives.

    Parameters
    ----------
    X_train, y_train : np.ndarray
        Training data
    X_test, y_test : np.ndarray
        Test data

    Returns
    -------
    models : dict
        Keys: 'XGB-MSE', 'XGB-MAE'
        Values: trained XGBoost models and results

    Notes
    -----
    XGBoost supports many objectives:
    - reg:squarederror (MSE)
    - reg:absoluteerror (MAE)
    - reg:quantileerror (quantile regression)
    """
    try:
        import xgboost as xgb
    except ImportError:
        raise ImportError("XGBoost is required for this function")

    results = {}

    # XGBoost with squared error (MSE)
    model_xgb_mse = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        random_state=42,
    )
    model_xgb_mse.fit(X_train, y_train)
    train_preds = model_xgb_mse.predict(X_train)
    test_preds = model_xgb_mse.predict(X_test)

    results["XGB-MSE"] = {
        "model": model_xgb_mse,
        "train_preds": train_preds,
        "test_preds": test_preds,
        "train_metrics": calculate_regression_metrics(y_train, train_preds),
        "test_metrics": calculate_regression_metrics(y_test, test_preds),
        "residuals_train": y_train - train_preds,
        "residuals_test": y_test - test_preds,
    }

    # XGBoost with absolute error (MAE)
    model_xgb_mae = xgb.XGBRegressor(
        objective="reg:absoluteerror",
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        random_state=42,
    )
    model_xgb_mae.fit(X_train, y_train)
    train_preds = model_xgb_mae.predict(X_train)
    test_preds = model_xgb_mae.predict(X_test)

    results["XGB-MAE"] = {
        "model": model_xgb_mae,
        "train_preds": train_preds,
        "test_preds": test_preds,
        "train_metrics": calculate_regression_metrics(y_train, train_preds),
        "test_metrics": calculate_regression_metrics(y_test, test_preds),
        "residuals_train": y_train - train_preds,
        "residuals_test": y_test - test_preds,
    }

    return results


def train_classification_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_weight: str | None = None,
) -> dict:
    """
    Train classification models with/without class weighting.

    Parameters
    ----------
    X_train, y_train : np.ndarray
        Training data
    X_test, y_test : np.ndarray
        Test data
    class_weight : str, optional
        'balanced' to weight classes inversely proportional to frequency

    Returns
    -------
    results : dict
        Keys: 'unweighted', 'weighted'
        Each contains:
        - model
        - predictions
        - predicted probabilities
        - metrics (accuracy, precision, recall, f1, auc)
        - confusion matrix

    Notes
    -----
    Shows how loss function weighting affects:
    - Class balance in predictions
    - Recall vs precision tradeoff
    - Metric values
    """
    results = {}

    # Unweighted model
    model_unweighted = LogisticRegression(max_iter=1000, random_state=42)
    model_unweighted.fit(X_train, y_train)

    train_preds = model_unweighted.predict(X_train)
    test_preds = model_unweighted.predict(X_test)
    train_proba = model_unweighted.predict_proba(X_train)[:, 1]
    test_proba = model_unweighted.predict_proba(X_test)[:, 1]

    results["unweighted"] = {
        "model": model_unweighted,
        "train_preds": train_preds,
        "test_preds": test_preds,
        "train_proba": train_proba,
        "test_proba": test_proba,
        "train_metrics": calculate_classification_metrics(y_train, train_preds, train_proba),
        "test_metrics": calculate_classification_metrics(y_test, test_preds, test_proba),
    }

    # Weighted model
    model_weighted = LogisticRegression(
        class_weight="balanced", max_iter=1000, random_state=42
    )
    model_weighted.fit(X_train, y_train)

    train_preds = model_weighted.predict(X_train)
    test_preds = model_weighted.predict(X_test)
    train_proba = model_weighted.predict_proba(X_train)[:, 1]
    test_proba = model_weighted.predict_proba(X_test)[:, 1]

    results["weighted"] = {
        "model": model_weighted,
        "train_preds": train_preds,
        "test_preds": test_preds,
        "train_proba": train_proba,
        "test_proba": test_proba,
        "train_metrics": calculate_classification_metrics(y_train, train_preds, train_proba),
        "test_metrics": calculate_classification_metrics(y_test, test_preds, test_proba),
    }

    return results


def train_polynomial_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    degree: int = 2,
) -> tuple[Any, Any]:
    """
    Train polynomial regression model.

    Parameters
    ----------
    X_train, y_train : np.ndarray
        Training data
    degree : int
        Polynomial degree

    Returns
    -------
    model : LinearRegression
        Fitted model
    poly_features : PolynomialFeatures
        Feature transformer

    Notes
    -----
    Useful for demonstrating model misspecification:
    - Linear model on quadratic data shows bias
    - Polynomial model fits correctly
    """
    from sklearn.preprocessing import PolynomialFeatures

    poly_features = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly_features.fit_transform(X_train)

    model = LinearRegression()
    model.fit(X_poly, y_train)

    return model, poly_features


def predict_with_polynomial(
    model: Any,
    poly_features: Any,
    X: np.ndarray,
) -> np.ndarray:
    """
    Make predictions with polynomial model.

    Parameters
    ----------
    model : LinearRegression
        Trained polynomial model
    poly_features : PolynomialFeatures
        Feature transformer
    X : np.ndarray
        Input features

    Returns
    -------
    predictions : np.ndarray
    """
    X_poly = poly_features.transform(X)
    return model.predict(X_poly)


def get_model_coefficients(model: BaseEstimator, feature_names: list[str] | None = None) -> dict:
    """
    Extract coefficients from a linear model.

    Parameters
    ----------
    model : BaseEstimator
        A fitted linear model
    feature_names : list[str], optional
        Names of features

    Returns
    -------
    coefficients : dict
        Dictionary mapping feature names to coefficients
    """
    if hasattr(model, "coef_"):
        coefs = model.coef_
    else:
        raise ValueError("Model does not have coefficients (coef_ attribute)")

    if feature_names is None:
        feature_names = [f"X{i}" for i in range(len(coefs))]

    result = {"intercept": model.intercept_ if hasattr(model, "intercept_") else 0}
    for name, coef in zip(feature_names, coefs):
        result[name] = coef

    return result
