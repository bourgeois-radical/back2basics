    """
Data generation module for MLE and Loss Functions presentation.

This module provides functions to generate synthetic datasets with known
properties, allowing us to demonstrate the relationship between noise
distributions and loss functions.
"""

import numpy as np
from sklearn.datasets import (
    fetch_california_housing,
    load_breast_cancer,
    load_diabetes,
    load_wine,
)


def generate_regression_data_with_gaussian_noise(
    n_samples: int = 1000,
    n_features: int = 5,
    noise_std: float = 1.0,
    true_coefficients: np.ndarray | None = None,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic regression data with Gaussian noise.

    Creates data following: y = X @ true_coefficients + epsilon, where epsilon ~ N(0, sigma^2)

    Parameters
    ----------
    n_samples : int
        Number of samples to generate
    n_features : int
        Number of features in X
    noise_std : float
        Standard deviation of Gaussian noise
    true_coefficients : np.ndarray, optional
        True coefficients for linear relationship. If None, generated randomly.
    random_state : int
        Random seed for reproducibility

    Returns
    -------
    X : np.ndarray of shape (n_samples, n_features)
        Feature matrix
    y : np.ndarray of shape (n_samples,)
        Target values (signal + noise)
    true_signal : np.ndarray of shape (n_samples,)
        True signal without noise (X @ true_coefficients)

    Notes
    -----
    The true signal is returned so we can later visualize:
    - True noise: y - true_signal
    - Model residuals: y - predictions
    - Model bias: true_signal - predictions
    """
    rng = np.random.default_rng(random_state)

    # Generate feature matrix from standard normal
    X = rng.standard_normal((n_samples, n_features))

    # Generate or use provided coefficients
    if true_coefficients is None:
        true_coefficients = rng.standard_normal(n_features)
    else:
        true_coefficients = np.asarray(true_coefficients)
        if len(true_coefficients) != n_features:
            raise ValueError(
                f"true_coefficients length {len(true_coefficients)} "
                f"!= n_features {n_features}"
            )

    # Compute true signal (without noise)
    true_signal = X @ true_coefficients

    # Add Gaussian noise
    noise = rng.normal(0, noise_std, n_samples)
    y = true_signal + noise

    return X, y, true_signal


def generate_regression_data_with_laplace_noise(
    n_samples: int = 1000,
    n_features: int = 5,
    noise_scale: float = 1.0,
    true_coefficients: np.ndarray | None = None,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic regression data with Laplace (heavy-tailed) noise.

    Creates data following: y = X @ true_coefficients + epsilon, where epsilon ~ Laplace(0, b)

    Parameters
    ----------
    n_samples : int
        Number of samples to generate
    n_features : int
        Number of features in X
    noise_scale : float
        Scale parameter of Laplace distribution (controls spread)
    true_coefficients : np.ndarray, optional
        True coefficients for linear relationship
    random_state : int
        Random seed for reproducibility

    Returns
    -------
    X : np.ndarray of shape (n_samples, n_features)
        Feature matrix
    y : np.ndarray of shape (n_samples,)
        Target values (signal + noise)
    true_signal : np.ndarray of shape (n_samples,)
        True signal without noise

    Notes
    -----
    Laplace distribution has heavier tails than Gaussian, creating more outliers.
    This is ideal for demonstrating MAE's robustness vs MSE's sensitivity.

    The Laplace PDF is: p(x) = (1/2b) * exp(-|x|/b)
    where b is the scale parameter.
    """
    rng = np.random.default_rng(random_state)

    # Generate feature matrix
    X = rng.standard_normal((n_samples, n_features))

    # Generate or use provided coefficients
    if true_coefficients is None:
        true_coefficients = rng.standard_normal(n_features)
    else:
        true_coefficients = np.asarray(true_coefficients)

    # Compute true signal
    true_signal = X @ true_coefficients

    # Add Laplace noise
    noise = rng.laplace(0, noise_scale, n_samples)
    y = true_signal + noise

    return X, y, true_signal


def generate_regression_data_with_outliers(
    n_samples: int = 1000,
    n_features: int = 5,
    noise_std: float = 1.0,
    outlier_fraction: float = 0.1,
    outlier_magnitude: float = 10.0,
    true_coefficients: np.ndarray | None = None,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate regression data with mixture of Gaussian noise and extreme outliers.

    Most samples have Gaussian noise, but a fraction have extreme values.

    Parameters
    ----------
    n_samples : int
        Number of samples
    n_features : int
        Number of features
    noise_std : float
        Standard deviation for normal noise
    outlier_fraction : float
        Fraction of samples that are outliers (0 to 1)
    outlier_magnitude : float
        How far outliers deviate from true signal (in units of noise_std)
    true_coefficients : np.ndarray, optional
        True coefficients
    random_state : int
        Random seed

    Returns
    -------
    X : np.ndarray
        Features
    y : np.ndarray
        Targets with outliers
    true_signal : np.ndarray
        Signal without any noise
    is_outlier : np.ndarray of bool
        Boolean mask indicating which samples are outliers

    Notes
    -----
    Perfect for showing MSE vs MAE behavior:
    - MSE gets pulled toward outliers
    - MAE ignores them and fits majority well
    """
    rng = np.random.default_rng(random_state)

    # Generate base data with Gaussian noise
    X, y, true_signal = generate_regression_data_with_gaussian_noise(
        n_samples=n_samples,
        n_features=n_features,
        noise_std=noise_std,
        true_coefficients=true_coefficients,
        random_state=random_state,
    )

    # Select outlier indices
    n_outliers = int(n_samples * outlier_fraction)
    outlier_indices = rng.choice(n_samples, size=n_outliers, replace=False)

    # Create outlier mask
    is_outlier = np.zeros(n_samples, dtype=bool)
    is_outlier[outlier_indices] = True

    # Add extreme noise to outliers (positive or negative randomly)
    outlier_noise = (
        rng.choice([-1, 1], size=n_outliers) * outlier_magnitude * noise_std
    )
    y[outlier_indices] = true_signal[outlier_indices] + outlier_noise

    return X, y, true_signal, is_outlier


def generate_nonlinear_data(
    n_samples: int = 1000,
    noise_std: float = 0.5,
    function_type: str = "quadratic",
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate nonlinear regression data (for demonstrating model misspecification).

    Parameters
    ----------
    n_samples : int
        Number of samples
    noise_std : float
        Standard deviation of Gaussian noise
    function_type : str
        Type of nonlinearity: 'quadratic', 'cubic', 'sine', or 'exponential'
    random_state : int
        Random seed

    Returns
    -------
    X : np.ndarray of shape (n_samples, 1)
        Feature (1D for easy visualization)
    y : np.ndarray of shape (n_samples,)
        Target with noise
    true_signal : np.ndarray of shape (n_samples,)
        True nonlinear function values

    Notes
    -----
    Use this to show:
    - Linear model on nonlinear data -> systematic residual patterns (BIAS)
    - Polynomial model on nonlinear data -> good fit, random residuals
    """
    rng = np.random.default_rng(random_state)

    # Generate X uniformly in a suitable range
    X = rng.uniform(-3, 3, (n_samples, 1))
    x_flat = X.flatten()

    # Define true functions
    functions = {
        "quadratic": lambda x: 0.5 * x**2,
        "cubic": lambda x: 0.2 * x**3 - x,
        "sine": lambda x: 2 * np.sin(x),
        "exponential": lambda x: np.exp(0.5 * x) - 1,
    }

    if function_type not in functions:
        available = ", ".join(functions.keys())
        raise ValueError(f"Unknown function_type: {function_type}. Available: {available}")

    # Compute true signal
    true_signal = functions[function_type](x_flat)

    # Add Gaussian noise
    noise = rng.normal(0, noise_std, n_samples)
    y = true_signal + noise

    return X, y, true_signal


def generate_imbalanced_classification_data(
    n_samples: int = 1000,
    n_features: int = 10,
    imbalance_ratio: float = 0.05,
    class_separation: float = 1.5,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate binary classification data with class imbalance.

    Parameters
    ----------
    n_samples : int
        Total number of samples
    n_features : int
        Number of features
    imbalance_ratio : float
        Fraction of samples in minority (positive) class
    class_separation : float
        How separated the classes are (higher = easier to classify)
    random_state : int
        Random seed

    Returns
    -------
    X : np.ndarray of shape (n_samples, n_features)
        Features
    y : np.ndarray of shape (n_samples,)
        Binary labels (0 or 1)

    Notes
    -----
    Creates realistic imbalanced scenario:
    - Minority class is harder to learn
    - Demonstrates how MLE is biased toward majority
    - Shows why different metrics matter
    """
    rng = np.random.default_rng(random_state)

    # Number of samples in each class
    n_minority = int(n_samples * imbalance_ratio)
    n_majority = n_samples - n_minority

    # Generate majority class (y=0) centered at origin
    X_majority = rng.standard_normal((n_majority, n_features))
    y_majority = np.zeros(n_majority)

    # Generate minority class (y=1) shifted away from origin
    X_minority = rng.standard_normal((n_minority, n_features))
    # Shift first few features to create separation
    shift = np.zeros(n_features)
    shift[: min(3, n_features)] = class_separation
    X_minority = X_minority + shift

    y_minority = np.ones(n_minority)

    # Combine and shuffle
    X = np.vstack([X_majority, X_minority])
    y = np.concatenate([y_majority, y_minority])

    # Shuffle
    shuffle_idx = rng.permutation(n_samples)
    X = X[shuffle_idx]
    y = y[shuffle_idx]

    return X, y


def load_real_dataset(
    dataset_name: str = "california_housing",
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Load a real-world dataset from sklearn.

    Parameters
    ----------
    dataset_name : str
        One of: 'california_housing', 'diabetes' (regression)
                'breast_cancer', 'wine' (classification)

    Returns
    -------
    X : np.ndarray
        Features
    y : np.ndarray
        Target
    feature_names : list[str]
        Names of features

    Notes
    -----
    Real data helps validate that synthetic examples generalize.
    """
    loaders = {
        "california_housing": fetch_california_housing,
        "diabetes": load_diabetes,
        "breast_cancer": load_breast_cancer,
        "wine": load_wine,
    }

    if dataset_name not in loaders:
        available = ", ".join(loaders.keys())
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {available}")

    data = loaders[dataset_name]()
    X = data.data
    y = data.target
    feature_names = list(data.feature_names)

    return X, y, feature_names


def generate_heteroscedastic_data(
    n_samples: int = 1000,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate data with heteroscedastic noise (variance depends on X).

    Parameters
    ----------
    n_samples : int
        Number of samples
    random_state : int
        Random seed

    Returns
    -------
    X : np.ndarray of shape (n_samples, 1)
        Features
    y : np.ndarray of shape (n_samples,)
        Targets
    true_signal : np.ndarray of shape (n_samples,)
        True signal without noise

    Notes
    -----
    Useful for showing residual analysis - variance should NOT be constant.
    This violates homoscedasticity assumption of linear regression.
    """
    rng = np.random.default_rng(random_state)

    X = rng.uniform(0, 10, (n_samples, 1))
    x_flat = X.flatten()

    # True linear relationship
    true_signal = 2 * x_flat + 1

    # Noise variance increases with X
    noise_std = 0.1 + 0.3 * x_flat
    noise = rng.normal(0, 1, n_samples) * noise_std

    y = true_signal + noise

    return X, y, true_signal
