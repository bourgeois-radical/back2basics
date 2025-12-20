"""
Theory module for MLE and Loss Functions presentation.

This module provides functions to demonstrate theoretical concepts,
including MLE derivations and the connection between distributions and loss functions.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from scipy import stats


def derive_mse_from_gaussian_mle(show_steps: bool = True) -> str:
    """
    Show mathematical derivation: Gaussian MLE -> MSE.

    Parameters
    ----------
    show_steps : bool
        If True, return formatted derivation

    Returns
    -------
    derivation : str
        Formatted markdown/text showing derivation

    Notes
    -----
    Makes the connection explicit:
    1. Assume residuals follow Gaussian distribution
    2. Write likelihood function
    3. Take log to get log-likelihood
    4. Negate to get loss function
    5. Result is proportional to MSE
    """
    derivation = """
================================================================================
                    DERIVING MSE FROM GAUSSIAN MLE
================================================================================

ASSUMPTION: Residuals follow a Gaussian distribution
            epsilon_i = y_i - f(x_i) ~ N(0, sigma^2)

STEP 1: Write the likelihood for a single observation
------------------------------------------------------------------------
        p(y_i | x_i, theta) = (1 / sqrt(2*pi*sigma^2)) * exp(-(y_i - f(x_i))^2 / (2*sigma^2))

STEP 2: Write the likelihood for all observations (assuming independence)
------------------------------------------------------------------------
        L(theta) = prod_{i=1}^{n} p(y_i | x_i, theta)

                 = prod_{i=1}^{n} (1 / sqrt(2*pi*sigma^2)) * exp(-(y_i - f(x_i))^2 / (2*sigma^2))

STEP 3: Take the log (log-likelihood)
------------------------------------------------------------------------
        log L(theta) = sum_{i=1}^{n} [ -1/2 * log(2*pi*sigma^2) - (y_i - f(x_i))^2 / (2*sigma^2) ]

                     = -n/2 * log(2*pi*sigma^2) - (1 / (2*sigma^2)) * sum_{i=1}^{n} (y_i - f(x_i))^2

STEP 4: Maximize log-likelihood = Minimize negative log-likelihood
------------------------------------------------------------------------
        To find the MLE, we maximize log L(theta).
        Equivalently, we minimize -log L(theta):

        -log L(theta) = n/2 * log(2*pi*sigma^2) + (1 / (2*sigma^2)) * sum_{i=1}^{n} (y_i - f(x_i))^2

STEP 5: Drop constants (they don't affect the argmin)
------------------------------------------------------------------------
        argmin_theta [ -log L(theta) ] = argmin_theta [ sum_{i=1}^{n} (y_i - f(x_i))^2 ]

                                        = argmin_theta [ n * MSE ]

        Since n is constant:

                                        = argmin_theta [ MSE ]

================================================================================
                            CONCLUSION
================================================================================

        Gaussian noise assumption  -->  MSE loss function

        When we minimize MSE, we are implicitly assuming that the residuals
        follow a Gaussian distribution!

        Corollary: The MEAN is the MLE estimator for the center of a Gaussian.
================================================================================
"""
    if show_steps:
        return derivation
    return "Gaussian MLE -> MSE (see derive_mse_from_gaussian_mle(show_steps=True))"


def derive_mae_from_laplace_mle(show_steps: bool = True) -> str:
    """
    Show mathematical derivation: Laplace MLE -> MAE.

    Parameters
    ----------
    show_steps : bool
        If True, return formatted derivation

    Returns
    -------
    derivation : str
        Formatted derivation

    Notes
    -----
    Parallel to Gaussian derivation but with Laplace distribution.
    """
    derivation = """
================================================================================
                    DERIVING MAE FROM LAPLACE MLE
================================================================================

ASSUMPTION: Residuals follow a Laplace distribution
            epsilon_i = y_i - f(x_i) ~ Laplace(0, b)

STEP 1: Write the Laplace PDF for a single observation
------------------------------------------------------------------------
        p(y_i | x_i, theta) = (1 / (2*b)) * exp(-|y_i - f(x_i)| / b)

        where b is the scale parameter (related to variance: Var = 2*b^2)

STEP 2: Write the likelihood for all observations
------------------------------------------------------------------------
        L(theta) = prod_{i=1}^{n} (1 / (2*b)) * exp(-|y_i - f(x_i)| / b)

STEP 3: Take the log (log-likelihood)
------------------------------------------------------------------------
        log L(theta) = sum_{i=1}^{n} [ -log(2*b) - |y_i - f(x_i)| / b ]

                     = -n * log(2*b) - (1/b) * sum_{i=1}^{n} |y_i - f(x_i)|

STEP 4: Maximize log-likelihood = Minimize negative log-likelihood
------------------------------------------------------------------------
        -log L(theta) = n * log(2*b) + (1/b) * sum_{i=1}^{n} |y_i - f(x_i)|

STEP 5: Drop constants
------------------------------------------------------------------------
        argmin_theta [ -log L(theta) ] = argmin_theta [ sum_{i=1}^{n} |y_i - f(x_i)| ]

                                        = argmin_theta [ n * MAE ]

                                        = argmin_theta [ MAE ]

================================================================================
                            CONCLUSION
================================================================================

        Laplace noise assumption  -->  MAE loss function

        When we minimize MAE, we are implicitly assuming that the residuals
        follow a Laplace distribution!

        Corollary: The MEDIAN is the MLE estimator for the center of a Laplace.

        WHY THE MEDIAN?
        - Laplace distribution has heavier tails than Gaussian
        - The median is robust to outliers (extreme values)
        - This explains why MAE is more robust than MSE!
================================================================================
"""
    if show_steps:
        return derivation
    return "Laplace MLE -> MAE (see derive_mae_from_laplace_mle(show_steps=True))"


def demonstrate_mean_minimizes_mse(
    data: np.ndarray,
    candidate_values: np.ndarray | None = None,
    show_plot: bool = True,
) -> tuple[float, np.ndarray, Figure | None]:
    """
    Empirically show that mean minimizes MSE.

    Parameters
    ----------
    data : np.ndarray
        Sample data
    candidate_values : np.ndarray, optional
        Candidate estimates to try (default: range around mean)
    show_plot : bool
        If True, plot MSE vs candidate value

    Returns
    -------
    optimal_value : float
        Value that minimizes MSE (should equal mean)
    mse_values : np.ndarray
        MSE for each candidate
    fig : matplotlib Figure (if show_plot=True)

    Notes
    -----
    Visual proof that arithmetic mean is MLE under Gaussian.
    Shows parabolic MSE curve with minimum at mean.
    """
    data = np.asarray(data).flatten()
    data_mean = np.mean(data)

    if candidate_values is None:
        data_range = data.max() - data.min()
        candidate_values = np.linspace(
            data_mean - 0.5 * data_range,
            data_mean + 0.5 * data_range,
            200,
        )

    # Calculate MSE for each candidate
    mse_values = np.array([np.mean((data - c) ** 2) for c in candidate_values])

    # Find optimal
    optimal_idx = np.argmin(mse_values)
    optimal_value = candidate_values[optimal_idx]

    fig = None
    if show_plot:
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(candidate_values, mse_values, "b-", linewidth=2, label="MSE")
        ax.axvline(data_mean, color="red", linestyle="--", linewidth=2, label=f"Mean = {data_mean:.3f}")
        ax.axvline(optimal_value, color="green", linestyle=":", linewidth=2, label=f"Argmin MSE = {optimal_value:.3f}")
        ax.scatter([optimal_value], [mse_values[optimal_idx]], color="green", s=100, zorder=5)

        ax.set_xlabel("Candidate Value (mu)", fontsize=12)
        ax.set_ylabel("Mean Squared Error", fontsize=12)
        ax.set_title("MSE is Minimized at the Mean", fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add annotation
        ax.text(
            0.02, 0.98,
            "The MSE curve is a parabola.\nIts minimum is at the arithmetic mean.\nThis is why Gaussian MLE -> MSE -> Mean!",
            transform=ax.transAxes,
            verticalalignment="top",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    return optimal_value, mse_values, fig


def demonstrate_median_minimizes_mae(
    data: np.ndarray,
    candidate_values: np.ndarray | None = None,
    show_plot: bool = True,
) -> tuple[float, np.ndarray, Figure | None]:
    """
    Empirically show that median minimizes MAE.

    Parameters
    ----------
    data : np.ndarray
        Sample data
    candidate_values : np.ndarray, optional
        Candidate estimates to try
    show_plot : bool
        If True, plot MAE vs candidate value

    Returns
    -------
    optimal_value : float
        Value that minimizes MAE (should equal median)
    mae_values : np.ndarray
        MAE for each candidate
    fig : matplotlib Figure (if show_plot=True)

    Notes
    -----
    Visual proof that median is MLE under Laplace.
    Shows V-shaped MAE curve with minimum at median.
    """
    data = np.asarray(data).flatten()
    data_median = np.median(data)

    if candidate_values is None:
        data_range = data.max() - data.min()
        candidate_values = np.linspace(
            data_median - 0.5 * data_range,
            data_median + 0.5 * data_range,
            200,
        )

    # Calculate MAE for each candidate
    mae_values = np.array([np.mean(np.abs(data - c)) for c in candidate_values])

    # Find optimal
    optimal_idx = np.argmin(mae_values)
    optimal_value = candidate_values[optimal_idx]

    fig = None
    if show_plot:
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(candidate_values, mae_values, "g-", linewidth=2, label="MAE")
        ax.axvline(data_median, color="red", linestyle="--", linewidth=2, label=f"Median = {data_median:.3f}")
        ax.axvline(optimal_value, color="orange", linestyle=":", linewidth=2, label=f"Argmin MAE = {optimal_value:.3f}")
        ax.scatter([optimal_value], [mae_values[optimal_idx]], color="orange", s=100, zorder=5)

        ax.set_xlabel("Candidate Value (mu)", fontsize=12)
        ax.set_ylabel("Mean Absolute Error", fontsize=12)
        ax.set_title("MAE is Minimized at the Median", fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add annotation
        ax.text(
            0.02, 0.98,
            "The MAE curve has a kink at the median.\nIts minimum is at the median.\nThis is why Laplace MLE -> MAE -> Median!",
            transform=ax.transAxes,
            verticalalignment="top",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    return optimal_value, mae_values, fig


def compare_estimators_with_outliers(
    data_clean: np.ndarray,
    data_with_outliers: np.ndarray,
    show_plot: bool = True,
) -> tuple[dict, Figure | None]:
    """
    Compare mean vs median as estimators with/without outliers.

    Parameters
    ----------
    data_clean : np.ndarray
        Data without outliers
    data_with_outliers : np.ndarray
        Same data with added outliers
    show_plot : bool
        Create visualization

    Returns
    -------
    results : dict
        Comparison results for clean and outlier data
    fig : matplotlib Figure (if show_plot=True)

    Notes
    -----
    Shows why MAE is "robust":
    - Mean shifts dramatically with outliers
    - Median stays stable
    """
    results = {
        "clean": {
            "mean": np.mean(data_clean),
            "median": np.median(data_clean),
            "std": np.std(data_clean),
        },
        "with_outliers": {
            "mean": np.mean(data_with_outliers),
            "median": np.median(data_with_outliers),
            "std": np.std(data_with_outliers),
        },
    }

    # Calculate shifts
    results["mean_shift"] = results["with_outliers"]["mean"] - results["clean"]["mean"]
    results["median_shift"] = results["with_outliers"]["median"] - results["clean"]["median"]

    fig = None
    if show_plot:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Clean data
        ax1 = axes[0]
        ax1.hist(data_clean, bins=30, alpha=0.7, edgecolor="black", color="steelblue")
        ax1.axvline(results["clean"]["mean"], color="red", linestyle="--", linewidth=2, label=f"Mean = {results['clean']['mean']:.2f}")
        ax1.axvline(results["clean"]["median"], color="green", linestyle=":", linewidth=2, label=f"Median = {results['clean']['median']:.2f}")
        ax1.set_title("Clean Data")
        ax1.set_xlabel("Value")
        ax1.set_ylabel("Frequency")
        ax1.legend()

        # Data with outliers
        ax2 = axes[1]
        ax2.hist(data_with_outliers, bins=30, alpha=0.7, edgecolor="black", color="steelblue")
        ax2.axvline(results["with_outliers"]["mean"], color="red", linestyle="--", linewidth=2, label=f"Mean = {results['with_outliers']['mean']:.2f}")
        ax2.axvline(results["with_outliers"]["median"], color="green", linestyle=":", linewidth=2, label=f"Median = {results['with_outliers']['median']:.2f}")
        ax2.set_title("Data with Outliers")
        ax2.set_xlabel("Value")
        ax2.set_ylabel("Frequency")
        ax2.legend()

        # Add summary
        fig.suptitle(
            f"Mean shift: {results['mean_shift']:.2f}  |  Median shift: {results['median_shift']:.2f}",
            fontsize=12,
            fontweight="bold",
        )

        plt.tight_layout()

    return results, fig


def simulate_bias_variance_tradeoff(
    true_function: callable,
    noise_std: float,
    model_complexities: list[int],
    n_datasets: int = 50,
    n_samples_per_dataset: int = 50,
    x_range: tuple = (-3, 3),
    random_state: int = 42,
) -> tuple[dict, Figure]:
    """
    Simulate bias-variance decomposition across model complexities.

    Parameters
    ----------
    true_function : callable
        True underlying function f(x)
    noise_std : float
        Standard deviation of Gaussian noise
    model_complexities : list
        List of polynomial degrees to try (e.g., [1, 3, 5, 10, 15])
    n_datasets : int
        Number of datasets to generate
    n_samples_per_dataset : int
        Samples per dataset
    x_range : tuple
        Range of x values
    random_state : int
        Random seed

    Returns
    -------
    results : dict
        For each complexity: bias^2, variance, total error
    fig : matplotlib Figure
        Plot showing bias-variance tradeoff
    """
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures

    rng = np.random.default_rng(random_state)

    # Fixed test points for evaluation
    x_test = np.linspace(x_range[0], x_range[1], 100).reshape(-1, 1)
    y_true = true_function(x_test.flatten())

    results = {complexity: {"predictions": []} for complexity in model_complexities}

    # Generate multiple datasets and train models
    for _ in range(n_datasets):
        # Generate training data
        x_train = rng.uniform(x_range[0], x_range[1], (n_samples_per_dataset, 1))
        y_train = true_function(x_train.flatten()) + rng.normal(0, noise_std, n_samples_per_dataset)

        for complexity in model_complexities:
            # Fit polynomial model
            poly = PolynomialFeatures(degree=complexity, include_bias=False)
            X_train_poly = poly.fit_transform(x_train)
            X_test_poly = poly.transform(x_test)

            model = LinearRegression()
            model.fit(X_train_poly, y_train)

            y_pred = model.predict(X_test_poly)
            results[complexity]["predictions"].append(y_pred)

    # Calculate bias, variance, and total error
    for complexity in model_complexities:
        predictions = np.array(results[complexity]["predictions"])  # (n_datasets, n_test_points)

        mean_prediction = predictions.mean(axis=0)  # Average prediction at each test point
        variance = predictions.var(axis=0).mean()  # Average variance across test points
        bias_squared = ((mean_prediction - y_true) ** 2).mean()  # Average squared bias

        results[complexity]["bias_squared"] = bias_squared
        results[complexity]["variance"] = variance
        results[complexity]["irreducible_error"] = noise_std**2
        results[complexity]["total_error"] = bias_squared + variance + noise_std**2

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    complexities = list(model_complexities)
    bias_squared = [results[c]["bias_squared"] for c in complexities]
    variance = [results[c]["variance"] for c in complexities]
    total_error = [results[c]["total_error"] for c in complexities]
    irreducible = noise_std**2

    ax.plot(complexities, bias_squared, "b-o", label="Bias^2", linewidth=2)
    ax.plot(complexities, variance, "r-o", label="Variance", linewidth=2)
    ax.plot(complexities, total_error, "g-o", label="Total Error", linewidth=2)
    ax.axhline(irreducible, color="gray", linestyle="--", label=f"Irreducible = {irreducible:.3f}")

    ax.set_xlabel("Model Complexity (Polynomial Degree)", fontsize=12)
    ax.set_ylabel("Error", fontsize=12)
    ax.set_title("Bias-Variance Tradeoff", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Find optimal complexity
    optimal_idx = np.argmin(total_error)
    ax.scatter([complexities[optimal_idx]], [total_error[optimal_idx]], s=200, color="green", zorder=5, marker="*")
    ax.annotate(
        f"Optimal: degree={complexities[optimal_idx]}",
        (complexities[optimal_idx], total_error[optimal_idx]),
        xytext=(10, 20),
        textcoords="offset points",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    return results, fig


def compare_gaussian_vs_laplace(
    data: np.ndarray,
    figsize: tuple = (12, 5),
) -> Figure:
    """
    Visually compare Gaussian vs Laplace fit to data.

    Parameters
    ----------
    data : np.ndarray
        Sample data (e.g., residuals)
    figsize : tuple
        Figure size

    Returns
    -------
    fig : matplotlib Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    data = np.asarray(data).flatten()

    # Fit parameters
    mu_gauss, std_gauss = np.mean(data), np.std(data)
    mu_lap = np.median(data)
    scale_lap = np.mean(np.abs(data - mu_lap))  # MLE for Laplace scale

    x_range = np.linspace(data.min() - 1, data.max() + 1, 200)

    # Panel 1: Histogram with both fits
    ax1 = axes[0]
    ax1.hist(data, bins=30, density=True, alpha=0.7, edgecolor="black", color="steelblue", label="Data")
    ax1.plot(x_range, stats.norm.pdf(x_range, mu_gauss, std_gauss), "r-", linewidth=2, label="Gaussian fit")
    ax1.plot(x_range, stats.laplace.pdf(x_range, mu_lap, scale_lap), "g-", linewidth=2, label="Laplace fit")
    ax1.set_xlabel("Value")
    ax1.set_ylabel("Density")
    ax1.set_title("Gaussian vs Laplace Fit")
    ax1.legend()

    # Panel 2: Q-Q plots
    ax2 = axes[1]

    # Gaussian Q-Q
    theoretical_quantiles_gauss = stats.norm.ppf(np.linspace(0.01, 0.99, len(data)))
    sample_quantiles = np.sort((data - mu_gauss) / std_gauss)
    ax2.scatter(theoretical_quantiles_gauss, sample_quantiles[:len(theoretical_quantiles_gauss)], alpha=0.5, s=20, label="Gaussian Q-Q")

    # Reference line
    ax2.plot([-3, 3], [-3, 3], "r--", linewidth=2)

    ax2.set_xlabel("Theoretical Quantiles")
    ax2.set_ylabel("Sample Quantiles")
    ax2.set_title("Q-Q Plot (Gaussian)")
    ax2.set_xlim(-3.5, 3.5)
    ax2.set_ylim(-3.5, 3.5)

    plt.tight_layout()
    return fig
