"""
Visualization module for MLE and Loss Functions presentation.

This module provides all visualization functions for the presentation,
including residual diagnostics, loss function comparisons, and MLE derivations.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from scipy import stats

from metrics import calculate_regression_metrics


def plot_three_distributions_concept(
    X: np.ndarray,
    y: np.ndarray,
    y_pred: np.ndarray,
    feature_idx: int = 0,
    figsize: tuple = (15, 5),
) -> Figure:
    """
    Create the KEY conceptual figure showing three distributions.

    This is the most important visualization for understanding.

    Parameters
    ----------
    X : np.ndarray
        Input features
    y : np.ndarray
        True targets
    y_pred : np.ndarray
        Model predictions
    feature_idx : int
        Which feature to plot on x-axis
    figsize : tuple
        Figure size

    Returns
    -------
    fig : matplotlib Figure
        Figure with 3 subplots showing:
        1. Distribution of input feature (histogram)
        2. Scatter plot: feature vs target, with model's learned function
        3. Histogram of residuals (y - y_pred) with fitted Gaussian

    Notes
    -----
    This figure makes crystal clear:
    - Input X has some distribution (often doesn't matter)
    - Model learns a FUNCTION f(X) -> predictions
    - Residuals have a distribution (this is what loss functions assume!)
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Get the feature to plot
    x_feature = X[:, feature_idx] if X.ndim > 1 else X.flatten()
    residuals = y - y_pred

    # Panel 1: Input distribution
    ax1 = axes[0]
    ax1.hist(x_feature, bins=30, edgecolor="black", alpha=0.7, color="steelblue")
    ax1.set_xlabel(f"Feature {feature_idx}")
    ax1.set_ylabel("Frequency")
    ax1.set_title("1. Distribution of Input Data (X)")
    ax1.axvline(np.mean(x_feature), color="red", linestyle="--", label=f"Mean={np.mean(x_feature):.2f}")
    ax1.legend()

    # Panel 2: Function learned by model
    ax2 = axes[1]
    # Sort for line plot
    sort_idx = np.argsort(x_feature)
    ax2.scatter(x_feature, y, alpha=0.3, s=10, label="Data points", color="steelblue")
    ax2.plot(x_feature[sort_idx], y_pred[sort_idx], color="red", linewidth=2, label="Model f(X)")
    ax2.set_xlabel(f"Feature {feature_idx}")
    ax2.set_ylabel("Target y")
    ax2.set_title("2. Function Learned by Model (f(X) -> y)")
    ax2.legend()

    # Panel 3: Residual distribution
    ax3 = axes[2]
    ax3.hist(residuals, bins=30, edgecolor="black", alpha=0.7, density=True, color="steelblue")

    # Overlay fitted Gaussian
    mu, std = np.mean(residuals), np.std(residuals)
    x_range = np.linspace(residuals.min(), residuals.max(), 100)
    gaussian = stats.norm.pdf(x_range, mu, std)
    ax3.plot(x_range, gaussian, "r-", linewidth=2, label=f"Gaussian fit\nmu={mu:.2f}, sigma={std:.2f}")

    ax3.set_xlabel("Residual (y - f(X))")
    ax3.set_ylabel("Density")
    ax3.set_title("3. Distribution of Residuals")
    ax3.axvline(0, color="black", linestyle="--", alpha=0.5)
    ax3.legend()

    plt.tight_layout()
    return fig


def plot_loss_function_shapes(
    error_range: np.ndarray | None = None,
    figsize: tuple = (15, 4),
) -> Figure:
    """
    Plot the shape of different loss functions vs error magnitude.

    Parameters
    ----------
    error_range : np.ndarray, optional
        Range of errors to plot (default: -5 to 5)
    figsize : tuple
        Figure size

    Returns
    -------
    fig : matplotlib Figure
        Figure showing MSE, MAE, and Huber loss shapes

    Notes
    -----
    Visual interpretation:
    - MSE: Quadratic (steep for large errors) -> sensitive to outliers
    - MAE: Linear (constant slope) -> treats all errors equally, robust
    - Huber: Hybrid (quadratic near 0, linear far away) -> best of both
    """
    if error_range is None:
        error_range = np.linspace(-5, 5, 200)

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # MSE
    mse_loss = error_range**2
    axes[0].plot(error_range, mse_loss, "b-", linewidth=2)
    axes[0].set_title("MSE: L = (y - y_hat)^2")
    axes[0].set_xlabel("Error (y - y_hat)")
    axes[0].set_ylabel("Loss")
    axes[0].axvline(0, color="gray", linestyle="--", alpha=0.5)
    axes[0].text(0.05, 0.95, "Sensitive to outliers\n(quadratic penalty)",
                 transform=axes[0].transAxes, verticalalignment="top",
                 bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    axes[0].set_ylim(0, 25)

    # MAE
    mae_loss = np.abs(error_range)
    axes[1].plot(error_range, mae_loss, "g-", linewidth=2)
    axes[1].set_title("MAE: L = |y - y_hat|")
    axes[1].set_xlabel("Error (y - y_hat)")
    axes[1].set_ylabel("Loss")
    axes[1].axvline(0, color="gray", linestyle="--", alpha=0.5)
    axes[1].text(0.05, 0.95, "Robust to outliers\n(linear penalty)",
                 transform=axes[1].transAxes, verticalalignment="top",
                 bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    # Huber
    delta = 1.0
    huber_loss = np.where(
        np.abs(error_range) <= delta,
        0.5 * error_range**2,
        delta * (np.abs(error_range) - 0.5 * delta),
    )
    axes[2].plot(error_range, huber_loss, "r-", linewidth=2)
    axes[2].axvline(-delta, color="orange", linestyle=":", alpha=0.7, label=f"delta={delta}")
    axes[2].axvline(delta, color="orange", linestyle=":", alpha=0.7)
    axes[2].set_title("Huber: Hybrid Loss")
    axes[2].set_xlabel("Error (y - y_hat)")
    axes[2].set_ylabel("Loss")
    axes[2].axvline(0, color="gray", linestyle="--", alpha=0.5)
    axes[2].text(0.05, 0.95, "Best of both worlds\n(quadratic small, linear large)",
                 transform=axes[2].transAxes, verticalalignment="top",
                 bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    axes[2].legend()

    plt.tight_layout()
    return fig


def plot_mle_derivation_visual(
    y: np.ndarray,
    y_pred: np.ndarray,
    assumed_distribution: str = "gaussian",
    figsize: tuple = (15, 10),
) -> Figure:
    """
    Visualize the MLE derivation step-by-step.

    Parameters
    ----------
    y : np.ndarray
        True values
    y_pred : np.ndarray
        Predicted values
    assumed_distribution : str
        'gaussian' or 'laplace'
    figsize : tuple
        Figure size

    Returns
    -------
    fig : matplotlib Figure
        Multi-panel figure showing MLE to loss function connection
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    residuals = y - y_pred
    n_samples = len(y)

    # Panel 1: Individual sample likelihoods (show a few examples)
    ax1 = axes[0, 0]
    sample_indices = np.linspace(0, n_samples - 1, min(5, n_samples), dtype=int)

    x_range = np.linspace(residuals.min() - 2, residuals.max() + 2, 200)
    colors = plt.cm.viridis(np.linspace(0, 1, len(sample_indices)))

    for idx, color in zip(sample_indices, colors):
        if assumed_distribution == "gaussian":
            sigma = np.std(residuals)
            pdf = stats.norm.pdf(x_range, 0, sigma)
        else:  # laplace
            scale = np.std(residuals) / np.sqrt(2)
            pdf = stats.laplace.pdf(x_range, 0, scale)

        ax1.plot(x_range, pdf, color=color, alpha=0.7)
        ax1.axvline(residuals[idx], color=color, linestyle="--", alpha=0.5)

    ax1.set_title("1. Likelihood of Each Residual")
    ax1.set_xlabel("Residual value")
    ax1.set_ylabel("Probability density")

    # Panel 2: Log-likelihoods
    ax2 = axes[0, 1]
    if assumed_distribution == "gaussian":
        sigma = np.std(residuals)
        log_likelihoods = stats.norm.logpdf(residuals, 0, sigma)
    else:
        scale = np.std(residuals) / np.sqrt(2)
        log_likelihoods = stats.laplace.logpdf(residuals, 0, scale)

    ax2.hist(log_likelihoods, bins=30, edgecolor="black", alpha=0.7, color="steelblue")
    ax2.axvline(np.mean(log_likelihoods), color="red", linestyle="--",
                label=f"Mean log-L = {np.mean(log_likelihoods):.2f}")
    ax2.set_title("2. Log-Likelihood per Sample")
    ax2.set_xlabel("Log-likelihood")
    ax2.set_ylabel("Frequency")
    ax2.legend()

    total_ll = np.sum(log_likelihoods)
    ax2.text(0.05, 0.95, f"Total log-L = {total_ll:.2f}\nNLL = {-total_ll:.2f}",
             transform=ax2.transAxes, verticalalignment="top",
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    # Panel 3: Show NLL equals loss (up to constant)
    ax3 = axes[1, 0]
    if assumed_distribution == "gaussian":
        # NLL proportional to sum of squared errors
        loss_per_sample = residuals**2
        loss_name = "Squared Error"
    else:
        # NLL proportional to sum of absolute errors
        loss_per_sample = np.abs(residuals)
        loss_name = "Absolute Error"

    ax3.scatter(-log_likelihoods, loss_per_sample, alpha=0.5, s=20)
    ax3.set_xlabel("Negative Log-Likelihood (per sample)")
    ax3.set_ylabel(f"{loss_name} (per sample)")
    ax3.set_title("3. NLL vs Loss (Linear Relationship!)")

    # Add correlation
    corr = np.corrcoef(-log_likelihoods, loss_per_sample)[0, 1]
    ax3.text(0.05, 0.95, f"Correlation: {corr:.4f}",
             transform=ax3.transAxes, verticalalignment="top",
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    # Panel 4: Residual histogram with fitted distribution
    ax4 = axes[1, 1]
    ax4.hist(residuals, bins=30, edgecolor="black", alpha=0.7, density=True, color="steelblue")

    x_range = np.linspace(residuals.min(), residuals.max(), 100)
    if assumed_distribution == "gaussian":
        pdf = stats.norm.pdf(x_range, 0, np.std(residuals))
        dist_name = "Gaussian"
    else:
        scale = np.std(residuals) / np.sqrt(2)
        pdf = stats.laplace.pdf(x_range, 0, scale)
        dist_name = "Laplace"

    ax4.plot(x_range, pdf, "r-", linewidth=2, label=f"Fitted {dist_name}")
    ax4.set_title(f"4. Residuals with Fitted {dist_name}")
    ax4.set_xlabel("Residual")
    ax4.set_ylabel("Density")
    ax4.legend()

    plt.tight_layout()
    return fig


def plot_residuals_diagnostic(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Residual Diagnostics",
    figsize: tuple = (12, 10),
) -> Figure:
    """
    Create comprehensive residual diagnostic plots.

    Parameters
    ----------
    y_true : np.ndarray
        True target values
    y_pred : np.ndarray
        Predicted values
    title : str
        Overall title for the figure
    figsize : tuple
        Figure size

    Returns
    -------
    fig : matplotlib Figure
        2x2 subplot grid with diagnostic plots
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(title, fontsize=14, fontweight="bold")

    residuals = y_true - y_pred

    # [0, 0]: Residuals vs Fitted
    ax1 = axes[0, 0]
    ax1.scatter(y_pred, residuals, alpha=0.5, s=20, color="steelblue")
    ax1.axhline(0, color="red", linestyle="--", linewidth=2)
    ax1.set_xlabel("Fitted values")
    ax1.set_ylabel("Residuals")
    ax1.set_title("Residuals vs Fitted")

    # Add lowess trend line
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess
        smooth = lowess(residuals, y_pred, frac=0.3)
        ax1.plot(smooth[:, 0], smooth[:, 1], "orange", linewidth=2, label="Trend")
        ax1.legend()
    except ImportError:
        pass

    ax1.text(0.02, 0.98, "Look for: patterns (bias),\nheteroscedasticity",
             transform=ax1.transAxes, verticalalignment="top", fontsize=9,
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    # [0, 1]: Histogram of Residuals
    ax2 = axes[0, 1]
    ax2.hist(residuals, bins=30, edgecolor="black", alpha=0.7, density=True, color="steelblue")

    mu, std = np.mean(residuals), np.std(residuals)
    x_range = np.linspace(residuals.min(), residuals.max(), 100)
    gaussian = stats.norm.pdf(x_range, mu, std)
    ax2.plot(x_range, gaussian, "r-", linewidth=2, label=f"N({mu:.2f}, {std:.2f})")

    ax2.set_xlabel("Residuals")
    ax2.set_ylabel("Density")
    ax2.set_title("Histogram of Residuals")
    ax2.legend()
    ax2.text(0.02, 0.98, "Should look Gaussian\nif using MSE",
             transform=ax2.transAxes, verticalalignment="top", fontsize=9,
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    # [1, 0]: Q-Q Plot
    ax3 = axes[1, 0]
    stats.probplot(residuals, dist="norm", plot=ax3)
    ax3.set_title("Q-Q Plot (Normal)")
    ax3.text(0.02, 0.98, "Points on line =\ndistribution matches",
             transform=ax3.transAxes, verticalalignment="top", fontsize=9,
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    # [1, 1]: Scale-Location Plot
    ax4 = axes[1, 1]
    sqrt_abs_residuals = np.sqrt(np.abs(residuals))
    ax4.scatter(y_pred, sqrt_abs_residuals, alpha=0.5, s=20, color="steelblue")
    ax4.set_xlabel("Fitted values")
    ax4.set_ylabel("sqrt(|Residuals|)")
    ax4.set_title("Scale-Location Plot")

    # Add trend line
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess
        smooth = lowess(sqrt_abs_residuals, y_pred, frac=0.3)
        ax4.plot(smooth[:, 0], smooth[:, 1], "red", linewidth=2)
    except ImportError:
        pass

    ax4.text(0.02, 0.98, "Flat trend =\nconstant variance",
             transform=ax4.transAxes, verticalalignment="top", fontsize=9,
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.tight_layout()
    return fig


def plot_bias_variance_decomposition(
    X: np.ndarray,
    y_true: np.ndarray,
    true_signal: np.ndarray,
    y_pred: np.ndarray,
    feature_idx: int = 0,
    figsize: tuple = (15, 5),
) -> Figure:
    """
    Visualize bias-variance decomposition in residuals.

    Parameters
    ----------
    X : np.ndarray
        Features
    y_true : np.ndarray
        Observed targets (signal + noise)
    true_signal : np.ndarray
        True underlying signal (available in synthetic data)
    y_pred : np.ndarray
        Model predictions
    feature_idx : int
        Which feature to plot
    figsize : tuple
        Figure size

    Returns
    -------
    fig : matplotlib Figure
        3 subplots showing decomposition

    Notes
    -----
    Makes explicit: residuals = irreducible noise + systematic bias
    Only possible with synthetic data where true_signal is known.
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    x_feature = X[:, feature_idx] if X.ndim > 1 else X.flatten()

    # Decomposition
    true_noise = y_true - true_signal
    model_bias = true_signal - y_pred
    observed_residuals = y_true - y_pred

    # Panel 1: Irreducible Noise
    ax1 = axes[0]
    ax1.hist(true_noise, bins=30, edgecolor="black", alpha=0.7, color="steelblue")
    ax1.axvline(0, color="red", linestyle="--")
    ax1.set_xlabel("True Noise (epsilon)")
    ax1.set_ylabel("Frequency")
    ax1.set_title("1. Irreducible Noise (y - f_true)")
    ax1.text(0.05, 0.95, f"Std: {np.std(true_noise):.3f}",
             transform=ax1.transAxes, verticalalignment="top",
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    # Panel 2: Model Bias
    ax2 = axes[1]
    sort_idx = np.argsort(x_feature)
    ax2.scatter(x_feature, model_bias, alpha=0.5, s=20, color="steelblue")
    ax2.axhline(0, color="red", linestyle="--")
    ax2.set_xlabel(f"Feature {feature_idx}")
    ax2.set_ylabel("Bias")
    ax2.set_title("2. Model Bias (f_true - f_model)")
    ax2.text(0.05, 0.95, f"Mean bias: {np.mean(model_bias):.3f}\nStd: {np.std(model_bias):.3f}",
             transform=ax2.transAxes, verticalalignment="top",
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    # Panel 3: Observed Residuals
    ax3 = axes[2]
    ax3.hist(observed_residuals, bins=30, edgecolor="black", alpha=0.7, color="steelblue")
    ax3.axvline(0, color="red", linestyle="--")
    ax3.set_xlabel("Residual")
    ax3.set_ylabel("Frequency")
    ax3.set_title("3. Observed Residuals (noise + bias)")
    ax3.text(0.05, 0.95, f"Std: {np.std(observed_residuals):.3f}",
             transform=ax3.transAxes, verticalalignment="top",
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.tight_layout()
    return fig


def plot_loss_function_comparison(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    models: dict,
    feature_idx: int = 0,
    figsize: tuple = (16, 12),
) -> Figure:
    """
    Compare multiple models trained with different loss functions.

    Parameters
    ----------
    X_train, y_train : np.ndarray
        Training data
    X_test, y_test : np.ndarray
        Test data
    models : dict
        Dictionary of trained models with structure:
        {'model_name': {'model': model, 'test_preds': array, ...}}
    feature_idx : int
        Which feature to plot
    figsize : tuple
        Figure size

    Returns
    -------
    fig : matplotlib Figure
        Grid of subplots comparing models
    """
    n_models = len(models)
    fig, axes = plt.subplots(n_models, 4, figsize=figsize)

    if n_models == 1:
        axes = axes.reshape(1, -1)

    x_test_feature = X_test[:, feature_idx] if X_test.ndim > 1 else X_test.flatten()

    for i, (name, results) in enumerate(models.items()):
        y_pred = results["test_preds"]
        residuals = y_test - y_pred
        metrics = calculate_regression_metrics(y_test, y_pred)

        # Column 1: Predictions vs True
        ax1 = axes[i, 0]
        ax1.scatter(y_test, y_pred, alpha=0.5, s=20, color="steelblue")
        lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
        ax1.plot(lims, lims, "r--", linewidth=2, label="Perfect fit")
        ax1.set_xlabel("True")
        ax1.set_ylabel("Predicted")
        ax1.set_title(f"{name}: Pred vs True")
        ax1.legend()

        # Column 2: Residual histogram
        ax2 = axes[i, 1]
        ax2.hist(residuals, bins=30, edgecolor="black", alpha=0.7, color="steelblue")
        ax2.axvline(0, color="red", linestyle="--")
        ax2.set_xlabel("Residual")
        ax2.set_ylabel("Frequency")
        ax2.set_title(f"{name}: Residuals")

        # Column 3: Residuals vs Fitted
        ax3 = axes[i, 2]
        ax3.scatter(y_pred, residuals, alpha=0.5, s=20, color="steelblue")
        ax3.axhline(0, color="red", linestyle="--")
        ax3.set_xlabel("Fitted")
        ax3.set_ylabel("Residual")
        ax3.set_title(f"{name}: Resid vs Fitted")

        # Column 4: Metrics
        ax4 = axes[i, 3]
        ax4.axis("off")
        metrics_text = (
            f"MSE:  {metrics['mse']:.4f}\n"
            f"RMSE: {metrics['rmse']:.4f}\n"
            f"MAE:  {metrics['mae']:.4f}\n"
            f"R^2:  {metrics['r2']:.4f}"
        )
        ax4.text(0.1, 0.5, metrics_text, transform=ax4.transAxes,
                 fontsize=12, verticalalignment="center", fontfamily="monospace",
                 bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
        ax4.set_title(f"{name}: Metrics")

    plt.tight_layout()
    return fig


def plot_classification_residuals(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    title: str = "Classification Residuals",
    figsize: tuple = (15, 5),
) -> Figure:
    """
    Visualize 'residuals' for classification (y - p_hat).

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels (0 or 1)
    y_pred_proba : np.ndarray
        Predicted probabilities P(y=1)
    title : str
        Figure title
    figsize : tuple
        Figure size

    Returns
    -------
    fig : matplotlib Figure
        3 subplots showing classification residuals analysis
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.suptitle(title, fontsize=14, fontweight="bold")

    residuals = y_true - y_pred_proba

    # Panel 1: Residuals by True Class (violin plot alternative using histograms)
    ax1 = axes[0]
    residuals_0 = residuals[y_true == 0]
    residuals_1 = residuals[y_true == 1]

    ax1.hist(residuals_0, bins=20, alpha=0.6, label="Class 0", color="blue", density=True)
    ax1.hist(residuals_1, bins=20, alpha=0.6, label="Class 1", color="orange", density=True)
    ax1.axvline(0, color="red", linestyle="--")
    ax1.set_xlabel("Residual (y - p_hat)")
    ax1.set_ylabel("Density")
    ax1.set_title("Residuals by True Class")
    ax1.legend()

    # Panel 2: Histogram of Predicted Probabilities
    ax2 = axes[1]
    proba_0 = y_pred_proba[y_true == 0]
    proba_1 = y_pred_proba[y_true == 1]

    ax2.hist(proba_0, bins=20, alpha=0.6, label="Class 0", color="blue", density=True)
    ax2.hist(proba_1, bins=20, alpha=0.6, label="Class 1", color="orange", density=True)
    ax2.set_xlabel("Predicted Probability")
    ax2.set_ylabel("Density")
    ax2.set_title("Predicted Probabilities by Class")
    ax2.legend()

    # Panel 3: Reliability Diagram (Calibration)
    ax3 = axes[2]
    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    bin_true_probs = []
    bin_pred_probs = []

    for i in range(n_bins):
        mask = (y_pred_proba >= bin_edges[i]) & (y_pred_proba < bin_edges[i + 1])
        if mask.sum() > 0:
            bin_true_probs.append(y_true[mask].mean())
            bin_pred_probs.append(y_pred_proba[mask].mean())
        else:
            bin_true_probs.append(np.nan)
            bin_pred_probs.append(np.nan)

    ax3.plot([0, 1], [0, 1], "r--", label="Perfect calibration")
    ax3.scatter(bin_pred_probs, bin_true_probs, s=100, color="steelblue", zorder=5)
    ax3.plot(bin_pred_probs, bin_true_probs, "b-", alpha=0.5)
    ax3.set_xlabel("Mean Predicted Probability")
    ax3.set_ylabel("Fraction of Positives")
    ax3.set_title("Reliability Diagram")
    ax3.legend()
    ax3.set_xlim(-0.05, 1.05)
    ax3.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    return fig


def plot_imbalance_effect(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    threshold: float = 0.5,
    figsize: tuple = (15, 10),
) -> Figure:
    """
    Visualize how class imbalance affects model and metrics.

    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_pred_proba : np.ndarray
        Predicted probabilities
    threshold : float
        Classification threshold
    figsize : tuple
        Figure size

    Returns
    -------
    fig : matplotlib Figure
        2x2 grid showing imbalance effects
    """
    from sklearn.metrics import precision_recall_curve

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    y_pred = (y_pred_proba >= threshold).astype(int)

    # [0, 0]: Class Distribution
    ax1 = axes[0, 0]
    class_counts = [np.sum(y_true == 0), np.sum(y_true == 1)]
    bars = ax1.bar(["Class 0 (Majority)", "Class 1 (Minority)"], class_counts,
                   color=["steelblue", "orange"])
    ax1.set_ylabel("Count")
    ax1.set_title("Class Distribution")

    # Add percentage labels
    total = sum(class_counts)
    for bar, count in zip(bars, class_counts):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f"{count} ({100 * count / total:.1f}%)",
                 ha="center", va="bottom")

    # [0, 1]: Predicted Probability Distribution by Class
    ax2 = axes[0, 1]
    proba_0 = y_pred_proba[y_true == 0]
    proba_1 = y_pred_proba[y_true == 1]

    ax2.hist(proba_0, bins=30, alpha=0.6, label="Class 0", color="steelblue", density=True)
    ax2.hist(proba_1, bins=30, alpha=0.6, label="Class 1", color="orange", density=True)
    ax2.axvline(threshold, color="red", linestyle="--", label=f"Threshold={threshold}")
    ax2.set_xlabel("Predicted Probability")
    ax2.set_ylabel("Density")
    ax2.set_title("Probability Distribution by Class")
    ax2.legend()

    # [1, 0]: Confusion Matrix
    ax3 = axes[1, 0]
    cm = np.array([[np.sum((y_true == 0) & (y_pred == 0)), np.sum((y_true == 0) & (y_pred == 1))],
                   [np.sum((y_true == 1) & (y_pred == 0)), np.sum((y_true == 1) & (y_pred == 1))]])

    im = ax3.imshow(cm, cmap="Blues")
    ax3.set_xticks([0, 1])
    ax3.set_yticks([0, 1])
    ax3.set_xticklabels(["Pred 0", "Pred 1"])
    ax3.set_yticklabels(["True 0", "True 1"])
    ax3.set_title(f"Confusion Matrix (threshold={threshold})")

    # Add text annotations
    for i in range(2):
        for j in range(2):
            ax3.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=14,
                     color="white" if cm[i, j] > cm.max() / 2 else "black")

    # [1, 1]: Precision, Recall, F1 vs Threshold
    ax4 = axes[1, 1]
    thresholds_range = np.linspace(0.01, 0.99, 50)

    precisions = []
    recalls = []
    f1s = []

    for t in thresholds_range:
        y_pred_t = (y_pred_proba >= t).astype(int)
        tp = np.sum((y_true == 1) & (y_pred_t == 1))
        fp = np.sum((y_true == 0) & (y_pred_t == 1))
        fn = np.sum((y_true == 1) & (y_pred_t == 0))

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)

    ax4.plot(thresholds_range, precisions, "b-", label="Precision")
    ax4.plot(thresholds_range, recalls, "g-", label="Recall")
    ax4.plot(thresholds_range, f1s, "r-", label="F1")
    ax4.axvline(threshold, color="gray", linestyle="--", alpha=0.5, label=f"Current={threshold}")
    ax4.set_xlabel("Threshold")
    ax4.set_ylabel("Score")
    ax4.set_title("Metrics vs Threshold")
    ax4.legend()
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1.05)

    plt.tight_layout()
    return fig


# =============================================================================
# PART 2: CLASSIFICATION VISUALIZATIONS
# =============================================================================


def plot_cross_entropy_loss_surface(
    y_true: np.ndarray | None = None,
    p_range: np.ndarray | None = None,
    figsize: tuple = (12, 5),
) -> Figure:
    """
    Visualize cross-entropy loss as function of predicted probability.

    Parallel to plotting MSE/MAE curves for regression.

    Parameters
    ----------
    y_true : np.ndarray, optional
        True labels (for computing overall loss). If None, shows theoretical curves.
    p_range : np.ndarray, optional
        Range of probabilities to plot (default: 0.01 to 0.99)
    figsize : tuple
        Figure size

    Returns
    -------
    fig : matplotlib Figure
        2 subplots showing loss curves for y=1 and y=0 cases

    Notes
    -----
    Shows how cross-entropy heavily penalizes confident wrong predictions.
    Explains why models become well-calibrated with this loss.
    Compare to MSE (quadratic) and MAE (linear) shapes from regression.
    """
    if p_range is None:
        p_range = np.linspace(0.01, 0.99, 200)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Panel 1: Loss for y=1 (positive class)
    ax1 = axes[0]
    loss_y1 = -np.log(p_range)  # -log(p) when y=1

    ax1.plot(p_range, loss_y1, "b-", linewidth=2)
    ax1.set_xlabel("Predicted Probability (p)", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12)
    ax1.set_title("Cross-Entropy Loss when y = 1", fontsize=14)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 5)
    ax1.axvline(1.0, color="green", linestyle="--", alpha=0.5, label="Perfect: p=1")
    ax1.grid(True, alpha=0.3)

    ax1.text(
        0.05, 0.95,
        "L = -log(p)\n\nAs p→0: L→∞\nAs p→1: L→0",
        transform=ax1.transAxes,
        verticalalignment="top",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )
    ax1.legend()

    # Panel 2: Loss for y=0 (negative class)
    ax2 = axes[1]
    loss_y0 = -np.log(1 - p_range)  # -log(1-p) when y=0

    ax2.plot(p_range, loss_y0, "r-", linewidth=2)
    ax2.set_xlabel("Predicted Probability (p)", fontsize=12)
    ax2.set_ylabel("Loss", fontsize=12)
    ax2.set_title("Cross-Entropy Loss when y = 0", fontsize=14)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 5)
    ax2.axvline(0.0, color="green", linestyle="--", alpha=0.5, label="Perfect: p=0")
    ax2.grid(True, alpha=0.3)

    ax2.text(
        0.65, 0.95,
        "L = -log(1-p)\n\nAs p→0: L→0\nAs p→1: L→∞",
        transform=ax2.transAxes,
        verticalalignment="top",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )
    ax2.legend()

    plt.tight_layout()
    return fig


def plot_bernoulli_mle_visual(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    figsize: tuple = (15, 10),
) -> Figure:
    """
    Visualize Bernoulli MLE process for classification.

    Parallel to plot_mle_derivation_visual() for regression.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels (0 or 1)
    y_pred_proba : np.ndarray
        Predicted probabilities P(y=1)
    figsize : tuple
        Figure size

    Returns
    -------
    fig : matplotlib Figure
        Multi-panel figure showing Bernoulli MLE connection to cross-entropy
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    y_true = np.asarray(y_true).flatten()
    y_pred_proba = np.asarray(y_pred_proba).flatten()

    # Panel 1: Individual Sample Likelihoods
    ax1 = axes[0, 0]
    # Bernoulli likelihood: p^y * (1-p)^(1-y)
    likelihoods = np.where(y_true == 1, y_pred_proba, 1 - y_pred_proba)

    # Show first 50 samples as bar plot
    n_show = min(50, len(y_true))
    indices = np.arange(n_show)
    colors = ["orange" if y == 1 else "blue" for y in y_true[:n_show]]

    ax1.bar(indices, likelihoods[:n_show], color=colors, alpha=0.7, edgecolor="black")
    ax1.set_xlabel("Sample Index", fontsize=10)
    ax1.set_ylabel("Likelihood", fontsize=10)
    ax1.set_title("1. Bernoulli Likelihood per Sample", fontsize=12)
    ax1.axhline(0.5, color="red", linestyle="--", alpha=0.5)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="orange", label="y=1: L=p"),
        Patch(facecolor="blue", label="y=0: L=1-p"),
    ]
    ax1.legend(handles=legend_elements, loc="upper right")

    # Panel 2: Log-Likelihood Contributions
    ax2 = axes[0, 1]
    log_likelihoods = np.where(
        y_true == 1,
        np.log(np.clip(y_pred_proba, 1e-10, 1)),
        np.log(np.clip(1 - y_pred_proba, 1e-10, 1)),
    )

    ax2.hist(log_likelihoods, bins=30, edgecolor="black", alpha=0.7, color="steelblue")
    ax2.axvline(np.mean(log_likelihoods), color="red", linestyle="--",
                label=f"Mean = {np.mean(log_likelihoods):.3f}")
    ax2.set_xlabel("Log-Likelihood", fontsize=10)
    ax2.set_ylabel("Frequency", fontsize=10)
    ax2.set_title("2. Log-Likelihood Contributions", fontsize=12)
    ax2.legend()

    total_ll = np.sum(log_likelihoods)
    ax2.text(
        0.05, 0.95,
        f"Total log-L = {total_ll:.2f}\nNLL = {-total_ll:.2f}",
        transform=ax2.transAxes,
        verticalalignment="top",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    # Panel 3: Cross-Entropy Loss per Sample
    ax3 = axes[1, 0]
    ce_loss = -log_likelihoods  # Cross-entropy = negative log-likelihood

    ax3.hist(ce_loss, bins=30, edgecolor="black", alpha=0.7, color="steelblue")
    ax3.axvline(np.mean(ce_loss), color="red", linestyle="--",
                label=f"Mean CE = {np.mean(ce_loss):.3f}")
    ax3.set_xlabel("Cross-Entropy Loss", fontsize=10)
    ax3.set_ylabel("Frequency", fontsize=10)
    ax3.set_title("3. Cross-Entropy = Negative Log-Likelihood", fontsize=12)
    ax3.legend()

    ax3.text(
        0.65, 0.95,
        "BCE = -[y·log(p) +\n       (1-y)·log(1-p)]",
        transform=ax3.transAxes,
        verticalalignment="top",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    # Panel 4: Calibration Curve
    ax4 = axes[1, 1]
    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)

    prob_true = []
    prob_pred = []

    for i in range(n_bins):
        mask = (y_pred_proba >= bin_edges[i]) & (y_pred_proba < bin_edges[i + 1])
        if i == n_bins - 1:
            mask = (y_pred_proba >= bin_edges[i]) & (y_pred_proba <= bin_edges[i + 1])
        if mask.sum() > 0:
            prob_true.append(y_true[mask].mean())
            prob_pred.append(y_pred_proba[mask].mean())

    ax4.plot([0, 1], [0, 1], "r--", linewidth=2, label="Perfect calibration")
    ax4.scatter(prob_pred, prob_true, s=80, color="steelblue", zorder=5)
    ax4.plot(prob_pred, prob_true, "b-", alpha=0.5)
    ax4.set_xlabel("Mean Predicted Probability", fontsize=10)
    ax4.set_ylabel("Fraction of Positives", fontsize=10)
    ax4.set_title("4. Calibration Curve (Residual Check)", fontsize=12)
    ax4.legend()
    ax4.set_xlim(-0.05, 1.05)
    ax4.set_ylim(-0.05, 1.05)
    ax4.grid(True, alpha=0.3)

    ax4.text(
        0.05, 0.95,
        "Diagonal = Bernoulli\nassumption correct",
        transform=ax4.transAxes,
        verticalalignment="top",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()
    return fig


def plot_metric_sensitivity_to_imbalance(
    imbalance_ratios: list | None = None,
    n_samples: int = 1000,
    n_repeats: int = 10,
    figsize: tuple = (15, 6),
) -> Figure:
    """
    Show how different metrics respond to class imbalance.

    Parameters
    ----------
    imbalance_ratios : list
        Different minority class proportions to test
    n_samples : int
        Samples per experiment
    n_repeats : int
        Number of repetitions for averaging
    figsize : tuple
        Figure size

    Returns
    -------
    fig : matplotlib Figure
        Line plots showing metric values vs imbalance ratio

    Notes
    -----
    Visual proof of which metrics "break" under imbalance.
    Shows why accuracy is misleading and why recall/AUC are preferred.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

    if imbalance_ratios is None:
        imbalance_ratios = [0.5, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01]

    # Storage for metrics
    metrics_data = {
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1": [],
        "auc_roc": [],
    }
    metrics_std = {k: [] for k in metrics_data}

    rng = np.random.default_rng(42)

    for ratio in imbalance_ratios:
        acc_runs, prec_runs, rec_runs, f1_runs, auc_runs = [], [], [], [], []

        for _ in range(n_repeats):
            # Generate imbalanced data
            n_pos = int(n_samples * ratio)
            n_neg = n_samples - n_pos

            # Simple 2D separable data
            X_pos = rng.normal(1.0, 1.0, (n_pos, 2))
            X_neg = rng.normal(-0.5, 1.0, (n_neg, 2))
            X = np.vstack([X_pos, X_neg])
            y = np.array([1] * n_pos + [0] * n_neg)

            # Shuffle
            perm = rng.permutation(len(y))
            X, y = X[perm], y[perm]

            # Split
            split = int(0.7 * len(y))
            X_train, X_test = X[:split], X[split:]
            y_train, y_test = y[:split], y[split:]

            # Train model
            model = LogisticRegression(random_state=42, max_iter=1000)
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_proba = model.predict_proba(X_test)[:, 1]

                acc_runs.append(accuracy_score(y_test, y_pred))
                prec_runs.append(precision_score(y_test, y_pred, zero_division=0))
                rec_runs.append(recall_score(y_test, y_pred, zero_division=0))
                f1_runs.append(f1_score(y_test, y_pred, zero_division=0))
                if len(np.unique(y_test)) > 1:
                    auc_runs.append(roc_auc_score(y_test, y_proba))
                else:
                    auc_runs.append(np.nan)
            except Exception:
                continue

        metrics_data["accuracy"].append(np.mean(acc_runs) if acc_runs else np.nan)
        metrics_data["precision"].append(np.mean(prec_runs) if prec_runs else np.nan)
        metrics_data["recall"].append(np.mean(rec_runs) if rec_runs else np.nan)
        metrics_data["f1"].append(np.mean(f1_runs) if f1_runs else np.nan)
        metrics_data["auc_roc"].append(np.nanmean(auc_runs) if auc_runs else np.nan)

        metrics_std["accuracy"].append(np.std(acc_runs) if acc_runs else 0)
        metrics_std["precision"].append(np.std(prec_runs) if prec_runs else 0)
        metrics_std["recall"].append(np.std(rec_runs) if rec_runs else 0)
        metrics_std["f1"].append(np.std(f1_runs) if f1_runs else 0)
        metrics_std["auc_roc"].append(np.nanstd(auc_runs) if auc_runs else 0)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Panel 1: All metrics
    ax1 = axes[0]
    x_positions = np.arange(len(imbalance_ratios))
    x_labels = [f"{r:.0%}" for r in imbalance_ratios]

    ax1.plot(x_positions, metrics_data["accuracy"], "b-o", label="Accuracy", linewidth=2)
    ax1.plot(x_positions, metrics_data["precision"], "g-s", label="Precision", linewidth=2)
    ax1.plot(x_positions, metrics_data["recall"], "r-^", label="Recall", linewidth=2)
    ax1.plot(x_positions, metrics_data["f1"], "m-d", label="F1", linewidth=2)
    ax1.plot(x_positions, metrics_data["auc_roc"], "k-*", label="AUC-ROC", linewidth=2, markersize=10)

    ax1.set_xlabel("Minority Class Proportion", fontsize=12)
    ax1.set_ylabel("Metric Value", fontsize=12)
    ax1.set_title("Metric Sensitivity to Class Imbalance", fontsize=14)
    ax1.set_xticks(x_positions)
    ax1.set_xticklabels(x_labels)
    ax1.legend(loc="best")
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.05)

    # Panel 2: Sensitivity interpretation
    ax2 = axes[1]
    ax2.axis("off")

    explanation = """
    METRIC SENSITIVITY TO IMBALANCE:

    ⚠️  SENSITIVE (use with caution):
    ────────────────────────────────────
    • ACCURACY = (TP + TN) / Total
      → Dominated by majority TN
      → High even with random guessing

    • PRECISION = TP / (TP + FP)
      → FP comes from majority class
      → Affected by class proportions

    ✓  NOT SENSITIVE (preferred):
    ────────────────────────────────────
    • RECALL = TP / (TP + FN)
      → Only looks at minority class
      → Independent of majority size

    • AUC-ROC
      → Normalized TPR and FPR
      → Scale-invariant

    • F1 Score
      → Harmonic mean of P and R
      → Moderately robust

    RULE OF THUMB:
    If denominator includes majority class
    → Metric is sensitive to imbalance
    """

    ax2.text(
        0.05, 0.95,
        explanation,
        transform=ax2.transAxes,
        verticalalignment="top",
        fontsize=11,
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
    )

    plt.tight_layout()
    return fig


def plot_weighted_vs_unweighted_crossentropy(
    y_true: np.ndarray,
    y_pred_proba_unweighted: np.ndarray,
    y_pred_proba_weighted: np.ndarray,
    figsize: tuple = (15, 5),
) -> Figure:
    """
    Compare unweighted vs weighted cross-entropy effects.

    Parameters
    ----------
    y_true : np.ndarray
        Binary labels
    y_pred_proba_unweighted : np.ndarray
        Probabilities from standard model
    y_pred_proba_weighted : np.ndarray
        Probabilities from weighted model
    figsize : tuple
        Figure size

    Returns
    -------
    fig : matplotlib Figure
        3 subplots comparing the two approaches
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    y_true = np.asarray(y_true).flatten()
    y_pred_proba_unweighted = np.asarray(y_pred_proba_unweighted).flatten()
    y_pred_proba_weighted = np.asarray(y_pred_proba_weighted).flatten()

    # Panel 1: Predicted Probability Distribution by Class
    ax1 = axes[0]

    # Unweighted
    ax1.hist(
        y_pred_proba_unweighted[y_true == 0],
        bins=20, alpha=0.4, label="Unweighted: Class 0",
        color="blue", density=True,
    )
    ax1.hist(
        y_pred_proba_unweighted[y_true == 1],
        bins=20, alpha=0.4, label="Unweighted: Class 1",
        color="blue", density=True, hatch="//",
    )

    # Weighted (shifted histogram)
    ax1.hist(
        y_pred_proba_weighted[y_true == 0],
        bins=20, alpha=0.4, label="Weighted: Class 0",
        color="orange", density=True,
    )
    ax1.hist(
        y_pred_proba_weighted[y_true == 1],
        bins=20, alpha=0.4, label="Weighted: Class 1",
        color="orange", density=True, hatch="\\\\",
    )

    ax1.axvline(0.5, color="red", linestyle="--", label="Threshold=0.5")
    ax1.set_xlabel("Predicted Probability", fontsize=10)
    ax1.set_ylabel("Density", fontsize=10)
    ax1.set_title("Predicted Probabilities by Class", fontsize=12)
    ax1.legend(loc="upper center", fontsize=8)

    # Panel 2: Residuals (y - p_hat) by Class
    ax2 = axes[1]

    residuals_unw = y_true - y_pred_proba_unweighted
    residuals_w = y_true - y_pred_proba_weighted

    positions = [1, 2, 4, 5]
    data = [
        residuals_unw[y_true == 0],
        residuals_unw[y_true == 1],
        residuals_w[y_true == 0],
        residuals_w[y_true == 1],
    ]
    bp = ax2.boxplot(data, positions=positions, widths=0.6, patch_artist=True)

    colors = ["lightblue", "lightsalmon", "lightgreen", "lightyellow"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)

    ax2.set_xticks([1.5, 4.5])
    ax2.set_xticklabels(["Unweighted", "Weighted"])
    ax2.axhline(0, color="red", linestyle="--", alpha=0.5)
    ax2.set_ylabel("Residual (y - p̂)", fontsize=10)
    ax2.set_title("Classification 'Residuals' by Model", fontsize=12)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="lightblue", label="Class 0"),
        Patch(facecolor="lightsalmon", label="Class 1"),
    ]
    ax2.legend(handles=legend_elements, loc="upper right")

    # Panel 3: Confusion Matrix Comparison
    ax3 = axes[2]

    # Calculate confusion matrices
    y_pred_unw = (y_pred_proba_unweighted >= 0.5).astype(int)
    y_pred_w = (y_pred_proba_weighted >= 0.5).astype(int)

    cm_unw = np.array([
        [np.sum((y_true == 0) & (y_pred_unw == 0)), np.sum((y_true == 0) & (y_pred_unw == 1))],
        [np.sum((y_true == 1) & (y_pred_unw == 0)), np.sum((y_true == 1) & (y_pred_unw == 1))],
    ])
    cm_w = np.array([
        [np.sum((y_true == 0) & (y_pred_w == 0)), np.sum((y_true == 0) & (y_pred_w == 1))],
        [np.sum((y_true == 1) & (y_pred_w == 0)), np.sum((y_true == 1) & (y_pred_w == 1))],
    ])

    # Side by side confusion matrices as text
    ax3.axis("off")

    cm_text = f"""
    CONFUSION MATRICES (threshold=0.5)

    UNWEIGHTED:                WEIGHTED:
    ┌────────────────────┐     ┌────────────────────┐
    │       Pred 0 Pred 1│     │       Pred 0 Pred 1│
    │True 0  {cm_unw[0,0]:5d} {cm_unw[0,1]:5d} │     │True 0  {cm_w[0,0]:5d} {cm_w[0,1]:5d} │
    │True 1  {cm_unw[1,0]:5d} {cm_unw[1,1]:5d} │     │True 1  {cm_w[1,0]:5d} {cm_w[1,1]:5d} │
    └────────────────────┘     └────────────────────┘

    Unweighted:                Weighted:
    • High TN (biased to       • More balanced
      majority)                • Higher TP (better
    • Low TP (misses             minority detection)
      minority)                • Some FP trade-off
    """

    ax3.text(
        0.05, 0.95,
        cm_text,
        transform=ax3.transAxes,
        verticalalignment="top",
        fontsize=10,
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
    )

    plt.tight_layout()
    return fig


# =============================================================================
# PART 1.5: REGULARIZATION AND BIAS-VARIANCE VISUALIZATIONS
# =============================================================================


def plot_regularization_path(
    X: np.ndarray,
    y: np.ndarray,
    alphas: np.ndarray | None = None,
    regularization_type: str = "ridge",
    figsize: tuple = (15, 5),
) -> Figure:
    """
    Show how coefficients and errors change with regularization strength.

    This is the key visualization for understanding regularization as shrinkage.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix (n_samples, n_features)
    y : np.ndarray
        Target values
    alphas : np.ndarray, optional
        Regularization strengths to evaluate (default: logspace from 1e-4 to 1e4)
    regularization_type : str
        'ridge' for L2 or 'lasso' for L1
    figsize : tuple
        Figure size

    Returns
    -------
    fig : matplotlib Figure
        3 subplots showing:
        1. Coefficient paths (each coefficient vs lambda)
        2. Training error vs lambda
        3. Cross-validation error vs lambda (optimal lambda)

    Notes
    -----
    Visual interpretation:
    - Large lambda: coefficients shrink to 0 (high bias, low variance)
    - Small lambda: coefficients close to OLS (low bias, high variance)
    - Optimal lambda: best trade-off (minimum CV error)
    """
    from sklearn.linear_model import Ridge, Lasso
    from sklearn.model_selection import cross_val_score

    if alphas is None:
        alphas = np.logspace(-4, 4, 50)

    n_features = X.shape[1] if X.ndim > 1 else 1
    X_2d = X.reshape(-1, 1) if X.ndim == 1 else X

    # Storage
    coef_paths = []
    train_errors = []
    cv_errors = []

    ModelClass = Ridge if regularization_type == "ridge" else Lasso

    for alpha in alphas:
        model = ModelClass(alpha=alpha, max_iter=10000)
        model.fit(X_2d, y)

        coef_paths.append(model.coef_.flatten())
        train_errors.append(np.mean((y - model.predict(X_2d)) ** 2))

        # Cross-validation error
        cv_scores = cross_val_score(
            ModelClass(alpha=alpha, max_iter=10000),
            X_2d, y, cv=5, scoring="neg_mean_squared_error"
        )
        cv_errors.append(-np.mean(cv_scores))

    coef_paths = np.array(coef_paths)
    train_errors = np.array(train_errors)
    cv_errors = np.array(cv_errors)

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    reg_name = "Ridge (L2)" if regularization_type == "ridge" else "Lasso (L1)"

    # Panel 1: Coefficient Paths
    ax1 = axes[0]
    for i in range(min(n_features, 10)):  # Plot up to 10 coefficients
        ax1.plot(alphas, coef_paths[:, i], linewidth=2, label=f"β_{i}")
    ax1.set_xscale("log")
    ax1.axhline(0, color="black", linestyle="--", alpha=0.3)
    ax1.set_xlabel("Regularization Strength (λ)", fontsize=12)
    ax1.set_ylabel("Coefficient Value", fontsize=12)
    ax1.set_title(f"{reg_name}: Coefficient Paths", fontsize=14)
    if n_features <= 5:
        ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax1.text(
        0.05, 0.95,
        "← Less regularization | More regularization →\n"
        "Large λ shrinks coefficients toward 0",
        transform=ax1.transAxes,
        verticalalignment="top",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    # Panel 2: Training Error
    ax2 = axes[1]
    ax2.plot(alphas, train_errors, "b-", linewidth=2, label="Training MSE")
    ax2.set_xscale("log")
    ax2.set_xlabel("Regularization Strength (λ)", fontsize=12)
    ax2.set_ylabel("Mean Squared Error", fontsize=12)
    ax2.set_title("Training Error vs λ", fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    ax2.text(
        0.05, 0.95,
        "Training error always increases\nwith more regularization\n(more bias)",
        transform=ax2.transAxes,
        verticalalignment="top",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    # Panel 3: Cross-Validation Error
    ax3 = axes[2]
    ax3.plot(alphas, cv_errors, "r-", linewidth=2, label="CV MSE")

    # Mark optimal lambda
    optimal_idx = np.argmin(cv_errors)
    optimal_alpha = alphas[optimal_idx]
    optimal_cv = cv_errors[optimal_idx]

    ax3.axvline(optimal_alpha, color="green", linestyle="--", alpha=0.7,
                label=f"Optimal λ = {optimal_alpha:.2e}")
    ax3.scatter([optimal_alpha], [optimal_cv], color="green", s=100, zorder=5)

    ax3.set_xscale("log")
    ax3.set_xlabel("Regularization Strength (λ)", fontsize=12)
    ax3.set_ylabel("Cross-Validation MSE", fontsize=12)
    ax3.set_title("CV Error vs λ (U-Curve)", fontsize=14)
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    ax3.text(
        0.55, 0.95,
        "CV error has optimal point:\n"
        "• Left: overfitting (high variance)\n"
        "• Right: underfitting (high bias)",
        transform=ax3.transAxes,
        verticalalignment="top",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()
    return fig


def plot_regularization_as_prior(
    figsize: tuple = (15, 5),
) -> Figure:
    """
    Visualize Gaussian (L2) and Laplace (L1) priors on coefficients.

    Shows the Bayesian interpretation: regularization = prior belief.

    Parameters
    ----------
    figsize : tuple
        Figure size

    Returns
    -------
    fig : matplotlib Figure
        3 subplots showing:
        1. Gaussian vs Laplace prior PDFs
        2. Log-prior (penalty) comparison
        3. Effect on coefficient estimation (2D contour)

    Notes
    -----
    Visual proof of:
    - Ridge = Gaussian prior -> soft shrinkage (all coefficients small)
    - Lasso = Laplace prior -> sparse shrinkage (some coefficients exactly 0)
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Prior parameters (equivalent strength)
    sigma = 1.0  # Gaussian prior std
    b = sigma / np.sqrt(2)  # Laplace scale for same variance

    x_range = np.linspace(-4, 4, 200)

    # Panel 1: Prior PDFs
    ax1 = axes[0]

    gaussian_pdf = stats.norm.pdf(x_range, 0, sigma)
    laplace_pdf = stats.laplace.pdf(x_range, 0, b)

    ax1.plot(x_range, gaussian_pdf, "b-", linewidth=2, label=f"Gaussian (σ={sigma})")
    ax1.plot(x_range, laplace_pdf, "r-", linewidth=2, label=f"Laplace (b={b:.2f})")
    ax1.fill_between(x_range, gaussian_pdf, alpha=0.2, color="blue")
    ax1.fill_between(x_range, laplace_pdf, alpha=0.2, color="red")

    ax1.set_xlabel("Coefficient Value (β)", fontsize=12)
    ax1.set_ylabel("Prior Probability Density", fontsize=12)
    ax1.set_title("Prior Distributions on Coefficients", fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax1.text(
        0.05, 0.95,
        "Laplace has heavier tails\n→ allows some large coefficients\n→ but sharp peak at 0",
        transform=ax1.transAxes,
        verticalalignment="top",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    # Panel 2: Log-prior (Penalty)
    ax2 = axes[1]

    # -log(prior) = penalty
    gaussian_penalty = x_range ** 2 / (2 * sigma ** 2)  # L2 penalty
    laplace_penalty = np.abs(x_range) / b  # L1 penalty

    ax2.plot(x_range, gaussian_penalty, "b-", linewidth=2, label="L2: β² (Ridge)")
    ax2.plot(x_range, laplace_penalty, "r-", linewidth=2, label="L1: |β| (Lasso)")

    ax2.set_xlabel("Coefficient Value (β)", fontsize=12)
    ax2.set_ylabel("-log(prior) = Penalty", fontsize=12)
    ax2.set_title("Penalty Functions", fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 8)

    ax2.text(
        0.55, 0.95,
        "L2 (Ridge):\n  ∂penalty/∂β = 2β (shrinks)\n\n"
        "L1 (Lasso):\n  ∂penalty/∂β = sign(β)\n  (can reach exactly 0)",
        transform=ax2.transAxes,
        verticalalignment="top",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    # Panel 3: 2D Contour Showing Constraint Regions
    ax3 = axes[2]

    beta1 = np.linspace(-2, 2, 200)
    beta2 = np.linspace(-2, 2, 200)
    B1, B2 = np.meshgrid(beta1, beta2)

    # L2 constraint: β₁² + β₂² ≤ t²
    L2_constraint = B1 ** 2 + B2 ** 2

    # L1 constraint: |β₁| + |β₂| ≤ t
    L1_constraint = np.abs(B1) + np.abs(B2)

    # Plot constraint regions
    ax3.contour(B1, B2, L2_constraint, levels=[1], colors="blue", linewidths=2)
    ax3.contour(B1, B2, L1_constraint, levels=[1], colors="red", linewidths=2)

    # Add OLS solution (example)
    ols_beta = [1.5, 0.3]
    ax3.scatter(*ols_beta, color="black", s=100, zorder=5, label="OLS solution")

    # Draw loss contours (ellipses centered on OLS)
    loss_contours = (B1 - ols_beta[0]) ** 2 + (B2 - ols_beta[1]) ** 2
    ax3.contour(B1, B2, loss_contours, levels=[0.5, 1, 2, 3], colors="gray",
                linestyles="--", alpha=0.5)

    ax3.set_xlabel("β₁", fontsize=12)
    ax3.set_ylabel("β₂", fontsize=12)
    ax3.set_title("Constraint Regions", fontsize=14)
    ax3.set_xlim(-2, 2)
    ax3.set_ylim(-2, 2)
    ax3.set_aspect("equal")
    ax3.axhline(0, color="black", linewidth=0.5)
    ax3.axvline(0, color="black", linewidth=0.5)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="blue", linewidth=2, label="L2 (circle)"),
        Line2D([0], [0], color="red", linewidth=2, label="L1 (diamond)"),
        Line2D([0], [0], color="gray", linestyle="--", label="Loss contours"),
    ]
    ax3.legend(handles=legend_elements, loc="lower right")

    ax3.text(
        0.05, 0.95,
        "L1 has corners on axes\n→ solution hits corner\n→ sparse (β=0)",
        transform=ax3.transAxes,
        verticalalignment="top",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()
    return fig


def plot_residuals_with_without_regularization(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float = 1.0,
    figsize: tuple = (15, 5),
) -> Figure:
    """
    Compare residual distributions with and without regularization.

    Shows how regularization affects residual patterns.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Target values
    alpha : float
        Regularization strength for Ridge
    figsize : tuple
        Figure size

    Returns
    -------
    fig : matplotlib Figure
        3 subplots comparing residuals

    Notes
    -----
    Regularized model:
    - May have larger training residuals (more bias)
    - But residuals may be more stable (less variance)
    - Residual distribution often more symmetric
    """
    from sklearn.linear_model import LinearRegression, Ridge

    X_2d = X.reshape(-1, 1) if X.ndim == 1 else X

    # Fit models
    ols = LinearRegression()
    ridge = Ridge(alpha=alpha)

    ols.fit(X_2d, y)
    ridge.fit(X_2d, y)

    y_pred_ols = ols.predict(X_2d)
    y_pred_ridge = ridge.predict(X_2d)

    residuals_ols = y - y_pred_ols
    residuals_ridge = y - y_pred_ridge

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Panel 1: Residual Histograms
    ax1 = axes[0]
    ax1.hist(residuals_ols, bins=30, alpha=0.6, label="OLS", color="blue", density=True)
    ax1.hist(residuals_ridge, bins=30, alpha=0.6, label=f"Ridge (λ={alpha})",
             color="orange", density=True)
    ax1.axvline(0, color="red", linestyle="--")
    ax1.set_xlabel("Residual", fontsize=12)
    ax1.set_ylabel("Density", fontsize=12)
    ax1.set_title("Residual Distributions", fontsize=14)
    ax1.legend()

    # Add statistics
    ax1.text(
        0.05, 0.95,
        f"OLS:\n  Mean={np.mean(residuals_ols):.3f}\n  Std={np.std(residuals_ols):.3f}\n\n"
        f"Ridge:\n  Mean={np.mean(residuals_ridge):.3f}\n  Std={np.std(residuals_ridge):.3f}",
        transform=ax1.transAxes,
        verticalalignment="top",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    # Panel 2: Residuals vs Fitted
    ax2 = axes[1]
    ax2.scatter(y_pred_ols, residuals_ols, alpha=0.5, s=20, label="OLS", color="blue")
    ax2.scatter(y_pred_ridge, residuals_ridge, alpha=0.5, s=20, label="Ridge", color="orange")
    ax2.axhline(0, color="red", linestyle="--")
    ax2.set_xlabel("Fitted Values", fontsize=12)
    ax2.set_ylabel("Residual", fontsize=12)
    ax2.set_title("Residuals vs Fitted", fontsize=14)
    ax2.legend()

    # Panel 3: Q-Q Plots
    ax3 = axes[2]

    # OLS Q-Q
    (osm_ols, osr_ols), (slope_ols, intercept_ols, r_ols) = stats.probplot(
        residuals_ols, dist="norm"
    )
    ax3.scatter(osm_ols, osr_ols, alpha=0.5, s=20, label="OLS", color="blue")

    # Ridge Q-Q
    (osm_ridge, osr_ridge), (slope_ridge, intercept_ridge, r_ridge) = stats.probplot(
        residuals_ridge, dist="norm"
    )
    ax3.scatter(osm_ridge, osr_ridge, alpha=0.5, s=20, label="Ridge", color="orange")

    # Reference line
    x_range = np.linspace(-3, 3, 100)
    ax3.plot(x_range, x_range * np.std(residuals_ols) + np.mean(residuals_ols),
             "b--", alpha=0.5)
    ax3.plot(x_range, x_range * np.std(residuals_ridge) + np.mean(residuals_ridge),
             "r--", alpha=0.5)

    ax3.set_xlabel("Theoretical Quantiles", fontsize=12)
    ax3.set_ylabel("Sample Quantiles", fontsize=12)
    ax3.set_title("Q-Q Plots (Normality Check)", fontsize=14)
    ax3.legend()

    plt.tight_layout()
    return fig


def plot_bias_variance_with_model_complexity(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    max_degree: int = 15,
    figsize: tuple = (15, 5),
) -> Figure:
    """
    Classic U-curve: error vs model complexity.

    This is THE fundamental bias-variance visualization.

    Parameters
    ----------
    X_train, y_train : np.ndarray
        Training data
    X_test, y_test : np.ndarray
        Test data
    max_degree : int
        Maximum polynomial degree to try
    figsize : tuple
        Figure size

    Returns
    -------
    fig : matplotlib Figure
        3 subplots showing:
        1. Training and test error vs complexity
        2. Bias² and Variance decomposition (estimated)
        3. Model fits at different complexities

    Notes
    -----
    Visual proof of:
    - Low complexity: high bias (underfitting)
    - High complexity: high variance (overfitting)
    - Optimal complexity: minimum test error
    """
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import make_pipeline

    X_train_1d = X_train.flatten() if X_train.ndim > 1 else X_train
    X_test_1d = X_test.flatten() if X_test.ndim > 1 else X_test

    degrees = list(range(1, max_degree + 1))
    train_errors = []
    test_errors = []

    # Store predictions for plotting
    best_predictions = {}

    for degree in degrees:
        model = make_pipeline(
            PolynomialFeatures(degree, include_bias=False),
            LinearRegression()
        )
        model.fit(X_train_1d.reshape(-1, 1), y_train)

        y_train_pred = model.predict(X_train_1d.reshape(-1, 1))
        y_test_pred = model.predict(X_test_1d.reshape(-1, 1))

        train_errors.append(np.mean((y_train - y_train_pred) ** 2))
        test_errors.append(np.mean((y_test - y_test_pred) ** 2))

        # Store predictions for key degrees
        if degree in [1, 3, max_degree]:
            x_plot = np.linspace(X_train_1d.min(), X_train_1d.max(), 100)
            best_predictions[degree] = {
                "x": x_plot,
                "y": model.predict(x_plot.reshape(-1, 1)),
            }

    train_errors = np.array(train_errors)
    test_errors = np.array(test_errors)

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Panel 1: Training and Test Error
    ax1 = axes[0]
    ax1.plot(degrees, train_errors, "b-o", label="Training Error", linewidth=2)
    ax1.plot(degrees, test_errors, "r-o", label="Test Error", linewidth=2)

    optimal_degree = degrees[np.argmin(test_errors)]
    ax1.axvline(optimal_degree, color="green", linestyle="--", alpha=0.7,
                label=f"Optimal: degree={optimal_degree}")

    ax1.set_xlabel("Model Complexity (Polynomial Degree)", fontsize=12)
    ax1.set_ylabel("Mean Squared Error", fontsize=12)
    ax1.set_title("The Bias-Variance Trade-off", fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Mark regions
    ax1.fill_betweenx(
        [0, ax1.get_ylim()[1]], 0, optimal_degree,
        alpha=0.1, color="blue", label="Underfitting"
    )
    ax1.fill_betweenx(
        [0, ax1.get_ylim()[1]], optimal_degree, max_degree + 1,
        alpha=0.1, color="red", label="Overfitting"
    )

    # Panel 2: Bias² and Variance (conceptual)
    ax2 = axes[1]

    # Estimate bias² as training error (lower bound)
    # Estimate variance as (test - train) error
    bias_squared = train_errors
    variance = np.maximum(test_errors - train_errors, 0)
    total = bias_squared + variance

    ax2.plot(degrees, bias_squared, "b-s", label="Bias² (≈ train error)", linewidth=2)
    ax2.plot(degrees, variance, "r-^", label="Variance (≈ test-train)", linewidth=2)
    ax2.plot(degrees, total, "g-o", label="Total = Bias² + Variance", linewidth=2)
    ax2.axvline(optimal_degree, color="green", linestyle="--", alpha=0.7)

    ax2.set_xlabel("Model Complexity", fontsize=12)
    ax2.set_ylabel("Error Component", fontsize=12)
    ax2.set_title("Bias-Variance Decomposition", fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax2.text(
        0.05, 0.95,
        "As complexity ↑:\n• Bias² ↓ (better fit)\n• Variance ↑ (overfit)",
        transform=ax2.transAxes,
        verticalalignment="top",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    # Panel 3: Model Fits at Different Complexities
    ax3 = axes[2]
    ax3.scatter(X_train_1d, y_train, alpha=0.3, s=20, color="steelblue", label="Training data")

    colors = {"1": "blue", "3": "green", str(max_degree): "red"}
    labels = {"1": "Degree 1 (underfit)", "3": f"Degree 3", str(max_degree): f"Degree {max_degree} (overfit)"}

    for degree, preds in best_predictions.items():
        ax3.plot(preds["x"], preds["y"], linewidth=2,
                 color=colors.get(str(degree), "gray"),
                 label=labels.get(str(degree), f"Degree {degree}"))

    ax3.set_xlabel("X", fontsize=12)
    ax3.set_ylabel("y", fontsize=12)
    ax3.set_title("Model Fits at Different Complexities", fontsize=14)
    ax3.legend()

    plt.tight_layout()
    return fig


def plot_learning_curves(
    X: np.ndarray,
    y: np.ndarray,
    model=None,
    train_sizes: np.ndarray | None = None,
    cv: int = 5,
    figsize: tuple = (12, 5),
) -> Figure:
    """
    Plot learning curves: error vs training set size.

    Shows how model performance improves with more data.

    Parameters
    ----------
    X : np.ndarray
        Features
    y : np.ndarray
        Targets
    model : estimator, optional
        Model to evaluate (default: LinearRegression)
    train_sizes : np.ndarray, optional
        Fractions of training data to use
    cv : int
        Number of cross-validation folds
    figsize : tuple
        Figure size

    Returns
    -------
    fig : matplotlib Figure
        2 subplots showing:
        1. Training and validation error vs training size
        2. Gap between training and validation error

    Notes
    -----
    Interpretation:
    - High bias: both curves plateau at high error
    - High variance: large gap between train and validation
    - More data helps: if validation still improving
    """
    from sklearn.model_selection import learning_curve
    from sklearn.linear_model import LinearRegression

    if model is None:
        model = LinearRegression()

    if train_sizes is None:
        train_sizes = np.linspace(0.1, 1.0, 10)

    X_2d = X.reshape(-1, 1) if X.ndim == 1 else X

    train_sizes_abs, train_scores, val_scores = learning_curve(
        model, X_2d, y,
        train_sizes=train_sizes,
        cv=cv,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
    )

    # Convert to positive MSE
    train_scores = -train_scores
    val_scores = -val_scores

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Panel 1: Learning Curves
    ax1 = axes[0]

    ax1.plot(train_sizes_abs, train_mean, "b-o", label="Training Error", linewidth=2)
    ax1.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std,
                     alpha=0.2, color="blue")

    ax1.plot(train_sizes_abs, val_mean, "r-o", label="Validation Error", linewidth=2)
    ax1.fill_between(train_sizes_abs, val_mean - val_std, val_mean + val_std,
                     alpha=0.2, color="red")

    ax1.set_xlabel("Training Set Size", fontsize=12)
    ax1.set_ylabel("Mean Squared Error", fontsize=12)
    ax1.set_title("Learning Curves", fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Panel 2: Generalization Gap
    ax2 = axes[1]

    gap = val_mean - train_mean
    ax2.plot(train_sizes_abs, gap, "g-o", linewidth=2)
    ax2.fill_between(train_sizes_abs, 0, gap, alpha=0.3, color="green")

    ax2.axhline(0, color="red", linestyle="--", alpha=0.5)
    ax2.set_xlabel("Training Set Size", fontsize=12)
    ax2.set_ylabel("Validation - Training Error", fontsize=12)
    ax2.set_title("Generalization Gap", fontsize=14)
    ax2.grid(True, alpha=0.3)

    ax2.text(
        0.05, 0.95,
        "Large gap = High variance\n(model memorizes training)\n\n"
        "Small gap = Low variance\n(model generalizes well)\n\n"
        "Gap decreasing = More data helps",
        transform=ax2.transAxes,
        verticalalignment="top",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()
    return fig


def plot_outlier_comparison(
    X: np.ndarray,
    y: np.ndarray,
    is_outlier: np.ndarray,
    models: dict,
    feature_idx: int = 0,
    figsize: tuple = (15, 5),
) -> Figure:
    """
    Compare model fits on data with outliers highlighted.

    Parameters
    ----------
    X : np.ndarray
        Features
    y : np.ndarray
        Targets
    is_outlier : np.ndarray
        Boolean mask for outliers
    models : dict
        Dictionary of trained models
    feature_idx : int
        Which feature to plot
    figsize : tuple
        Figure size

    Returns
    -------
    fig : matplotlib Figure
    """
    x_feature = X[:, feature_idx] if X.ndim > 1 else X.flatten()
    sort_idx = np.argsort(x_feature)

    n_models = len(models)
    fig, axes = plt.subplots(1, n_models, figsize=figsize)

    if n_models == 1:
        axes = [axes]

    for ax, (name, results) in zip(axes, models.items()):
        # Plot clean points
        ax.scatter(x_feature[~is_outlier], y[~is_outlier],
                   alpha=0.5, s=30, color="steelblue", label="Clean data")
        # Plot outliers
        ax.scatter(x_feature[is_outlier], y[is_outlier],
                   alpha=0.8, s=50, color="red", marker="x", label="Outliers")

        # Plot model fit
        y_pred = results["train_preds"] if "train_preds" in results else results["test_preds"]
        ax.plot(x_feature[sort_idx], y_pred[sort_idx],
                color="green", linewidth=2, label="Model fit")

        ax.set_xlabel(f"Feature {feature_idx}")
        ax.set_ylabel("Target")
        ax.set_title(f"{name} Loss")
        ax.legend()

    plt.tight_layout()
    return fig
