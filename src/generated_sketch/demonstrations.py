"""
Demonstrations module for MLE and Loss Functions presentation.

This module provides high-level demonstration functions that combine
data generation, model training, and visualization for each section
of the presentation.
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

from data_generation import (
    generate_imbalanced_classification_data,
    generate_nonlinear_data,
    generate_regression_data_with_gaussian_noise,
    generate_regression_data_with_laplace_noise,
    generate_regression_data_with_outliers,
    load_real_dataset,
)
from metrics import (
    analyze_residual_distribution,
    compare_model_metrics,
    print_residual_analysis,
)
from models import (
    train_classification_models,
    train_models_with_different_losses,
)
from theory import (
    compare_classification_distributions,
    compare_estimators_with_outliers,
    demonstrate_mean_minimizes_mse,
    demonstrate_median_minimizes_mae,
    demonstrate_probability_calibration,
    derive_crossentropy_from_bernoulli_mle,
    derive_mae_from_laplace_mle,
    derive_mse_from_gaussian_mle,
    explain_cross_entropy_properties,
    simulate_bias_variance_tradeoff,
    why_bernoulli_distribution,
)
from utils import print_key_insight, print_section_header
from visualization import (
    plot_bernoulli_mle_visual,
    plot_bias_variance_decomposition,
    plot_classification_residuals,
    plot_cross_entropy_loss_surface,
    plot_imbalance_effect,
    plot_loss_function_comparison,
    plot_loss_function_shapes,
    plot_metric_sensitivity_to_imbalance,
    plot_mle_derivation_visual,
    plot_outlier_comparison,
    plot_residuals_diagnostic,
    plot_three_distributions_concept,
    plot_weighted_vs_unweighted_crossentropy,
)


def demonstration_1_three_distributions(
    n_samples: int = 500,
    noise_std: float = 2.0,
    random_state: int = 42,
) -> None:
    """
    SECTION 1 DEMO: The three distributions concept.

    Creates synthetic data, fits model, shows three distributions:
    1. Input distribution
    2. Learned function
    3. Residual distribution

    Parameters
    ----------
    n_samples : int
        Number of samples
    noise_std : float
        Noise level

    Notes
    -----
    This is THE KEY concept.
    Spend time here to make sure audience gets it.
    """
    print_section_header(1, "The Three Distributions", "Understanding what models actually learn")

    # Generate data with Gaussian noise
    X, y, true_signal = generate_regression_data_with_gaussian_noise(
        n_samples=n_samples,
        n_features=5,
        noise_std=noise_std,
        random_state=random_state,
    )

    # Fit a simple linear model
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)

    # Create the key visualization
    fig = plot_three_distributions_concept(X, y, y_pred, feature_idx=0)
    plt.show()

    # Print explanation
    print_key_insight("""
KEY INSIGHT: There are THREE distributions in ML, not one!

1. INPUT DISTRIBUTION (left panel)
   - The distribution of your features X
   - Often complex, multivariate
   - The model doesn't learn this!

2. LEARNED FUNCTION (middle panel)
   - Model learns f: X -> y
   - This is a FUNCTION, not a distribution!
   - Maps inputs to predictions

3. RESIDUAL DISTRIBUTION (right panel)
   - The distribution of y - f(X)
   - THIS is what loss functions assume!
   - Gaussian residuals -> MSE
   - Laplace residuals -> MAE

Most confusion comes from conflating these three!
""")

    # Show residual analysis
    residuals = y - y_pred
    analysis = analyze_residual_distribution(residuals, "gaussian")
    print_residual_analysis(analysis)


def demonstration_2_mle_to_mse(
    n_samples: int = 300,
    random_state: int = 42,
) -> None:
    """
    SECTION 2 DEMO: Show MLE with Gaussian assumption leads to MSE.

    Notes
    -----
    Walk through:
    1. Write down Gaussian likelihood
    2. Take log
    3. Negate
    4. Show it equals MSE (up to constant)
    """
    print_section_header(2, "From MLE to MSE", "Why Gaussian assumption leads to Mean Squared Error")

    # Show mathematical derivation
    derivation = derive_mse_from_gaussian_mle(show_steps=True)
    print(derivation)

    # Generate data and fit
    X, y, true_signal = generate_regression_data_with_gaussian_noise(
        n_samples=n_samples,
        noise_std=1.5,
        random_state=random_state,
    )
    model = LinearRegression().fit(X, y)
    y_pred = model.predict(X)

    # Show MLE visualization
    fig = plot_mle_derivation_visual(y, y_pred, "gaussian")
    plt.show()

    # Demonstrate mean minimizes MSE
    print("\n--- Empirical Proof: Mean Minimizes MSE ---")
    residuals = y - y_pred
    optimal, mse_values, fig = demonstrate_mean_minimizes_mse(residuals)
    plt.show()

    print_key_insight("""
CONNECTION:
    Gaussian noise assumption
         |
         v
    Maximum Likelihood Estimation
         |
         v
    Negative Log-Likelihood
         |
         v
    Mean Squared Error (MSE)

Therefore: MSE optimizes for MEAN because mean is MLE for Gaussian!
""")


def demonstration_3_mle_to_mae(
    n_samples: int = 300,
    random_state: int = 42,
) -> None:
    """
    SECTION 2 DEMO: Show MLE with Laplace assumption leads to MAE.

    Notes
    -----
    Parallel to demo 2, but with Laplace distribution.
    """
    print_section_header(2, "From MLE to MAE", "Why Laplace assumption leads to Mean Absolute Error")

    # Show mathematical derivation
    derivation = derive_mae_from_laplace_mle(show_steps=True)
    print(derivation)

    # Generate data with Laplace noise
    X, y, true_signal = generate_regression_data_with_laplace_noise(
        n_samples=n_samples,
        noise_scale=1.0,
        random_state=random_state,
    )

    # Fit model
    model = LinearRegression().fit(X, y)
    y_pred = model.predict(X)

    # Show MLE visualization
    fig = plot_mle_derivation_visual(y, y_pred, "laplace")
    plt.show()

    # Demonstrate median minimizes MAE
    print("\n--- Empirical Proof: Median Minimizes MAE ---")
    residuals = y - y_pred
    optimal, mae_values, fig = demonstrate_median_minimizes_mae(residuals)
    plt.show()

    print_key_insight("""
CONNECTION:
    Laplace noise assumption
         |
         v
    Maximum Likelihood Estimation
         |
         v
    Negative Log-Likelihood
         |
         v
    Mean Absolute Error (MAE)

Therefore: MAE optimizes for MEDIAN because median is MLE for Laplace!

WHY IS THIS IMPORTANT?
- Laplace has heavier tails than Gaussian
- Median is robust to outliers
- This explains MAE's robustness!
""")


def demonstration_4_mse_vs_mae_outliers(
    n_samples: int = 500,
    outlier_fraction: float = 0.1,
    random_state: int = 42,
) -> None:
    """
    SECTION 3 DEMO: Compare MSE vs MAE with outliers.

    This is the MONEY SHOT - shows why loss function choice matters.

    Parameters
    ----------
    n_samples : int
        Sample size
    outlier_fraction : float
        Fraction of samples that are outliers
    """
    print_section_header(3, "MSE vs MAE with Outliers", "When loss function choice really matters")

    # Generate data with outliers
    X, y, true_signal, is_outlier = generate_regression_data_with_outliers(
        n_samples=n_samples,
        outlier_fraction=outlier_fraction,
        outlier_magnitude=10.0,
        random_state=random_state,
    )

    print(f"Generated {n_samples} samples with {is_outlier.sum()} outliers ({100*outlier_fraction:.0f}%)")

    # Split data
    X_train, X_test, y_train, y_test, is_out_train, is_out_test = train_test_split(
        X, y, is_outlier, test_size=0.2, random_state=random_state
    )

    # Train models with different losses
    results = train_models_with_different_losses(X_train, y_train, X_test, y_test)

    # Show loss function shapes first
    print("\n--- Loss Function Shapes ---")
    fig = plot_loss_function_shapes()
    plt.show()

    # Show how different losses fit the data
    print("\n--- Model Fits with Outliers Highlighted ---")
    fig = plot_outlier_comparison(X_train, y_train, is_out_train, results, feature_idx=0)
    plt.show()

    # Compare model performance
    print("\n--- Model Comparison ---")
    fig = plot_loss_function_comparison(X_train, y_train, X_test, y_test, results)
    plt.show()

    # Show metrics table
    comparison_df = compare_model_metrics(results, metric_type="regression")
    print("\n--- Test Set Metrics ---")
    print(comparison_df.to_string())

    print_key_insight("""
OBSERVATION:
- MSE model: Gets pulled toward outliers (high variance)
- MAE model: Ignores outliers, fits majority well (robust)
- Huber: Compromise between the two

WHICH IS BETTER? Depends on your goal!
- If outliers are measurement errors -> MAE
- If all data points matter equally -> MSE
- If unsure -> Huber (hybrid)

The residual distributions tell the story:
- MSE: Few extreme outliers (model chases them)
- MAE: Many moderate "outliers" (model ignores them)
""")


def demonstration_5_model_misspecification(
    n_samples: int = 300,
    random_state: int = 42,
) -> None:
    """
    SECTION 3 DEMO: Show bias from wrong model architecture.

    Fit linear model to nonlinear data.

    Notes
    -----
    Key message: Loss function can't fix wrong architecture.
    - Linear model on quadratic data -> parabolic residuals
    - This is BIAS (not variance or noise)
    """
    print_section_header(3, "Model Misspecification", "When residuals reveal model bias")

    # Generate nonlinear data
    X, y, true_signal = generate_nonlinear_data(
        n_samples=n_samples,
        noise_std=0.5,
        function_type="quadratic",
        random_state=random_state,
    )

    # Fit WRONG model (linear)
    model_linear = LinearRegression().fit(X, y)
    y_pred_linear = model_linear.predict(X)

    # Fit CORRECT model (polynomial)
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X)
    model_poly = LinearRegression().fit(X_poly, y)
    y_pred_poly = model_poly.predict(X_poly)

    # Compare residuals
    print("\n--- Linear Model (WRONG) ---")
    fig = plot_residuals_diagnostic(y, y_pred_linear, "Linear Model on Quadratic Data")
    plt.show()

    print("\n--- Polynomial Model (CORRECT) ---")
    fig = plot_residuals_diagnostic(y, y_pred_poly, "Polynomial Model on Quadratic Data")
    plt.show()

    # Show bias-variance decomposition
    print("\n--- Bias-Variance Decomposition ---")
    fig = plot_bias_variance_decomposition(X, y, true_signal, y_pred_linear, feature_idx=0)
    plt.suptitle("Linear Model: Decomposition of Residuals", fontsize=14)
    plt.show()

    print_key_insight("""
BIAS IN RESIDUALS:

Linear model on quadratic data:
- Clear parabolic pattern in residuals
- This is SYSTEMATIC ERROR (bias)
- Loss function doesn't matter here!
- The architecture is fundamentally wrong

Polynomial model:
- Random residuals (no pattern)
- Model captures true relationship
- Only irreducible noise remains

TAKEAWAY:
If residuals have patterns, your model is WRONG.
No loss function can fix model misspecification.
First get the architecture right, then choose loss.
""")


def demonstration_6_real_world_example(
    dataset_name: str = "california_housing",
    random_state: int = 42,
) -> None:
    """
    SECTION 3 DEMO: Apply to real dataset.

    Parameters
    ----------
    dataset_name : str
        Which real dataset to use

    Notes
    -----
    Shows that synthetic examples generalize to real problems.
    """
    print_section_header(3, "Real World Example", f"Applying to {dataset_name} dataset")

    # Load data
    X, y, feature_names = load_real_dataset(dataset_name)
    print(f"Loaded {dataset_name}: {X.shape[0]} samples, {X.shape[1]} features")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    # Train models with different losses
    print("\nTraining models with different loss functions...")
    results = train_models_with_different_losses(X_train, y_train, X_test, y_test)

    # Try XGBoost if available
    try:
        from models import train_xgboost_models
        xgb_results = train_xgboost_models(X_train, y_train, X_test, y_test)
        results.update(xgb_results)
        print("Added XGBoost models")
    except ImportError:
        print("XGBoost not available, skipping")

    # Compare metrics
    print("\n--- Model Comparison ---")
    comparison_df = compare_model_metrics(results, metric_type="regression")
    print(comparison_df.to_string())

    # Show residual diagnostics for best model
    best_model_name = comparison_df["r2"].idxmax()
    print(f"\n--- Residual Diagnostics for Best Model ({best_model_name}) ---")
    fig = plot_residuals_diagnostic(
        y_test,
        results[best_model_name]["test_preds"],
        title=f"{dataset_name}: {best_model_name} Residuals",
    )
    plt.show()

    print_key_insight(f"""
REAL WORLD APPLICATION:

Dataset: {dataset_name}
Best model by R^2: {best_model_name}

Key observations:
1. Different loss functions give different results
2. Check residual diagnostics for patterns
3. Real data often has outliers -> consider robust losses
4. No single loss function is universally best
""")


def demonstration_7_classification_imbalance(
    n_samples: int = 1000,
    imbalance_ratio: float = 0.05,
    random_state: int = 42,
) -> None:
    """
    SECTION 4 DEMO: Imbalanced classification.

    Parameters
    ----------
    imbalance_ratio : float
        Fraction in minority class

    Notes
    -----
    Shows:
    1. Standard MLE biased toward majority
    2. Accuracy is misleading
    3. Recall vs precision tradeoff
    4. Weighted loss fixes it
    """
    print_section_header(4, "Classification with Imbalanced Data", "How MLE fails with class imbalance")

    # Generate imbalanced data
    X, y = generate_imbalanced_classification_data(
        n_samples=n_samples,
        imbalance_ratio=imbalance_ratio,
        class_separation=1.5,
        random_state=random_state,
    )

    n_minority = int(y.sum())
    n_majority = len(y) - n_minority
    print(f"Class 0 (majority): {n_majority} samples ({100*n_majority/len(y):.1f}%)")
    print(f"Class 1 (minority): {n_minority} samples ({100*n_minority/len(y):.1f}%)")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    # Train with and without weighting
    results = train_classification_models(X_train, y_train, X_test, y_test)

    # Compare metrics
    print("\n--- Model Comparison ---")
    comparison_df = compare_model_metrics(results, metric_type="classification")
    print(comparison_df.to_string())

    # Visualize classification residuals
    print("\n--- Classification 'Residuals' (Unweighted) ---")
    fig = plot_classification_residuals(
        y_test,
        results["unweighted"]["test_proba"],
        title="Unweighted Model",
    )
    plt.show()

    print("\n--- Classification 'Residuals' (Weighted) ---")
    fig = plot_classification_residuals(
        y_test,
        results["weighted"]["test_proba"],
        title="Weighted Model",
    )
    plt.show()

    # Show imbalance effect
    print("\n--- Imbalance Effect Analysis ---")
    fig = plot_imbalance_effect(y_test, results["unweighted"]["test_proba"])
    plt.suptitle("Unweighted Model: Imbalance Effects", fontsize=14)
    plt.show()

    unw = results["unweighted"]["test_metrics"]
    w = results["weighted"]["test_metrics"]

    print_key_insight(f"""
IMBALANCE EFFECT:

Unweighted model:
- Accuracy: {unw['accuracy']:.3f} (looks great, but misleading!)
- Recall:   {unw['recall']:.3f} (terrible for minority class)
- Precision:{unw['precision']:.3f}

Weighted model:
- Accuracy: {w['accuracy']:.3f} (lower, but more honest)
- Recall:   {w['recall']:.3f} (much better!)
- Precision:{w['precision']:.3f}

WHY THIS HAPPENS:
- Standard MLE optimizes AVERAGE loss
- Majority class dominates the average
- Model learns to always predict majority -> high accuracy!
- But completely fails on minority class

SOLUTION:
- Weight loss function to equalize class contributions
- Or use appropriate metrics (F1, AUC-PR)
- Don't trust accuracy alone!
""")


def demonstration_8_bias_variance_residuals(
    model_complexities: list[int] | None = None,
    random_state: int = 42,
) -> None:
    """
    BONUS DEMO: Connect bias-variance to residuals.

    Parameters
    ----------
    model_complexities : list[int]
        Polynomial degrees to try

    Notes
    -----
    Only possible with synthetic data where true function is known.
    """
    print_section_header(5, "Bias-Variance in Residuals", "Understanding what residuals contain")

    if model_complexities is None:
        model_complexities = [1, 2, 3, 5, 7, 10, 15]

    # Define true function
    def true_function(x):
        return np.sin(x) + 0.5 * x

    # Simulate bias-variance tradeoff
    print("Simulating bias-variance tradeoff across model complexities...")
    results, fig = simulate_bias_variance_tradeoff(
        true_function=true_function,
        noise_std=0.5,
        model_complexities=model_complexities,
        n_datasets=50,
        n_samples_per_dataset=50,
    )
    plt.show()

    # Show decomposition for one complexity
    print("\n--- Residual Decomposition for Linear Model (Degree 1) ---")
    X, y, true_signal = generate_nonlinear_data(
        n_samples=200,
        noise_std=0.5,
        function_type="sine",
        random_state=random_state,
    )

    # Fit linear model (high bias)
    model = LinearRegression().fit(X, y)
    y_pred = model.predict(X)

    fig = plot_bias_variance_decomposition(X, y, true_signal, y_pred, feature_idx=0)
    plt.suptitle("Linear Model on Sinusoidal Data: High Bias", fontsize=14)
    plt.show()

    print_key_insight("""
RESIDUAL DECOMPOSITION:

Observed Residuals = Irreducible Noise + Model Bias + Model Variance

Where:
- Irreducible Noise: y_true - f_true (can't be reduced)
- Model Bias: f_true - E[f_model] (systematic error)
- Model Variance: E[f_model] - f_model (varies with training data)

IMPLICATIONS:
- Low complexity: High bias, low variance
- High complexity: Low bias, high variance
- Sweet spot: Balanced bias-variance

Loss function affects this tradeoff!
- MSE sensitive to variance (outliers)
- MAE more robust (less variance sensitivity)
""")


# =============================================================================
# PART 2: CLASSIFICATION DEMONSTRATIONS
# =============================================================================


def demonstration_9_bernoulli_to_crossentropy(
    n_samples: int = 500,
    random_state: int = 42,
) -> None:
    """
    SECTION 2.2: Derive cross-entropy from Bernoulli MLE.

    Parallel to demonstration_2_mle_to_mse() for regression.

    Parameters
    ----------
    n_samples : int
        Number of samples
    random_state : int
        Random seed

    Notes
    -----
    This is THE core concept for classification.
    Makes explicit: Cross-entropy is not arbitrary, it's MLE under Bernoulli.
    Students should walk away understanding this connection.
    """
    from sklearn.linear_model import LogisticRegression

    from theory import (
        derive_crossentropy_from_bernoulli_mle,
        why_bernoulli_distribution,
        demonstrate_probability_calibration,
    )
    from visualization import plot_bernoulli_mle_visual, plot_cross_entropy_loss_surface

    print_section_header(
        2.2,
        "From Bernoulli to Cross-Entropy",
        "Why cross-entropy is the natural loss for classification"
    )

    # Show Bernoulli distribution
    print("""
================================================================================
                        THE BERNOULLI DISTRIBUTION
================================================================================

For binary outcome y in {0, 1} with probability p:

    P(y=1 | p) = p
    P(y=0 | p) = 1 - p

Combined in one formula:
    P(y | p) = p^y * (1-p)^(1-y)

This works because:
    - If y=1: p^1 * (1-p)^0 = p       ✓
    - If y=0: p^0 * (1-p)^1 = 1-p     ✓
""")

    # Visualize Bernoulli for different p values
    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
    for i, p in enumerate([0.2, 0.5, 0.8]):
        axes[i].bar([0, 1], [1-p, p], color=["steelblue", "orange"], edgecolor="black")
        axes[i].set_title(f"Bernoulli(p={p})", fontsize=12)
        axes[i].set_xlabel("Outcome (y)")
        axes[i].set_ylabel("Probability")
        axes[i].set_xticks([0, 1])
        axes[i].set_ylim(0, 1)
    plt.tight_layout()
    plt.show()

    # Mathematical derivation
    print("\n--- Mathematical Derivation ---")
    derivation = derive_crossentropy_from_bernoulli_mle(show_steps=True)
    print(derivation)

    # Why Bernoulli?
    print("\n--- Why Bernoulli? ---")
    reasoning = why_bernoulli_distribution(show_reasoning=True)
    print(reasoning)

    # Generate classification data
    print("\n--- Empirical Demonstration ---")
    X, y = generate_imbalanced_classification_data(
        n_samples=n_samples,
        imbalance_ratio=0.4,  # Fairly balanced for this demo
        class_separation=1.5,
        random_state=random_state,
    )

    # Fit logistic regression
    model = LogisticRegression(random_state=random_state, max_iter=1000)
    model.fit(X, y)
    y_pred_proba = model.predict_proba(X)[:, 1]

    # Show MLE visualization
    print("Visualizing the MLE process for classification...")
    fig = plot_bernoulli_mle_visual(y, y_pred_proba)
    plt.show()

    # Show loss function shapes
    print("\n--- Cross-Entropy Loss Curves ---")
    fig = plot_cross_entropy_loss_surface()
    plt.show()

    # Calibration check
    print("\n--- Probability Calibration (The 'Residual Check' for Classification) ---")
    calibration_curve, brier_score, fig = demonstrate_probability_calibration(
        y, y_pred_proba, n_bins=10, show_plot=True
    )
    plt.show()

    print_key_insight("""
KEY TAKEAWAY:

    Cross-Entropy = Negative Log-Likelihood of Bernoulli Distribution

Just like:
    - MSE = Negative log-likelihood of Gaussian
    - MAE = Negative log-likelihood of Laplace

THE PATTERN:
    1. Choose distribution assumption for y|x
    2. Write likelihood function
    3. Take negative log-likelihood
    4. That IS your loss function!

For classification:
    Bernoulli assumption  -->  Cross-Entropy loss

CALIBRATION IS THE RESIDUAL CHECK:
    - Good calibration = Bernoulli assumption is correct
    - Poor calibration = Consider other distributions (probit, cloglog)
    - Or recalibrate with Platt scaling / isotonic regression
""")


def demonstration_10_alternative_classification_losses(
    n_samples: int = 500,
    random_state: int = 42,
) -> None:
    """
    SECTION 2.3: When Bernoulli is wrong - Alternatives.

    Parameters
    ----------
    n_samples : int
        Sample size
    random_state : int
        Random seed

    Notes
    -----
    Quick overview - not deep dive.
    Message: Bernoulli is standard and works 90% of the time.
    But knowing alternatives exist is important for edge cases.
    """
    from theory import compare_classification_distributions, explain_cross_entropy_properties

    print_section_header(
        2.3,
        "Alternative Classification Distributions",
        "When Bernoulli might not be the best choice"
    )

    # Compare distributions
    comparison = compare_classification_distributions()
    print(comparison)

    # Explain cross-entropy properties
    print("\n--- Cross-Entropy Properties ---")
    properties = explain_cross_entropy_properties()
    print(properties)

    # Visual comparison of link functions
    print("\n--- Link Functions Comparison ---")
    fig, ax = plt.subplots(figsize=(10, 6))

    z = np.linspace(-5, 5, 200)

    # Logistic (sigmoid)
    logistic = 1 / (1 + np.exp(-z))

    # Probit (Gaussian CDF)
    from scipy import stats
    probit = stats.norm.cdf(z)

    # Complementary log-log
    cloglog = 1 - np.exp(-np.exp(z))

    ax.plot(z, logistic, "b-", linewidth=2, label="Logistic (sigmoid)")
    ax.plot(z, probit, "g--", linewidth=2, label="Probit (Gaussian CDF)")
    ax.plot(z, cloglog, "r:", linewidth=2, label="Complementary log-log")
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
    ax.axvline(0, color="gray", linestyle="--", alpha=0.5)

    ax.set_xlabel("Linear Predictor (z = w^T x)", fontsize=12)
    ax.set_ylabel("Probability", fontsize=12)
    ax.set_title("Link Functions: Converting Linear Predictor to Probability", fontsize=14)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-0.05, 1.05)

    ax.text(
        0.02, 0.98,
        "All three are similar in the middle.\nDifferences are in the tails.\nLogistic is most common.",
        transform=ax.transAxes,
        verticalalignment="top",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()
    plt.show()

    print_key_insight("""
PRACTICAL ADVICE:

1. START with Bernoulli (logistic regression, cross-entropy)
   - Works 90%+ of the time
   - Well understood, fast, stable

2. CHECK calibration with reliability diagram
   - Points on diagonal = good calibration
   - Systematic deviation = consider alternatives

3. ALTERNATIVES for special cases:
   - Probit: When you believe there's a latent Gaussian variable
   - Cloglog: For rare events (asymmetric)
   - Beta-Binomial: For overdispersed data

4. NON-PROBABILISTIC option:
   - Hinge loss (SVM): When you only care about decision boundary
   - No calibrated probabilities, but maximum margin

For most problems, just use cross-entropy and you'll be fine!
""")


def demonstration_11_imbalance_and_metrics(
    n_samples: int = 1000,
    imbalance_ratio: float = 0.05,
    random_state: int = 42,
) -> None:
    """
    SECTION 2.4: Class imbalance, MLE bias, and metric sensitivity.

    This is the PAYOFF - connects everything together.

    Parameters
    ----------
    n_samples : int
        Number of samples
    imbalance_ratio : float
        Fraction in minority class
    random_state : int
        Random seed

    Notes
    -----
    Brings together:
    - MLE theory (why imbalance biases it)
    - Residual thinking (how it shows up)
    - Metric understanding (which break and why)
    - Solution (weighted loss)
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, average_precision_score,
    )

    from visualization import (
        plot_metric_sensitivity_to_imbalance,
        plot_weighted_vs_unweighted_crossentropy,
        plot_classification_residuals,
    )

    print_section_header(
        2.4,
        "Class Imbalance and MLE Bias",
        "How majority class dominates loss function and breaks metrics"
    )

    # Generate imbalanced data
    print(f"\nGenerating data: {imbalance_ratio:.1%} minority class")
    X, y = generate_imbalanced_classification_data(
        n_samples=n_samples,
        imbalance_ratio=imbalance_ratio,
        class_separation=1.5,
        random_state=random_state,
    )

    n_minority = int(y.sum())
    n_majority = len(y) - n_minority
    print(f"Class distribution: {n_majority} negative (Class 0), {n_minority} positive (Class 1)")
    print(f"Ratio: {n_majority/n_minority:.1f}:1")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    # Train unweighted model
    print("\n" + "="*70)
    print("1. UNWEIGHTED MODEL (Standard MLE)")
    print("="*70)
    model_unweighted = LogisticRegression(random_state=random_state, max_iter=1000)
    model_unweighted.fit(X_train, y_train)
    y_pred_proba_unweighted = model_unweighted.predict_proba(X_test)[:, 1]
    y_pred_unweighted = model_unweighted.predict(X_test)

    print(f"Mean predicted probability: {y_pred_proba_unweighted.mean():.3f}")
    print(f"True minority rate in test: {y_test.mean():.3f}")
    print("→ Model is conservative (predicts lower probabilities)")

    # Train weighted model
    print("\n" + "="*70)
    print("2. WEIGHTED MODEL (Balanced MLE)")
    print("="*70)
    model_weighted = LogisticRegression(
        class_weight="balanced", random_state=random_state, max_iter=1000
    )
    model_weighted.fit(X_train, y_train)
    y_pred_proba_weighted = model_weighted.predict_proba(X_test)[:, 1]
    y_pred_weighted = model_weighted.predict(X_test)

    print(f"Mean predicted probability: {y_pred_proba_weighted.mean():.3f}")
    print("→ Model predictions closer to true rate")

    # Compare residuals
    print("\n" + "="*70)
    print("3. RESIDUAL ANALYSIS")
    print("="*70)

    fig = plot_weighted_vs_unweighted_crossentropy(
        y_test, y_pred_proba_unweighted, y_pred_proba_weighted
    )
    plt.suptitle("Unweighted vs Weighted Cross-Entropy", fontsize=14)
    plt.show()

    # Metrics comparison
    print("\n" + "="*70)
    print("4. METRIC COMPARISON")
    print("="*70)

    def calc_metrics(y_true, y_pred, y_proba):
        return {
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred, zero_division=0),
            "Recall": recall_score(y_true, y_pred, zero_division=0),
            "F1": f1_score(y_true, y_pred, zero_division=0),
            "AUC-ROC": roc_auc_score(y_true, y_proba) if len(np.unique(y_true)) > 1 else 0,
            "Avg Precision": average_precision_score(y_true, y_proba) if len(np.unique(y_true)) > 1 else 0,
        }

    metrics_unw = calc_metrics(y_test, y_pred_unweighted, y_pred_proba_unweighted)
    metrics_w = calc_metrics(y_test, y_pred_weighted, y_pred_proba_weighted)

    print("\n{:<15} {:>12} {:>12}".format("Metric", "Unweighted", "Weighted"))
    print("-" * 42)
    for metric in metrics_unw:
        print("{:<15} {:>12.3f} {:>12.3f}".format(
            metric, metrics_unw[metric], metrics_w[metric]
        ))

    # Explain metric sensitivity
    print("""
WHY METRICS RESPOND DIFFERENTLY:
────────────────────────────────────────────────────────────────────────

ACCURACY = (TP + TN) / Total
    • Dominated by TN from majority class
    • Unweighted: High accuracy from predicting all negative
    • Weighted: Lower accuracy, but more honest
    • IMBALANCE SENSITIVE ⚠️

PRECISION = TP / (TP + FP)
    • FP comes from majority class
    • Small FP rate × large majority = many FPs
    • Affected even if minority recall is good
    • MODERATELY SENSITIVE ⚠️

RECALL = TP / (TP + FN)
    • Only looks at minority class (TP + FN)
    • Denominator doesn't include majority at all
    • Unaffected by class proportions
    • NOT SENSITIVE ✓

AUC-ROC = Area under (TPR vs FPR) curve
    • TPR = TP/(TP+FN) - minority only
    • FPR = FP/(FP+TN) - normalized by majority
    • Both rates scale-invariant
    • NOT SENSITIVE ✓
""")

    # Show metric sensitivity across imbalance ratios
    print("\n--- Metric Sensitivity Visualization ---")
    fig = plot_metric_sensitivity_to_imbalance(
        imbalance_ratios=[0.5, 0.3, 0.1, 0.05, 0.02, 0.01]
    )
    plt.show()

    # The MLE connection
    print("""
================================================================================
                         THE MLE CONNECTION
================================================================================

Cross-entropy loss: L = -Σ[yᵢ·log(pᵢ) + (1-yᵢ)·log(1-pᵢ)]

With imbalance (95% negative):
    • 950 negative samples contribute: -Σlog(1-pᵢ)
    • 50 positive samples contribute:  -Σlog(pᵢ)

Gradient: ∂L/∂θ ∝ 950·E[grad|negative] + 50·E[grad|positive]

→ Model is 19× more concerned about negatives!
→ Learns to be conservative (low pᵢ) to avoid hurting majority
→ Minority class gets large residuals (y=1, p̂=0.1 → residual=0.9)

WEIGHTED LOSS FIX:

L = -Σ[w₁·yᵢ·log(pᵢ) + w₀·(1-yᵢ)·log(1-pᵢ)]

Set w₁ = 950/50 = 19, w₀ = 1:
    • Now both classes contribute equally to loss
    • MLE treats them with equal importance
    • Better recall, more balanced predictions
""")

    print_key_insight(f"""
SUMMARY:

Unweighted model:
    • Accuracy: {metrics_unw['Accuracy']:.3f} (looks great, but misleading!)
    • Recall:   {metrics_unw['Recall']:.3f} (terrible for minority class)

Weighted model:
    • Accuracy: {metrics_w['Accuracy']:.3f} (lower, but more honest)
    • Recall:   {metrics_w['Recall']:.3f} (much better!)

RULE OF THUMB:
    • If you care about minority class → weight your loss
    • If accuracy is high but recall is low → you have a problem
    • Prefer metrics insensitive to imbalance: Recall, AUC-ROC, Avg Precision
""")


# =============================================================================
# PART 1.5: REGULARIZATION AND BIAS-VARIANCE DEMONSTRATIONS
# =============================================================================


def demonstration_12_regularization_as_map(
    n_samples: int = 200,
    n_features: int = 10,
    random_state: int = 42,
) -> None:
    """
    SECTION 1.5.1: Regularization as Maximum A Posteriori (MAP) estimation.

    Shows that:
    - Ridge regression = MLE with Gaussian prior on coefficients (L2)
    - Lasso regression = MLE with Laplace prior on coefficients (L1)

    Parameters
    ----------
    n_samples : int
        Number of samples
    n_features : int
        Number of features
    random_state : int
        Random seed

    Notes
    -----
    This is the Bayesian interpretation of regularization.
    Key insight: regularization is not arbitrary - it's prior belief!
    """
    from sklearn.linear_model import LinearRegression, Ridge, Lasso
    from sklearn.preprocessing import StandardScaler

    from theory import derive_ridge_from_map, derive_lasso_from_map
    from visualization import (
        plot_regularization_path,
        plot_regularization_as_prior,
        plot_residuals_with_without_regularization,
    )

    print_section_header(
        1.5,
        "Regularization as MAP Estimation",
        "Ridge = Gaussian prior, Lasso = Laplace prior"
    )

    # Mathematical derivation - Ridge
    print("\n" + "="*70)
    print("RIDGE REGRESSION: MLE + GAUSSIAN PRIOR")
    print("="*70)
    derivation_ridge = derive_ridge_from_map(show_steps=True)
    print(derivation_ridge)

    # Mathematical derivation - Lasso
    print("\n" + "="*70)
    print("LASSO REGRESSION: MLE + LAPLACE PRIOR")
    print("="*70)
    derivation_lasso = derive_lasso_from_map(show_steps=True)
    print(derivation_lasso)

    # Visualize priors
    print("\n--- Prior Distributions on Coefficients ---")
    fig = plot_regularization_as_prior()
    plt.show()

    # Generate data
    print("\n--- Empirical Demonstration ---")
    rng = np.random.default_rng(random_state)

    # Create some correlated features
    X = rng.normal(0, 1, (n_samples, n_features))
    # True coefficients - some large, some small
    true_coefs = np.array([3.0, -2.0, 1.5, 0, 0, 0, 0.5, 0, -0.3, 0])
    y = X @ true_coefs + rng.normal(0, 1, n_samples)

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Show regularization paths
    print("\n--- Ridge Regularization Path ---")
    fig = plot_regularization_path(X_scaled, y, regularization_type="ridge")
    plt.show()

    print("\n--- Lasso Regularization Path ---")
    fig = plot_regularization_path(X_scaled, y, regularization_type="lasso")
    plt.show()

    # Compare residuals
    print("\n--- Residual Comparison: OLS vs Ridge ---")
    fig = plot_residuals_with_without_regularization(X_scaled, y, alpha=1.0)
    plt.show()

    # Fit models and compare coefficients
    print("\n--- Coefficient Comparison ---")
    ols = LinearRegression().fit(X_scaled, y)
    ridge = Ridge(alpha=1.0).fit(X_scaled, y)
    lasso = Lasso(alpha=0.1).fit(X_scaled, y)

    print("\n{:<10} {:>10} {:>10} {:>10} {:>10}".format(
        "Coef", "True", "OLS", "Ridge", "Lasso"
    ))
    print("-" * 52)
    for i in range(n_features):
        print("{:<10} {:>10.3f} {:>10.3f} {:>10.3f} {:>10.3f}".format(
            f"β_{i}",
            true_coefs[i],
            ols.coef_[i],
            ridge.coef_[i],
            lasso.coef_[i],
        ))

    print_key_insight("""
KEY INSIGHT: REGULARIZATION = PRIOR BELIEF

    ┌─────────────────────────────────────────────────────────────────────┐
    │  MLE (no regularization):                                          │
    │      Maximize P(data | parameters)                                 │
    │                                                                    │
    │  MAP (with regularization):                                        │
    │      Maximize P(data | parameters) × P(parameters)                 │
    │                     ↑                      ↑                       │
    │                Likelihood              Prior                       │
    │                   (MLE)           (Regularization)                 │
    └─────────────────────────────────────────────────────────────────────┘

GAUSSIAN PRIOR (RIDGE):
    • Prior: β ~ N(0, τ²)
    • -log(prior) ∝ β²  →  L2 penalty
    • Effect: ALL coefficients shrink toward 0
    • Never exactly 0, just smaller

LAPLACE PRIOR (LASSO):
    • Prior: β ~ Laplace(0, b)
    • -log(prior) ∝ |β|  →  L1 penalty
    • Effect: SOME coefficients become exactly 0
    • Sparsity! Automatic feature selection

WHY THIS MATTERS:
    • Regularization is not arbitrary
    • It's your prior belief about coefficient magnitudes
    • Gaussian prior = "I believe all coefficients are small"
    • Laplace prior = "I believe most coefficients are zero"
""")


def demonstration_13_bias_variance_in_residuals(
    n_samples: int = 300,
    max_degree: int = 12,
    random_state: int = 42,
) -> None:
    """
    SECTION 1.5.2: Bias-variance tradeoff visualized through residuals.

    Shows how model complexity affects:
    - Training vs test error
    - Residual patterns
    - Bias and variance decomposition

    Parameters
    ----------
    n_samples : int
        Number of samples
    max_degree : int
        Maximum polynomial degree
    random_state : int
        Random seed

    Notes
    -----
    The classic U-curve, but connected to residual analysis.
    Shows that underfitting = systematic residuals (bias),
    overfitting = noisy residuals with high variance.
    """
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression

    from theory import (
        bias_variance_decomposition_theorem,
        regularization_effect_on_bias_variance,
    )
    from visualization import (
        plot_bias_variance_with_model_complexity,
        plot_learning_curves,
    )

    print_section_header(
        1.52,
        "Bias-Variance Tradeoff",
        "The fundamental tradeoff in machine learning"
    )

    # Theorem
    print("\n" + "="*70)
    print("THE BIAS-VARIANCE DECOMPOSITION THEOREM")
    print("="*70)
    theorem = bias_variance_decomposition_theorem(show_proof=True)
    print(theorem)

    # Generate nonlinear data
    print("\n--- Generating Nonlinear Data ---")
    rng = np.random.default_rng(random_state)

    X = np.linspace(-3, 3, n_samples).reshape(-1, 1)
    # True function: sin wave
    true_signal = np.sin(X.flatten()) + 0.5 * X.flatten()
    noise = rng.normal(0, 0.5, n_samples)
    y = true_signal + noise

    # Train-test split
    split_idx = int(0.7 * n_samples)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Show bias-variance vs complexity
    print("\n--- Bias-Variance vs Model Complexity ---")
    fig = plot_bias_variance_with_model_complexity(
        X_train, y_train, X_test, y_test, max_degree=max_degree
    )
    plt.show()

    # Show learning curves for different complexities
    print("\n--- Learning Curves: Low Complexity (Degree 1) ---")
    from sklearn.pipeline import make_pipeline
    model_low = make_pipeline(
        PolynomialFeatures(1, include_bias=False),
        LinearRegression()
    )
    fig = plot_learning_curves(X, y, model=model_low)
    plt.suptitle("Low Complexity: High Bias", fontsize=14)
    plt.show()

    print("\n--- Learning Curves: High Complexity (Degree 10) ---")
    model_high = make_pipeline(
        PolynomialFeatures(10, include_bias=False),
        LinearRegression()
    )
    fig = plot_learning_curves(X, y, model=model_high)
    plt.suptitle("High Complexity: High Variance", fontsize=14)
    plt.show()

    # Show residual patterns at different complexities
    print("\n--- Residual Patterns ---")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for i, degree in enumerate([1, 3, 10]):
        model = make_pipeline(
            PolynomialFeatures(degree, include_bias=False),
            LinearRegression()
        )
        model.fit(X_train, y_train)
        y_pred_test = model.predict(X_test)
        residuals = y_test - y_pred_test

        axes[i].scatter(y_pred_test, residuals, alpha=0.5, s=30)
        axes[i].axhline(0, color="red", linestyle="--")
        axes[i].set_xlabel("Predicted", fontsize=10)
        axes[i].set_ylabel("Residual", fontsize=10)

        if degree == 1:
            axes[i].set_title(f"Degree {degree}: HIGH BIAS\n(Systematic pattern)", fontsize=11)
        elif degree == 10:
            axes[i].set_title(f"Degree {degree}: HIGH VARIANCE\n(Noisy, unstable)", fontsize=11)
        else:
            axes[i].set_title(f"Degree {degree}: BALANCED\n(Random residuals)", fontsize=11)

    plt.tight_layout()
    plt.show()

    # Regularization effect
    print("\n" + "="*70)
    print("REGULARIZATION'S EFFECT ON BIAS-VARIANCE")
    print("="*70)
    reg_effect = regularization_effect_on_bias_variance()
    print(reg_effect)

    print_key_insight("""
KEY INSIGHT: RESIDUALS REVEAL BIAS AND VARIANCE

UNDERFITTING (High Bias):
    • Training error ≈ Test error (both high)
    • Residuals show SYSTEMATIC PATTERN
    • Model misses true structure
    • Adding more data won't help!
    • Solution: Increase complexity

OVERFITTING (High Variance):
    • Training error << Test error (gap)
    • Training residuals small, test residuals large
    • Model memorizes training noise
    • More data might help
    • Solution: Regularize or simplify

THE U-CURVE:
    • Sweet spot minimizes TEST error
    • Not training error (that always decreases)
    • Cross-validation finds this sweet spot

LEARNING CURVES TELL THE STORY:
    • Converging curves = Low variance (good generalization)
    • Large gap = High variance (overfitting)
    • Both curves plateau high = High bias (need more features)
""")


def demonstration_14_overfitting_detection(
    n_samples: int = 500,
    random_state: int = 42,
) -> None:
    """
    SECTION 1.5.3: How to detect overfitting using residuals and learning curves.

    Practical guide to diagnosing model problems.

    Parameters
    ----------
    n_samples : int
        Number of samples
    random_state : int
        Random seed

    Notes
    -----
    Brings together all the diagnostic tools:
    - Training vs test error gap
    - Learning curves
    - Residual analysis
    - Regularization as solution
    """
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.pipeline import make_pipeline

    from theory import mse_vs_mae_bias_variance_tradeoff
    from visualization import plot_learning_curves, plot_residuals_diagnostic

    print_section_header(
        1.53,
        "Detecting and Fixing Overfitting",
        "Practical diagnostics using residuals"
    )

    # Generate data
    print("\n--- Generating Nonlinear Data ---")
    rng = np.random.default_rng(random_state)

    X = np.sort(rng.uniform(-3, 3, n_samples)).reshape(-1, 1)
    true_signal = np.sin(1.5 * X.flatten()) + 0.3 * X.flatten() ** 2
    noise = rng.normal(0, 0.5, n_samples)
    y = true_signal + noise

    # Split
    split_idx = int(0.7 * n_samples)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Train overfit model
    print("\n" + "="*70)
    print("STEP 1: DETECT OVERFITTING")
    print("="*70)

    degree = 15  # Intentionally too complex
    model_overfit = make_pipeline(
        PolynomialFeatures(degree, include_bias=False),
        LinearRegression()
    )
    model_overfit.fit(X_train, y_train)

    y_train_pred = model_overfit.predict(X_train)
    y_test_pred = model_overfit.predict(X_test)

    train_mse = np.mean((y_train - y_train_pred) ** 2)
    test_mse = np.mean((y_test - y_test_pred) ** 2)

    print(f"\nPolynomial Degree: {degree}")
    print(f"Training MSE: {train_mse:.4f}")
    print(f"Test MSE:     {test_mse:.4f}")
    print(f"Gap:          {test_mse - train_mse:.4f}")
    print("\n→ Large gap indicates OVERFITTING!")

    # Show fit
    print("\n--- Overfitting Model Fit ---")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(X_train, y_train, alpha=0.3, s=20, label="Training data")
    ax.scatter(X_test, y_test, alpha=0.3, s=20, color="red", label="Test data")

    x_plot = np.linspace(-3, 3, 200).reshape(-1, 1)
    y_plot = model_overfit.predict(x_plot)
    ax.plot(x_plot, y_plot, "g-", linewidth=2, label=f"Degree {degree} fit")
    ax.plot(x_plot, np.sin(1.5 * x_plot.flatten()) + 0.3 * x_plot.flatten() ** 2,
            "k--", linewidth=2, label="True function")

    ax.set_xlabel("X", fontsize=12)
    ax.set_ylabel("y", fontsize=12)
    ax.set_title("Overfitting: Model chases training noise", fontsize=14)
    ax.legend()
    ax.set_ylim(-3, 5)
    plt.tight_layout()
    plt.show()

    # Learning curves
    print("\n--- Learning Curves (Overfitting Detection) ---")
    fig = plot_learning_curves(X, y, model=model_overfit)
    plt.suptitle("Overfit Model: Large Gap Between Curves", fontsize=14)
    plt.show()

    # Residual analysis
    print("\n--- Residual Diagnostics ---")
    fig = plot_residuals_diagnostic(y_test, y_test_pred, "Overfit Model Residuals")
    plt.show()

    # Fix with regularization
    print("\n" + "="*70)
    print("STEP 2: FIX WITH REGULARIZATION")
    print("="*70)

    alphas = [0, 0.1, 1.0, 10.0, 100.0]
    print("\n{:<10} {:>12} {:>12} {:>12}".format("Alpha", "Train MSE", "Test MSE", "Gap"))
    print("-" * 48)

    best_alpha = 0
    best_test_mse = float("inf")

    for alpha in alphas:
        model = make_pipeline(
            PolynomialFeatures(degree, include_bias=False),
            Ridge(alpha=alpha)
        )
        model.fit(X_train, y_train)

        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)

        train_mse = np.mean((y_train - train_pred) ** 2)
        test_mse = np.mean((y_test - test_pred) ** 2)
        gap = test_mse - train_mse

        print("{:<10} {:>12.4f} {:>12.4f} {:>12.4f}".format(alpha, train_mse, test_mse, gap))

        if test_mse < best_test_mse:
            best_test_mse = test_mse
            best_alpha = alpha

    print(f"\nBest regularization: α = {best_alpha}")

    # Show regularized fit
    print("\n--- Regularized Model Fit ---")
    model_regularized = make_pipeline(
        PolynomialFeatures(degree, include_bias=False),
        Ridge(alpha=best_alpha)
    )
    model_regularized.fit(X_train, y_train)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(X_train, y_train, alpha=0.3, s=20, label="Training data")
    ax.scatter(X_test, y_test, alpha=0.3, s=20, color="red", label="Test data")

    y_plot_reg = model_regularized.predict(x_plot)
    ax.plot(x_plot, y_plot, "g--", linewidth=1, alpha=0.5, label=f"Unregularized")
    ax.plot(x_plot, y_plot_reg, "b-", linewidth=2, label=f"Ridge (α={best_alpha})")
    ax.plot(x_plot, np.sin(1.5 * x_plot.flatten()) + 0.3 * x_plot.flatten() ** 2,
            "k--", linewidth=2, label="True function")

    ax.set_xlabel("X", fontsize=12)
    ax.set_ylabel("y", fontsize=12)
    ax.set_title("Regularization Reduces Overfitting", fontsize=14)
    ax.legend()
    ax.set_ylim(-3, 5)
    plt.tight_layout()
    plt.show()

    # Show learning curves for regularized model
    print("\n--- Learning Curves (After Regularization) ---")
    fig = plot_learning_curves(X, y, model=model_regularized)
    plt.suptitle("Regularized Model: Curves Converge", fontsize=14)
    plt.show()

    # Loss function perspective
    print("\n" + "="*70)
    print("MSE vs MAE: BIAS-VARIANCE IMPLICATIONS")
    print("="*70)
    mse_mae_comparison = mse_vs_mae_bias_variance_tradeoff()
    print(mse_mae_comparison)

    print_key_insight("""
OVERFITTING DETECTION CHECKLIST:

1. TRAIN-TEST GAP
   • Large gap = overfitting
   • Train MSE << Test MSE
   • Gap should be small relative to test error

2. LEARNING CURVES
   • Overfitting: Large gap, validation plateaus while training drops
   • Underfitting: Both curves plateau at high error
   • Good fit: Both curves converge to low error

3. RESIDUAL PATTERNS
   • Overfitting: Training residuals tiny, test residuals large
   • Underfitting: Both residuals show systematic patterns
   • Good fit: Random residuals, similar distributions

SOLUTIONS:

┌─────────────────────────────────────────────────────────────────────┐
│  Problem         │  Diagnosis           │  Solution                │
├─────────────────────────────────────────────────────────────────────┤
│  Overfitting     │  Large train-test    │  • Regularization (L1/L2)│
│                  │  gap                 │  • More data             │
│                  │                      │  • Simpler model         │
│                  │                      │  • Dropout (neural nets) │
├─────────────────────────────────────────────────────────────────────┤
│  Underfitting    │  Both errors high,   │  • More features         │
│                  │  systematic pattern  │  • More complex model    │
│                  │  in residuals        │  • Less regularization   │
└─────────────────────────────────────────────────────────────────────┘

THE GOLDEN RULE:
    Always check BOTH training AND test error.
    Only test error tells you how well you'll generalize!
""")


def demonstration_summary() -> None:
    """
    Print summary of all key takeaways.
    """
    print_section_header("Final", "Key Takeaways", "What to remember")

    print("""
================================================================================
                              MAIN POINTS
================================================================================

PART 1: REGRESSION
────────────────────────────────────────────────────────────────────────────────

1. THREE DISTRIBUTIONS:
   - Input data: X has some distribution
   - Model: Learns FUNCTION f(X) -> y, not a distribution
   - Residuals: y - f(X) has distribution (this is what loss assumes!)

2. LOSS = NEGATIVE LOG-LIKELIHOOD (Regression):
   - Gaussian noise -> MSE -> Optimizes for mean
   - Laplace noise  -> MAE -> Optimizes for median
   - Choosing loss = choosing residual distribution assumption

3. CHECK YOUR RESIDUALS (Regression):
   - Random, no patterns -> Good fit
   - Patterns -> Model bias (wrong architecture)
   - Wrong distribution -> Wrong loss function

PART 1.5: REGULARIZATION AND BIAS-VARIANCE
────────────────────────────────────────────────────────────────────────────────

4. REGULARIZATION = MAP ESTIMATION:
   - MLE: Maximize P(data | parameters)
   - MAP: Maximize P(data | params) × P(params)  [with prior]

   - Gaussian prior → L2 penalty (Ridge) → shrinks all coefficients
   - Laplace prior  → L1 penalty (Lasso) → makes coefficients exactly 0 (sparsity)

5. BIAS-VARIANCE TRADEOFF:
   - E[Error] = Bias² + Variance + Irreducible Noise
   - Low complexity / high λ: High bias, low variance (underfitting)
   - High complexity / low λ: Low bias, high variance (overfitting)
   - Sweet spot: Minimum TEST error (use cross-validation)

6. DETECTING OVERFITTING:
   - Train-test gap: Large gap = overfitting
   - Learning curves: Gap between curves = variance
   - Residual patterns: Systematic = bias, random = good
   - Solution: Regularization trades bias for variance

PART 2: CLASSIFICATION
────────────────────────────────────────────────────────────────────────────────

7. THE SAME PATTERN APPLIES:
   - Bernoulli distribution -> Cross-entropy loss
   - The likelihood framework is universal!

8. CALIBRATION IS THE RESIDUAL CHECK:
   - Points on diagonal in reliability diagram = good calibration
   - Deviation = Bernoulli assumption may be wrong
   - Or need to recalibrate (Platt scaling, isotonic regression)

9. CLASS IMBALANCE AND METRICS:
   - Standard MLE biased toward majority class
   - Majority dominates the loss function

   SENSITIVE metrics (be careful):
   • Accuracy - dominated by majority class TN
   • Precision - FP comes from majority

   NOT SENSITIVE metrics (preferred):
   • Recall - only looks at minority class
   • AUC-ROC - normalized rates, scale-invariant
   • Average Precision - focuses on positive class

   Solution: Weight the loss function!

================================================================================
                         THE COMPLETE PATTERN
================================================================================

    ┌─────────────────────────────────────────────────────────────────────┐
    │  1. Choose distributional assumption for y|x                        │
    │  2. Write likelihood function                                       │
    │  3. Take negative log-likelihood                                    │
    │  4. That IS your loss function                                      │
    │  5. MLE = minimizing this loss                                      │
    └─────────────────────────────────────────────────────────────────────┘

    REGRESSION:                         CLASSIFICATION:
    ────────────                        ────────────────
    Gaussian -> MSE -> Mean             Bernoulli -> Cross-Entropy
    Laplace  -> MAE -> Median           Probit    -> Probit NLL
    Student-t -> Robust loss            Cloglog   -> Cloglog NLL

================================================================================
                    GO FORTH AND CHECK YOUR RESIDUALS!
                    (Or calibration curves for classification)
================================================================================
""")

    print("""
ANTICIPATED QUESTIONS:

REGRESSION:
────────────────────────────────────────────────────────────────────────────────

Q: "How do I know which loss to use?"
A: Start with domain knowledge about noise. If unsure, try multiple
   and compare residuals.

Q: "Can I use different loss for different samples?"
A: Yes! Sample weighting is exactly this. Important for imbalanced data.

Q: "What if neither Gaussian nor Laplace fits?"
A: Try Huber (hybrid), or custom loss.
   XGBoost supports many: Gamma, Tweedie, quantile.

REGULARIZATION:
────────────────────────────────────────────────────────────────────────────────

Q: "How do I choose the regularization strength (λ)?"
A: Cross-validation! Use GridSearchCV or RidgeCV/LassoCV.
   Plot CV error vs λ to find the sweet spot.

Q: "Ridge or Lasso? Which one should I use?"
A: Ridge: When all features are relevant (no sparsity expected)
   Lasso: When many features are irrelevant (want feature selection)
   Elastic Net: Compromise between the two

Q: "How does regularization relate to bias-variance?"
A: Regularization trades bias for variance!
   - More regularization = more bias, less variance
   - Less regularization = less bias, more variance
   - Optimal λ minimizes total error (bias² + variance)

Q: "My learning curves show a gap. What does this mean?"
A: Large gap = high variance = overfitting
   Solutions: More regularization, more data, or simpler model.
   If both curves plateau high = high bias = need more features.

CLASSIFICATION:
────────────────────────────────────────────────────────────────────────────────

Q: "Why cross-entropy and not MSE for classification?"
A: MSE has vanishing gradients near 0 and 1. Cross-entropy penalizes
   confident wrong predictions heavily. Plus, it's proper MLE under Bernoulli!

Q: "My model has high accuracy but low recall. What's wrong?"
A: Class imbalance! Your model learned to predict majority class.
   Solution: Use weighted loss or oversample minority.

Q: "Which metrics should I use for imbalanced data?"
A: Prefer metrics NOT sensitive to imbalance:
   - Recall (sensitivity)
   - AUC-ROC
   - Average Precision (AUC-PR)
   Avoid raw accuracy!

Q: "How do I know if my probabilities are calibrated?"
A: Plot a reliability diagram. Points on diagonal = calibrated.
   This is the "residual check" for classification.

GENERAL:
────────────────────────────────────────────────────────────────────────────────

Q: "What about neural networks?"
A: Same principles! Any differentiable loss works.
   Common: MSE, MAE, cross-entropy, focal loss.

Q: "Does this apply to deep learning?"
A: Absolutely. Loss function is loss function, regardless of model
   complexity. The MLE framework is universal.

Q: "How do I check if my residuals match assumed distribution?"
A: Q-Q plot, Shapiro-Wilk test, histogram + overlay.
   But patterns are more important than perfect match.
""")
