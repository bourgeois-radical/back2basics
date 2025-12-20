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
    compare_estimators_with_outliers,
    demonstrate_mean_minimizes_mse,
    demonstrate_median_minimizes_mae,
    derive_mae_from_laplace_mle,
    derive_mse_from_gaussian_mle,
    simulate_bias_variance_tradeoff,
)
from utils import print_key_insight, print_section_header
from visualization import (
    plot_bias_variance_decomposition,
    plot_classification_residuals,
    plot_imbalance_effect,
    plot_loss_function_comparison,
    plot_loss_function_shapes,
    plot_mle_derivation_visual,
    plot_outlier_comparison,
    plot_residuals_diagnostic,
    plot_three_distributions_concept,
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


def demonstration_summary() -> None:
    """
    Print summary of all key takeaways.
    """
    print_section_header(6, "Key Takeaways", "What to remember")

    print("""
================================================================================
                              MAIN POINTS
================================================================================

1. THREE DISTRIBUTIONS:
   - Input data: X has some distribution
   - Model: Learns FUNCTION f(X) -> y, not a distribution
   - Residuals: y - f(X) has distribution (this is what loss assumes!)

2. LOSS = NEGATIVE LOG-LIKELIHOOD:
   - Gaussian noise -> MSE -> Optimizes for mean
   - Laplace noise  -> MAE -> Optimizes for median
   - Choosing loss = choosing residual distribution assumption

3. CHECK YOUR RESIDUALS:
   - Random, no patterns -> Good fit
   - Patterns -> Model bias (wrong architecture)
   - Wrong distribution -> Wrong loss function

4. IMBALANCE:
   - Standard MLE biased toward majority class
   - Solution: Weight loss function
   - Don't trust accuracy alone!

5. PRACTICAL WORKFLOW:
   a. Think about noise distribution (domain knowledge)
   b. Choose appropriate loss function
   c. Train model
   d. Check residuals (patterns? distribution match?)
   e. Iterate if needed

================================================================================
                    GO FORTH AND CHECK YOUR RESIDUALS!
================================================================================
""")

    print("""
ANTICIPATED QUESTIONS:

Q: "How do I know which loss to use?"
A: Start with domain knowledge about noise. If unsure, try multiple
   and compare residuals.

Q: "Can I use different loss for different samples?"
A: Yes! Sample weighting is exactly this. Important for imbalanced data.

Q: "What about neural networks?"
A: Same principles! Any differentiable loss works.
   Common: MSE, MAE, cross-entropy, focal loss.

Q: "Does this apply to deep learning?"
A: Absolutely. Loss function is loss function, regardless of model
   complexity.

Q: "How do I check if my residuals match assumed distribution?"
A: Q-Q plot, Shapiro-Wilk test, histogram + overlay.
   But patterns are more important than perfect match.

Q: "What if neither Gaussian nor Laplace fits?"
A: Try Huber (hybrid), or custom loss.
   XGBoost supports many: Gamma, Tweedie, quantile.
""")
