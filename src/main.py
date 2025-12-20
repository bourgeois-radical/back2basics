#!/usr/bin/env python3
"""
Understanding MLE and Loss Functions Through Residual Distributions
====================================================================

A Deep Dive for Data Scientists

Duration: ~40 minutes

This is the main orchestration file for the presentation. It runs through
all demonstrations in order with explanatory comments that serve as
speaker notes.

Key Questions We'll Answer:
1. What are the THREE distributions in machine learning?
2. Why does MSE optimize for the mean?
3. Why does MAE optimize for the median?
4. How does loss function choice affect our models?
5. Why does class imbalance break our models?

Usage:
    python main.py              # Run full presentation
    python main.py --section 1  # Run specific section
    python main.py --interactive  # Interactive mode (pause after each section)
"""

import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np

# Set random seed for reproducibility across all demonstrations
np.random.seed(42)


# =============================================================================
# SETUP: Import all modules and configure plotting
# =============================================================================
#
# SPEAKER NOTE: Before starting, make sure all dependencies are installed.
# Run: pip install numpy pandas matplotlib seaborn scikit-learn scipy
# Optional: pip install xgboost (for additional model examples)

from utils import print_section_header, set_plot_style
from demonstrations import (
    demonstration_1_three_distributions,
    demonstration_2_mle_to_mse,
    demonstration_3_mle_to_mae,
    demonstration_4_mse_vs_mae_outliers,
    demonstration_5_model_misspecification,
    demonstration_6_real_world_example,
    demonstration_7_classification_imbalance,
    demonstration_8_bias_variance_residuals,
    demonstration_summary,
)


def setup_presentation():
    """
    Configure the presentation environment.

    SPEAKER NOTE: This sets up consistent styling for all plots.
    The 'talk' context makes fonts large enough for presentation.
    """
    set_plot_style(context="talk", font_scale=1.2)
    print("=" * 70)
    print("  UNDERSTANDING MLE AND LOSS FUNCTIONS")
    print("  Through Residual Distributions")
    print("=" * 70)
    print()
    print("This presentation covers the fundamental connection between:")
    print("  - Maximum Likelihood Estimation (MLE)")
    print("  - Loss functions (MSE, MAE, Huber, Cross-Entropy)")
    print("  - Distributional assumptions about residuals")
    print()
    print("=" * 70)
    print()


def section_1():
    """
    SECTION 1: The Three Distributions (~8 minutes)
    ================================================

    SPEAKER NOTE: This is THE most important concept in the presentation.
    Take your time here. Make sure the audience understands these three
    distributions before moving on.

    Key points to emphasize:
    1. Most people think about "the distribution of the data" as one thing
    2. In reality, there are THREE different distributions:
       - Input distribution P(X)
       - The learned function f: X -> y (NOT a distribution!)
       - Residual distribution P(y - f(X))
    3. Loss functions make assumptions about the RESIDUAL distribution,
       not the input distribution!

    Common misconception to address:
    "But my data isn't normally distributed, so I shouldn't use MSE"
    -> WRONG! MSE assumes RESIDUALS are normal, not inputs!
    """
    demonstration_1_three_distributions(n_samples=500, noise_std=2.0)


def section_2a():
    """
    SECTION 2a: MLE to MSE (~5 minutes)
    ====================================

    SPEAKER NOTE: Now we derive WHY MSE corresponds to Gaussian residuals.

    Walk through the mathematical derivation step by step:
    1. Start with Gaussian likelihood for each residual
    2. Multiply all likelihoods together (independence assumption)
    3. Take the log (converts product to sum, more numerically stable)
    4. Negate (we minimize loss, not maximize likelihood)
    5. Drop constants (they don't affect the argmin)
    6. Result: Sum of squared errors!

    KEY INSIGHT to emphasize:
    - When you choose MSE, you're implicitly assuming Gaussian residuals
    - The MEAN is the MLE estimator for the center of a Gaussian
    - This is why MSE "pulls toward the mean"
    """
    demonstration_2_mle_to_mse(n_samples=300)


def section_2b():
    """
    SECTION 2b: MLE to MAE (~5 minutes)
    ====================================

    SPEAKER NOTE: Parallel derivation, but now with Laplace distribution.

    The Laplace distribution:
    - Looks like a "pointy" Gaussian (exponential tails)
    - Has heavier tails than Gaussian (more probability mass in extremes)
    - The MEDIAN is its MLE estimator (not the mean!)

    Walk through the derivation:
    1. Laplace likelihood: p(x) = (1/2b) * exp(-|x|/b)
    2. Log-likelihood: -log(2b) - |x|/b per sample
    3. Negate and drop constants
    4. Result: Sum of absolute errors!

    KEY INSIGHT:
    - MAE assumes Laplace residuals
    - Median is robust to outliers (doesn't get "pulled")
    - This explains WHY MAE is robust!
    """
    demonstration_3_mle_to_mae(n_samples=300)


def section_3a():
    """
    SECTION 3a: MSE vs MAE with Outliers (~5 minutes)
    ==================================================

    SPEAKER NOTE: This is the "money shot" - shows the practical difference.

    What we'll see:
    1. Generate data with ~10% outliers
    2. Fit models using MSE, MAE, and Huber loss
    3. Compare how each model handles the outliers

    Points to emphasize:
    - MSE model gets "pulled" toward outliers
    - MAE model ignores outliers, fits the majority well
    - Huber is a compromise: quadratic for small errors, linear for large

    The residual histograms tell the story:
    - MSE: Few extreme residuals (model chased the outliers)
    - MAE: Many moderate residuals (model ignored outliers)

    Practical advice:
    - If outliers are measurement errors -> MAE
    - If all data points matter -> MSE
    - If unsure -> Huber (best of both worlds)
    """
    demonstration_4_mse_vs_mae_outliers(n_samples=500, outlier_fraction=0.1)


def section_3b():
    """
    SECTION 3b: Model Misspecification (~3 minutes)
    ================================================

    SPEAKER NOTE: This is a warning about what loss functions CAN'T fix.

    Key message: Loss function choice CANNOT fix wrong model architecture!

    What we'll show:
    1. Generate quadratic data: y = x^2 + noise
    2. Fit a LINEAR model (wrong!)
    3. Look at the residuals

    The residuals will show a clear parabolic pattern.
    This is BIAS - systematic error from the wrong model.

    No loss function can fix this! You need to:
    1. First get the architecture right (polynomial features, neural net, etc.)
    2. THEN choose the appropriate loss function

    This is why residual analysis is so important:
    - Patterns in residuals = model bias = wrong architecture
    - Random residuals = good fit
    """
    demonstration_5_model_misspecification(n_samples=300)


def section_3c():
    """
    SECTION 3c: Real World Example (~4 minutes)
    ============================================

    SPEAKER NOTE: Show that synthetic examples generalize to real data.

    Using California Housing dataset:
    1. Try multiple loss functions
    2. Compare metrics
    3. Analyze residuals

    Points to make:
    - Real data is messy, often has outliers
    - Different losses give different results
    - No single loss function is universally best
    - Always check residuals!

    If XGBoost is available, also show:
    - XGBoost with squared error vs absolute error
    - Tree-based models can use any differentiable loss
    """
    demonstration_6_real_world_example(dataset_name="california_housing")


def section_4():
    """
    SECTION 4: Classification with Imbalanced Data (~6 minutes)
    ============================================================

    SPEAKER NOTE: MLE fails spectacularly with class imbalance!

    Setup:
    - 95% majority class (negative), 5% minority class (positive)
    - Think: fraud detection, rare disease diagnosis, etc.

    What happens with standard MLE:
    1. MLE optimizes AVERAGE loss
    2. Majority class dominates the average
    3. Model learns to always predict majority
    4. Gets 95% accuracy! (By predicting all negative)
    5. But 0% recall on the minority class we care about!

    The "accuracy paradox":
    - Accuracy is a terrible metric for imbalanced data
    - A "dumb" model that always predicts majority wins!

    Solution: Weight the loss function
    - Weight minority class higher
    - Equivalently: oversample minority, undersample majority
    - Use appropriate metrics: F1, AUC-PR, recall at fixed precision

    Show the probability distributions:
    - Unweighted model: Conservative (all low probabilities)
    - Weighted model: Better calibrated
    """
    demonstration_7_classification_imbalance(imbalance_ratio=0.05)


def section_5():
    """
    SECTION 5 (BONUS): Bias-Variance in Residuals (~if time permits)
    =================================================================

    SPEAKER NOTE: Only cover this if you have extra time!

    This is more theoretical but connects everything together.

    Key equation:
        Observed Residuals = Irreducible Noise + Model Bias + Model Variance

    Where:
    - Irreducible Noise: y_true - f_true (can't be reduced with more data/complexity)
    - Model Bias: f_true - E[f_model] (systematic error from model assumptions)
    - Model Variance: E[f_model] - f_model (variability across training sets)

    Loss function affects this tradeoff:
    - MSE is sensitive to variance (outliers increase variance)
    - MAE is more robust (less variance sensitivity)

    The classic bias-variance tradeoff:
    - Low complexity: High bias, low variance
    - High complexity: Low bias, high variance
    - Sweet spot: Minimizes total error
    """
    demonstration_8_bias_variance_residuals()


def section_summary():
    """
    SECTION 6: Summary and Takeaways (~2 minutes)
    ==============================================

    SPEAKER NOTE: Reinforce the key messages.

    Main points:
    1. THREE distributions (input, function, residuals)
    2. Loss = Negative Log-Likelihood of assumed residual distribution
    3. Check your residuals! (patterns = bias, distribution mismatch = wrong loss)
    4. Imbalance requires weighted loss
    5. Practical workflow for choosing loss functions

    End with Q&A.
    """
    demonstration_summary()


def run_presentation(sections=None, interactive=False):
    """
    Run the full presentation or specific sections.

    Parameters
    ----------
    sections : list of int, optional
        Which sections to run (1-6). If None, run all.
    interactive : bool
        If True, pause after each section for discussion.
    """
    setup_presentation()

    all_sections = {
        1: ("The Three Distributions", section_1),
        2: ("MLE to MSE and MAE", lambda: (section_2a(), section_2b())),
        3: ("Loss Functions in Action", lambda: (section_3a(), section_3b(), section_3c())),
        4: ("Imbalanced Classification", section_4),
        5: ("Bias-Variance (Bonus)", section_5),
        6: ("Summary", section_summary),
    }

    if sections is None:
        sections = list(all_sections.keys())

    for section_num in sections:
        if section_num not in all_sections:
            print(f"Unknown section: {section_num}")
            continue

        name, func = all_sections[section_num]
        print(f"\n{'='*70}")
        print(f"  RUNNING SECTION {section_num}: {name}")
        print(f"{'='*70}\n")

        func()

        if interactive and section_num != max(sections):
            input("\n[Press Enter to continue to next section...]\n")

    print("\n" + "=" * 70)
    print("  PRESENTATION COMPLETE")
    print("  Thank you for your attention!")
    print("=" * 70 + "\n")


def main():
    """
    Main entry point with argument parsing.

    SPEAKER NOTE: You can run specific sections for practice:
        python main.py --section 1 2
        python main.py --interactive
    """
    parser = argparse.ArgumentParser(
        description="MLE and Loss Functions Presentation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py              # Run full presentation
    python main.py --section 1  # Run section 1 only
    python main.py --section 1 2 3  # Run sections 1, 2, and 3
    python main.py --interactive    # Pause after each section
    python main.py --list       # List all sections
        """,
    )
    parser.add_argument(
        "--section", "-s",
        type=int,
        nargs="+",
        help="Which sections to run (1-6)",
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Interactive mode: pause after each section",
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List all sections and exit",
    )

    args = parser.parse_args()

    if args.list:
        print("\nAvailable sections:")
        print("-" * 50)
        print("1. The Three Distributions (~8 min)")
        print("2. MLE to MSE and MAE (~10 min)")
        print("3. Loss Functions in Action (~12 min)")
        print("   3a. MSE vs MAE with Outliers")
        print("   3b. Model Misspecification")
        print("   3c. Real World Example")
        print("4. Imbalanced Classification (~6 min)")
        print("5. Bias-Variance in Residuals (Bonus)")
        print("6. Summary (~2 min)")
        print("-" * 50)
        print("\nTotal: ~40 minutes")
        return

    run_presentation(sections=args.section, interactive=args.interactive)


if __name__ == "__main__":
    main()
