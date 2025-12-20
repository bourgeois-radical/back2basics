"""
Utility functions for MLE and Loss Functions presentation.

This module provides helper utilities for consistent styling, formatting,
and display throughout the presentation.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def set_plot_style(
    style: str = "seaborn-v0_8-darkgrid",
    context: str = "talk",
    font_scale: float = 1.2,
) -> None:
    """
    Set consistent plot styling for presentation.

    Parameters
    ----------
    style : str
        Matplotlib style
    context : str
        Seaborn context ('paper', 'notebook', 'talk', 'poster')
    font_scale : float
        Scale factor for fonts

    Side Effects
    ------------
    Updates matplotlib/seaborn global settings.

    Notes
    -----
    Call this at the beginning of the notebook.
    Ensures all plots are readable in presentation.
    """
    try:
        plt.style.use(style)
    except OSError:
        # Fallback if style not available
        plt.style.use("seaborn-v0_8" if "seaborn" in style else "ggplot")

    sns.set_context(context, font_scale=font_scale)

    # Additional settings for presentation
    plt.rcParams.update(
        {
            "figure.figsize": (12, 6),
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.dpi": 100,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


def create_latex_equation(equation_type: str) -> str:
    """
    Generate LaTeX strings for key equations.

    Parameters
    ----------
    equation_type : str
        'gaussian_likelihood', 'laplace_likelihood',
        'mse_loss', 'mae_loss', 'cross_entropy', etc.

    Returns
    -------
    latex_str : str
        LaTeX formatted equation

    Notes
    -----
    Use with IPython.display.Math for rendering.
    """
    equations = {
        "gaussian_likelihood": r"p(y_i | \hat{y}_i, \sigma) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(y_i - \hat{y}_i)^2}{2\sigma^2}\right)",
        "gaussian_log_likelihood": r"\log L = -\frac{n}{2}\log(2\pi\sigma^2) - \frac{1}{2\sigma^2}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2",
        "gaussian_nll": r"\text{NLL} \propto \sum_{i=1}^{n}(y_i - \hat{y}_i)^2 = \text{MSE} \times n",
        "laplace_likelihood": r"p(y_i | \hat{y}_i, b) = \frac{1}{2b} \exp\left(-\frac{|y_i - \hat{y}_i|}{b}\right)",
        "laplace_log_likelihood": r"\log L = -n\log(2b) - \frac{1}{b}\sum_{i=1}^{n}|y_i - \hat{y}_i|",
        "laplace_nll": r"\text{NLL} \propto \sum_{i=1}^{n}|y_i - \hat{y}_i| = \text{MAE} \times n",
        "mse_loss": r"\mathcal{L}_{\text{MSE}} = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2",
        "mae_loss": r"\mathcal{L}_{\text{MAE}} = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|",
        "huber_loss": r"\mathcal{L}_{\delta}(r) = \begin{cases} \frac{1}{2}r^2 & \text{if } |r| \leq \delta \\ \delta(|r| - \frac{1}{2}\delta) & \text{otherwise} \end{cases}",
        "cross_entropy": r"\mathcal{L}_{\text{CE}} = -\frac{1}{n}\sum_{i=1}^{n}\left[y_i \log(\hat{p}_i) + (1-y_i)\log(1-\hat{p}_i)\right]",
        "mle_principle": r"\hat{\theta}_{\text{MLE}} = \arg\max_{\theta} \prod_{i=1}^{n} p(y_i | x_i, \theta)",
        "nll_principle": r"\hat{\theta}_{\text{MLE}} = \arg\min_{\theta} \left[-\sum_{i=1}^{n} \log p(y_i | x_i, \theta)\right]",
    }

    if equation_type not in equations:
        available = ", ".join(equations.keys())
        raise ValueError(
            f"Unknown equation type: {equation_type}. Available: {available}"
        )

    return equations[equation_type]


def print_section_header(
    section_number: int, title: str, description: str = ""
) -> None:
    """
    Print formatted section header for notebook.

    Parameters
    ----------
    section_number : int
        Section number
    title : str
        Section title
    description : str
        Optional description

    Side Effects
    ------------
    Prints formatted header with separators.
    """
    width = 70
    print("\n" + "=" * width)
    print(f"  SECTION {section_number}: {title.upper()}")
    if description:
        print(f"  {description}")
    print("=" * width + "\n")


def save_figure(
    fig: plt.Figure,
    filename: str,
    output_dir: str = "./figures",
    dpi: int = 300,
) -> None:
    """
    Save figure to file.

    Parameters
    ----------
    fig : Figure
        Matplotlib figure
    filename : str
        Output filename
    output_dir : str
        Directory to save in
    dpi : int
        Resolution

    Side Effects
    ------------
    Creates file on disk.

    Notes
    -----
    Useful for extracting figures for slides separately.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    filepath = output_path / filename
    fig.savefig(filepath, dpi=dpi, bbox_inches="tight", facecolor="white")
    print(f"Figure saved to: {filepath}")


def create_summary_table(
    results: dict,
    metric_names: list[str],
    highlight_best: bool = True,
) -> pd.DataFrame:
    """
    Create formatted summary table.

    Parameters
    ----------
    results : dict
        Model results dictionary with structure:
        {'model_name': {'metrics': {'metric1': value, ...}}}
    metric_names : list
        Which metrics to include
    highlight_best : bool
        Whether to highlight best values

    Returns
    -------
    df : pd.DataFrame
        Formatted dataframe for display

    Notes
    -----
    Applies formatting:
    - Highlight best values
    - Format decimals
    """
    # Build data for dataframe
    data = {}
    for model_name, model_results in results.items():
        if "metrics" in model_results or "test_metrics" in model_results:
            metrics = model_results.get("test_metrics", model_results.get("metrics", {}))
            data[model_name] = {
                metric: metrics.get(metric, float("nan")) for metric in metric_names
            }

    df = pd.DataFrame(data).T
    df = df.round(4)

    return df


def format_metric_value(value: float, metric_name: str) -> str:
    """
    Format a metric value for display.

    Parameters
    ----------
    value : float
        Metric value
    metric_name : str
        Name of the metric (affects formatting)

    Returns
    -------
    str
        Formatted string
    """
    if metric_name in ["r2", "accuracy", "precision", "recall", "f1", "auc_roc"]:
        return f"{value:.4f}"
    elif metric_name in ["mse", "rmse", "mae"]:
        return f"{value:.4f}"
    else:
        return f"{value:.4f}"


def print_key_insight(text: str) -> None:
    """
    Print a key insight in a formatted box.

    Parameters
    ----------
    text : str
        The insight text to display
    """
    lines = text.strip().split("\n")
    max_len = max(len(line) for line in lines)
    width = max_len + 4

    print("\n" + "-" * width)
    for line in lines:
        print(f"  {line}")
    print("-" * width + "\n")


def print_speaker_note(note: str) -> None:
    """
    Print a speaker note (for presenter reference).

    Parameters
    ----------
    note : str
        The speaker note text
    """
    print(f"\n[SPEAKER NOTE]: {note}\n")
