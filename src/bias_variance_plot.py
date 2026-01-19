"""
bias_variance_plot
------------------
Generate a 2×2 bias–variance visualization panel.

Usage:
    from bias_variance_plot import plot_bias_variance
    plot_bias_variance(COLORS)
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_bias_variance(COLORS, seed=42, n_points=25, figsize=(14, 14)):
    """Plot the classic 2×2 bias/variance visualization.

    COLORS: dict with keys bg, paper, cyan, text, blue, red, yellow
    seed:   int, random seed
    n_points: points per quadrant
    figsize: figure size
    """
    # Set random seed for reproducibility
    np.random.seed(seed)

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.patch.set_facecolor(COLORS['bg'])

    # Helper: draw concentric circles (bullseye)
    def draw_bullseye(ax):
        circles = [3, 2, 1]
        for radius in circles:
            circle = plt.Circle(
                (0, 0), radius,
                color=COLORS['cyan'],
                fill=True,
                alpha=0.15,
                linewidth=2,
                edgecolor=COLORS['text'],
                zorder=1
            )
            ax.add_patch(circle)

    # Helper: generate scattered prediction points
    def generate_predictions(bias_offset, variance_scale):
        angles = np.random.uniform(0, 2 * np.pi, n_points)
        distances = np.random.rayleigh(variance_scale, n_points)
        x = bias_offset[0] + distances * np.cos(angles)
        y = bias_offset[1] + distances * np.sin(angles)
        return x, y

    # Quadrant configs: (bias_x, bias_y, variance)
    configs = {
        (0, 0): (0, 0, 0.15),
        (0, 1): (0, 0, 0.6),
        (1, 0): (1.2, 0.8, 0.15),
        (1, 1): (1.2, 0.8, 0.6),
    }

    titles = {
        (0, 0): 'Low Variance, Low Bias',
        (0, 1): 'High Variance, Low Bias',
        (1, 0): 'Low Variance, High Bias',
        (1, 1): 'High Variance, High Bias',
    }

    # Draw quadrants
    for i in range(2):
        for j in range(2):
            ax = axes[i, j]
            ax.set_facecolor(COLORS['paper'])
            draw_bullseye(ax)

            bias_x, bias_y, variance = configs[(i, j)]
            x_pred, y_pred = generate_predictions((bias_x, bias_y), variance)

            # Predictions
            ax.scatter(
                x_pred, y_pred,
                c=COLORS['blue'],
                s=100,
                alpha=0.8,
                edgecolors=COLORS['text'],
                linewidth=1.5,
                zorder=3
            )

            # True target
            ax.scatter(
                0, 0,
                c=COLORS['red'],
                s=300,
                edgecolors=COLORS['text'],
                linewidth=2,
                zorder=4
            )

            # Average prediction
            avg_x, avg_y = np.mean(x_pred), np.mean(y_pred)
            ax.scatter(
                avg_x, avg_y,
                c=COLORS['blue'],
                s=400,
                alpha=0.5,
                edgecolors=COLORS['yellow'],
                linewidth=3,
                marker='o',
                zorder=2
            )

            # Formatting
            ax.set_xlim(-3.5, 3.5)
            ax.set_ylim(-3.5, 3.5)
            ax.set_aspect('equal')
            ax.axis('off')
            ax.text(
                0, -3.8,
                titles[(i, j)],
                fontsize=18,
                fontweight='bold',
                color=COLORS['text'],
                ha='center',
                va='center'
            )
            ax.grid(False)

    plt.tight_layout()
    plt.show()
