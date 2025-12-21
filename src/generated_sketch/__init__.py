"""
MLE and Loss Functions Presentation Package.

This package contains all modules for the presentation on
Understanding MLE and Loss Functions Through Residual Distributions.

Modules:
    - data_generation: Synthetic data with known properties
    - visualization: All presentation visualizations
    - models: Train models with different loss functions
    - metrics: Calculate and compare metrics
    - theory: MLE derivations and theoretical demonstrations
    - demonstrations: High-level demo functions for each section
    - utils: Helper utilities for formatting and styling
    - main: Orchestration and presentation flow
"""

from . import (
    data_generation,
    demonstrations,
    metrics,
    models,
    theory,
    utils,
    visualization,
)

__all__ = [
    "data_generation",
    "demonstrations",
    "metrics",
    "models",
    "theory",
    "utils",
    "visualization",
]
