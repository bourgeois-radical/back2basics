"""
Gradient Descent Visualization.
Show how optimization finds the minimum of different loss surfaces.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from src.theme import COLORS, apply_dark_theme


def gradient_descent_demo(loss_type='mse', learning_rate=0.1, n_steps=20, seed=42):
    """
    Run gradient descent on a simple 1D problem.
    
    Parameters
    ----------
    loss_type : str
        'mse' or 'mae'
    learning_rate : float
        Step size
    n_steps : int
        Number of iterations
    
    Returns
    -------
    history : dict
        {'theta': [...], 'loss': [...]}
    """
    np.random.seed(seed)
    
    # Simple data: y = 2x + noise
    X = np.linspace(0, 10, 20)
    y = 2 * X + np.random.normal(0, 2, 20)
    
    # Start with wrong parameter
    theta = 0.5
    history = {'theta': [theta], 'loss': []}
    
    for step in range(n_steps):
        # Predictions
        pred = theta * X
        residuals = y - pred
        
        # Compute loss
        if loss_type == 'mse':
            loss = np.mean(residuals ** 2)
            # Gradient of MSE: -2 * mean(residuals * X)
            grad = -2 * np.mean(residuals * X)
        else:  # mae
            loss = np.mean(np.abs(residuals))
            # Subgradient of MAE: -mean(sign(residuals) * X)
            grad = -np.mean(np.sign(residuals) * X)
        
        history['loss'].append(loss)
        
        # Update
        theta = theta - learning_rate * grad
        history['theta'].append(theta)
    
    # Final loss
    pred = theta * X
    residuals = y - pred
    final_loss = np.mean(residuals ** 2) if loss_type == 'mse' else np.mean(np.abs(residuals))
    history['loss'].append(final_loss)
    
    return history, X, y


def plot_gradient_descent_comparison():
    """
    Compare gradient descent on MSE vs MAE loss.
    Shows convergence behavior.
    """
    
    history_mse, X, y = gradient_descent_demo('mse', learning_rate=0.01, n_steps=30)
    history_mae, _, _ = gradient_descent_demo('mae', learning_rate=0.05, n_steps=30)
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Parameter θ over iterations', 'Loss over iterations')
    )
    
    steps = list(range(len(history_mse['theta'])))
    
    # Theta trajectory
    fig.add_trace(go.Scatter(
        x=steps, y=history_mse['theta'],
        mode='lines+markers',
        name='MSE',
        line=dict(color=COLORS['blue'], width=2),
        marker=dict(size=6)
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=steps, y=history_mae['theta'],
        mode='lines+markers',
        name='MAE',
        line=dict(color=COLORS['purple'], width=2),
        marker=dict(size=6)
    ), row=1, col=1)
    
    # True theta
    fig.add_hline(y=2.0, line=dict(color=COLORS['green'], width=2, dash='dash'),
                  annotation_text="True θ = 2", row=1, col=1)
    
    # Loss trajectory
    fig.add_trace(go.Scatter(
        x=steps, y=history_mse['loss'],
        mode='lines+markers',
        name='MSE loss',
        line=dict(color=COLORS['blue'], width=2),
        marker=dict(size=6),
        showlegend=False
    ), row=1, col=2)
    
    fig.add_trace(go.Scatter(
        x=steps, y=history_mae['loss'],
        mode='lines+markers',
        name='MAE loss',
        line=dict(color=COLORS['purple'], width=2),
        marker=dict(size=6),
        showlegend=False
    ), row=1, col=2)
    
    apply_dark_theme(fig, title="📉 Gradient Descent: Finding the Minimum", height=450)
    
    fig.update_xaxes(title_text='Iteration', row=1, col=1)
    fig.update_xaxes(title_text='Iteration', row=1, col=2)
    fig.update_yaxes(title_text='θ', row=1, col=1)
    fig.update_yaxes(title_text='Loss', row=1, col=2)
    
    for ann in fig['layout']['annotations']:
        if hasattr(ann, 'font'):
            ann['font'] = dict(size=13, color=COLORS['yellow'])
    
    return fig


def plot_loss_surface_2d():
    """
    2D loss surface showing MSE and MAE.
    Parameters: intercept (β₀) and slope (β₁).
    """
    
    # Simple data
    np.random.seed(42)
    X = np.linspace(0, 10, 30)
    y = 3 + 2 * X + np.random.normal(0, 2, 30)
    
    # Parameter grid
    beta0_range = np.linspace(-2, 8, 50)
    beta1_range = np.linspace(0, 4, 50)
    B0, B1 = np.meshgrid(beta0_range, beta1_range)
    
    # Compute MSE loss surface
    MSE_surface = np.zeros_like(B0)
    for i in range(B0.shape[0]):
        for j in range(B0.shape[1]):
            pred = B0[i, j] + B1[i, j] * X
            MSE_surface[i, j] = np.mean((y - pred) ** 2)
    
    fig = go.Figure()
    
    # Loss surface
    fig.add_trace(go.Surface(
        x=beta0_range,
        y=beta1_range,
        z=MSE_surface,
        colorscale=[[0, COLORS['blue']], [0.5, COLORS['purple']], [1, COLORS['pink']]],
        opacity=0.8,
        name='MSE Loss Surface',
        showscale=True,
        colorbar=dict(
            title='Loss',
            tickfont=dict(color=COLORS['text']),
            titlefont=dict(color=COLORS['text'])
        )
    ))
    
    # Mark minimum
    min_idx = np.unravel_index(np.argmin(MSE_surface), MSE_surface.shape)
    opt_b0 = beta0_range[min_idx[1]]
    opt_b1 = beta1_range[min_idx[0]]
    opt_loss = MSE_surface[min_idx]
    
    fig.add_trace(go.Scatter3d(
        x=[opt_b0], y=[opt_b1], z=[opt_loss],
        mode='markers',
        marker=dict(size=10, color=COLORS['green'], symbol='diamond'),
        name=f'Minimum (β₀={opt_b0:.1f}, β₁={opt_b1:.1f})'
    ))
    
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor=COLORS['bg'],
        font=dict(family='JetBrains Mono, Consolas, monospace', color=COLORS['text']),
        title=dict(text="📊 MSE Loss Surface", x=0.5, font=dict(size=18)),
        height=600,
        scene=dict(
            xaxis=dict(title='β₀ (intercept)', backgroundcolor=COLORS['paper'], gridcolor=COLORS['grid']),
            yaxis=dict(title='β₁ (slope)', backgroundcolor=COLORS['paper'], gridcolor=COLORS['grid']),
            zaxis=dict(title='Loss', backgroundcolor=COLORS['paper'], gridcolor=COLORS['grid']),
            bgcolor=COLORS['bg']
        )
    )
    
    return fig


def plot_gradient_intuition():
    """
    Simple visualization showing what gradient means.
    """
    
    # Quadratic loss
    theta = np.linspace(-3, 5, 200)
    loss = (theta - 1) ** 2  # Minimum at theta=1
    
    fig = go.Figure()
    
    # Loss curve
    fig.add_trace(go.Scatter(
        x=theta, y=loss,
        mode='lines',
        name='Loss L(θ)',
        line=dict(color=COLORS['blue'], width=3)
    ))
    
    # Show gradient at different points
    points = [-1, 1, 3]
    for p in points:
        loss_at_p = (p - 1) ** 2
        grad_at_p = 2 * (p - 1)  # Derivative
        
        # Tangent line
        x_tang = np.linspace(p - 1.5, p + 1.5, 50)
        y_tang = loss_at_p + grad_at_p * (x_tang - p)
        
        color = COLORS['green'] if p == 1 else (COLORS['red'] if grad_at_p > 0 else COLORS['cyan'])
        
        fig.add_trace(go.Scatter(
            x=x_tang, y=y_tang,
            mode='lines',
            name=f'Gradient at θ={p}: {grad_at_p:+.1f}',
            line=dict(color=color, width=2, dash='dash')
        ))
        
        fig.add_trace(go.Scatter(
            x=[p], y=[loss_at_p],
            mode='markers',
            marker=dict(size=12, color=color),
            showlegend=False
        ))
        
        # Arrow showing direction to move
        if p != 1:
            arrow_dir = -np.sign(grad_at_p) * 0.8
            fig.add_annotation(
                x=p, y=loss_at_p + 1,
                ax=p + arrow_dir, ay=loss_at_p + 1,
                xref='x', yref='y',
                axref='x', ayref='y',
                showarrow=True,
                arrowhead=2,
                arrowsize=1.5,
                arrowwidth=2,
                arrowcolor=color
            )
    
    apply_dark_theme(fig, title="📉 Gradient Points Downhill", height=450)
    
    fig.update_xaxes(title_text='Parameter θ')
    fig.update_yaxes(title_text='Loss L(θ)')
    
    fig.add_annotation(
        x=1, y=-1,
        text="Minimum: gradient = 0",
        showarrow=True,
        arrowhead=2,
        arrowcolor=COLORS['green'],
        font=dict(color=COLORS['green'], size=12),
        ay=-40
    )
    
    return fig