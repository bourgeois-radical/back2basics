"""
Residual Distribution Visualization.
The key insight: loss functions assume distributions on RESIDUALS, not features.

MSE  → Normal residuals
MAE  → Laplace residuals
Huber → Hybrid
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from src.theme import COLORS, apply_dark_theme


def plot_residual_distributions(experiment, show_theoretical=True):
    """
    Plot residual distributions for all trained models.
    Overlays theoretical Normal and Laplace for comparison.
    
    Parameters
    ----------
    experiment : RegressionExperiment
        Trained experiment object with residuals
    show_theoretical : bool
        Whether to overlay theoretical PDFs
    """
    
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('MSE (assumes Normal)', 'MAE (assumes Laplace)', 'Huber (hybrid)')
    )
    
    model_colors = {
        'MSE': COLORS['blue'],
        'MAE': COLORS['purple'],
        'Huber': COLORS['orange']
    }
    
    for i, name in enumerate(['MSE', 'MAE', 'Huber'], 1):
        if name not in experiment.residuals:
            continue
            
        resid = experiment.residuals[name]
        color = model_colors[name]
        
        # Histogram of residuals
        fig.add_trace(go.Histogram(
            x=resid,
            nbinsx=15,
            name=f'{name} residuals',
            marker=dict(color=color, opacity=0.7, line=dict(color=COLORS['text'], width=1)),
            histnorm='probability density',
            showlegend=False
        ), row=1, col=i)
        
        if show_theoretical:
            x_range = np.linspace(resid.min() - 5, resid.max() + 5, 200)
            
            # Fit and plot Normal
            mu, sigma = resid.mean(), resid.std()
            normal_pdf = stats.norm.pdf(x_range, mu, sigma)
            fig.add_trace(go.Scatter(
                x=x_range, y=normal_pdf,
                mode='lines',
                name='Normal fit',
                line=dict(color=COLORS['cyan'], width=2, dash='dash'),
                showlegend=(i == 1)
            ), row=1, col=i)
            
            # Fit and plot Laplace
            b = np.mean(np.abs(resid - np.median(resid)))  # Laplace scale parameter
            laplace_pdf = stats.laplace.pdf(x_range, loc=np.median(resid), scale=b)
            fig.add_trace(go.Scatter(
                x=x_range, y=laplace_pdf,
                mode='lines',
                name='Laplace fit',
                line=dict(color=COLORS['pink'], width=2, dash='dot'),
                showlegend=(i == 1)
            ), row=1, col=i)
    
    apply_dark_theme(fig, title="📊 Residual Distributions: What Does Your Loss Assume?", height=450)
    
    fig.update_xaxes(title_text='Residual (y - ŷ)')
    fig.update_yaxes(title_text='Density', col=1)
    
    # Style subplot titles
    for ann in fig['layout']['annotations']:
        ann['font'] = dict(size=13, color=COLORS['yellow'])
    
    return fig


def plot_residuals_vs_predicted(experiment):
    """
    Residuals vs predicted values - diagnostic plot.
    Good model: random scatter around zero.
    """
    
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('MSE', 'MAE', 'Huber')
    )
    
    model_colors = {
        'MSE': COLORS['blue'],
        'MAE': COLORS['purple'],
        'Huber': COLORS['orange']
    }
    
    for i, name in enumerate(['MSE', 'MAE', 'Huber'], 1):
        if name not in experiment.residuals:
            continue
        
        pred = experiment.predictions[name]
        resid = experiment.residuals[name]
        color = model_colors[name]
        
        fig.add_trace(go.Scatter(
            x=pred, y=resid,
            mode='markers',
            marker=dict(size=10, color=color, opacity=0.7, line=dict(color=COLORS['text'], width=1)),
            name=name,
            showlegend=False
        ), row=1, col=i)
        
        # Zero line
        fig.add_hline(y=0, line=dict(color=COLORS['red'], width=2, dash='dash'), row=1, col=i)
    
    apply_dark_theme(fig, title="📊 Residuals vs Predicted", height=400)
    
    fig.update_xaxes(title_text='Predicted')
    fig.update_yaxes(title_text='Residual', col=1)
    
    for ann in fig['layout']['annotations']:
        ann['font'] = dict(size=13, color=COLORS['yellow'])
    
    return fig


def plot_loss_landscape_1d():
    """
    Show how MSE vs MAE behave differently with outliers.
    1D toy example for intuition.
    """
    
    # Simple data: most points near 0, one outlier
    data_clean = np.array([-1, -0.5, 0, 0.5, 1])
    data_outlier = np.array([-1, -0.5, 0, 0.5, 10])  # outlier at 10
    
    # Range of possible predictions
    pred_range = np.linspace(-2, 12, 200)
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Clean Data', 'Data with Outlier (y=10)')
    )
    
    for col, data in enumerate([data_clean, data_outlier], 1):
        # MSE loss landscape
        mse_loss = [np.mean((data - p) ** 2) for p in pred_range]
        fig.add_trace(go.Scatter(
            x=pred_range, y=mse_loss,
            mode='lines',
            name='MSE',
            line=dict(color=COLORS['blue'], width=3),
            showlegend=(col == 1)
        ), row=1, col=col)
        
        # MAE loss landscape
        mae_loss = [np.mean(np.abs(data - p)) for p in pred_range]
        fig.add_trace(go.Scatter(
            x=pred_range, y=mae_loss,
            mode='lines',
            name='MAE',
            line=dict(color=COLORS['purple'], width=3),
            showlegend=(col == 1)
        ), row=1, col=col)
        
        # Mark optimal points
        mse_opt = pred_range[np.argmin(mse_loss)]
        mae_opt = pred_range[np.argmin(mae_loss)]
        
        fig.add_trace(go.Scatter(
            x=[mse_opt], y=[min(mse_loss)],
            mode='markers',
            marker=dict(size=15, color=COLORS['blue'], symbol='star'),
            name=f'MSE opt: {mse_opt:.1f}',
            showlegend=False
        ), row=1, col=col)
        
        fig.add_trace(go.Scatter(
            x=[mae_opt], y=[min(mae_loss)],
            mode='markers',
            marker=dict(size=15, color=COLORS['purple'], symbol='star'),
            name=f'MAE opt: {mae_opt:.1f}',
            showlegend=False
        ), row=1, col=col)
        
        # Annotate optimal values
        fig.add_annotation(
            x=mse_opt, y=min(mse_loss) + 2,
            text=f"MSE→{mse_opt:.1f}",
            showarrow=False,
            font=dict(color=COLORS['blue'], size=12),
            row=1, col=col
        )
        fig.add_annotation(
            x=mae_opt, y=min(mae_loss) + 1,
            text=f"MAE→{mae_opt:.1f}",
            showarrow=False,
            font=dict(color=COLORS['purple'], size=12),
            row=1, col=col
        )
    
    apply_dark_theme(fig, title="📊 Loss Landscape: MSE vs MAE", height=450)
    
    fig.update_xaxes(title_text='Prediction')
    fig.update_yaxes(title_text='Loss', col=1)
    
    for ann in fig['layout']['annotations']:
        if 'subplot' not in str(ann):
            continue
        ann['font'] = dict(size=13, color=COLORS['yellow'])
    
    return fig


def plot_normal_vs_laplace():
    """
    Compare Normal and Laplace distributions.
    Shows why Laplace (MAE) is robust: heavier tails expect outliers.
    """
    
    x = np.linspace(-6, 6, 300)
    
    # Both with same variance for fair comparison
    sigma = 1.5
    b = sigma / np.sqrt(2)  # Laplace scale for same variance
    
    normal_pdf = stats.norm.pdf(x, 0, sigma)
    laplace_pdf = stats.laplace.pdf(x, 0, b)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=x, y=normal_pdf,
        mode='lines',
        name='Normal (MSE assumes this)',
        line=dict(color=COLORS['cyan'], width=3),
        fill='tozeroy',
        fillcolor='rgba(86, 156, 214, 0.2)'
    ))
    
    fig.add_trace(go.Scatter(
        x=x, y=laplace_pdf,
        mode='lines',
        name='Laplace (MAE assumes this)',
        line=dict(color=COLORS['pink'], width=3),
        fill='tozeroy',
        fillcolor='rgba(255, 121, 198, 0.2)'
    ))
    
    apply_dark_theme(fig, title="📊 Normal vs Laplace: Which Residual Distribution?", height=450)
    
    fig.update_xaxes(title_text='Residual value')
    fig.update_yaxes(title_text='Density')
    
    # Add annotations
    fig.add_annotation(
        x=3.5, y=0.08,
        text="Laplace has<br>heavier tails",
        showarrow=True,
        arrowhead=2,
        arrowcolor=COLORS['pink'],
        font=dict(color=COLORS['pink'], size=12),
        ax=-40, ay=-40
    )
    
    fig.add_annotation(
        x=0, y=0.35,
        text="Normal is<br>more peaked",
        showarrow=True,
        arrowhead=2,
        arrowcolor=COLORS['cyan'],
        font=dict(color=COLORS['cyan'], size=12),
        ax=60, ay=-30
    )
    
    return fig


def plot_huber_loss_function():
    """
    Show the Huber loss function: quadratic near zero, linear in tails.
    """
    
    x = np.linspace(-5, 5, 300)
    delta = 1.35  # Standard Huber threshold
    
    # Loss functions
    mse = x ** 2
    mae = np.abs(x)
    huber = np.where(np.abs(x) <= delta,
                     0.5 * x ** 2,
                     delta * np.abs(x) - 0.5 * delta ** 2)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=x, y=mse,
        mode='lines',
        name='MSE (quadratic)',
        line=dict(color=COLORS['blue'], width=2, dash='dash')
    ))
    
    fig.add_trace(go.Scatter(
        x=x, y=mae,
        mode='lines',
        name='MAE (linear)',
        line=dict(color=COLORS['purple'], width=2, dash='dot')
    ))
    
    fig.add_trace(go.Scatter(
        x=x, y=huber,
        mode='lines',
        name=f'Huber (δ={delta})',
        line=dict(color=COLORS['orange'], width=3)
    ))
    
    # Mark the transition point
    fig.add_vline(x=delta, line=dict(color=COLORS['grid'], width=1, dash='dash'))
    fig.add_vline(x=-delta, line=dict(color=COLORS['grid'], width=1, dash='dash'))
    
    fig.add_annotation(
        x=delta, y=8,
        text=f"δ = {delta}",
        showarrow=False,
        font=dict(color=COLORS['text'], size=11)
    )
    
    apply_dark_theme(fig, title="📊 Loss Functions: MSE vs MAE vs Huber", height=450)
    
    fig.update_xaxes(title_text='Residual (y - ŷ)', range=[-5, 5])
    fig.update_yaxes(title_text='Loss', range=[0, 10])
    
    return fig