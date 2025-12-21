"""
MLE Interactive Widget
Dark theme visualization for demonstrating Maximum Likelihood Estimation
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

# Dark theme colors (PyCharm Darcula-inspired)
COLORS = {
    'bg': '#1e1e1e',
    'paper': '#252526',
    'grid': '#3c3c3c',
    'text': '#d4d4d4',
    'blue': '#569cd6',
    'orange': '#ce9178',
    'purple': '#c586c0',
    'yellow': '#dcdcaa',
    'cyan': '#4ec9b0',
    'pink': '#ff79c6',
    'green': '#50fa7b',
    'red': '#ff5555',
}


def run_mle_widget(data, feature_name="Feature", height=700, width=None):
    """
    Launch interactive MLE widget.

    Parameters
    ----------
    data : array-like
        Sample data (1D array)
    feature_name : str
        Label for x-axis
    height : int
        Figure height in pixels (default 700)
    width : int or None
        Figure width in pixels (default None = full width)
    """
    from ipywidgets import FloatSlider, VBox, HTML
    from IPython.display import display

    data = np.asarray(data).flatten()
    n = len(data)
    sample_mean = float(np.mean(data))
    sample_std = float(np.std(data, ddof=0))
    data = data.tolist()

    # Slider ranges
    mu_min, mu_max = sample_mean - 3*sample_std, sample_mean + 3*sample_std
    sigma_min, sigma_max = max(0.5, sample_std * 0.3), sample_std * 2.5

    # Start away from optimal
    init_mu = sample_mean + sample_std * 0.8
    init_sigma = sample_std * 1.4

    # Build initial figure
    fig = _build_figure(data, init_mu, init_sigma, feature_name, sample_mean, sample_std, height, width)
    fig_widget = go.FigureWidget(fig)

    # Sliders
    mu_slider = FloatSlider(
        value=init_mu, min=mu_min, max=mu_max,
        step=(mu_max - mu_min) / 100,
        description='μ (mean):',
        style={'description_width': '80px'},
        readout_format='.2f',
        layout={'width': '450px'}
    )

    sigma_slider = FloatSlider(
        value=init_sigma, min=sigma_min, max=sigma_max,
        step=(sigma_max - sigma_min) / 100,
        description='σ (std):',
        style={'description_width': '80px'},
        readout_format='.2f',
        layout={'width': '450px'}
    )

    def on_slider_change(_):
        mu = mu_slider.value
        sigma = sigma_slider.value
        _update_figure(fig_widget, data, mu, sigma, sample_mean, sample_std)

    mu_slider.observe(on_slider_change, names='value')
    sigma_slider.observe(on_slider_change, names='value')

    header = HTML(f"""
    <div style="background:{COLORS['bg']}; padding:12px; border-radius:8px;
                border:1px solid {COLORS['blue']}; margin-bottom:10px; font-family:monospace;">
        <h3 style="color:#61dafb; margin:0 0 8px 0;">Maximum Likelihood Estimation</h3>
        <p style="color:{COLORS['text']}; margin:0;">
            <b>n = {n}</b> &nbsp;|&nbsp;
            Sample mean: <span style="color:{COLORS['cyan']}">{sample_mean:.3f}</span> &nbsp;|&nbsp;
            Sample std: <span style="color:{COLORS['purple']}">{sample_std:.3f}</span>
        </p>
    </div>
    """)

    display(VBox([header, VBox([mu_slider, sigma_slider]), fig_widget]))


def _log_likelihood(data, mu, sigma):
    """Compute log-likelihood of data under N(mu, sigma^2)"""
    if sigma <= 0:
        return float('-inf')
    n = len(data)
    ss = sum((x - mu)**2 for x in data)
    return -n/2 * np.log(2*np.pi) - n*np.log(sigma) - ss / (2*sigma**2)


def _build_figure(data, mu, sigma, feature_name, mle_mu, mle_sigma, height, width):
    """Create the initial plotly figure"""

    n = len(data)
    ll = _log_likelihood(data, mu, sigma)
    optimal_ll = _log_likelihood(data, mle_mu, mle_sigma)

    # Plot range
    data_min, data_max = min(data), max(data)
    margin = (data_max - data_min) * 0.3
    x_min, x_max = data_min - margin, data_max + margin

    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{"colspan": 2}, None], [{}, {}]],
        row_heights=[0.65, 0.35],
        subplot_titles=[f"{feature_name} Distribution", "Log-likelihood vs μ", "Log-likelihood vs σ"],
        vertical_spacing=0.12
    )

    # --- Top plot: PDF + data ---
    x_pdf = np.linspace(x_min, x_max, 200)
    y_pdf = stats.norm.pdf(x_pdf, mu, sigma)
    y_max = max(y_pdf) if max(y_pdf) > 0 else 1
    y_scaled = [y / y_max * 0.4 for y in y_pdf]

    # PDF curve
    fig.add_trace(go.Scatter(
        x=x_pdf.tolist(), y=y_scaled, mode='lines',
        name=f'N(μ={mu:.1f}, σ={sigma:.1f})',
        line=dict(color=COLORS['pink'], width=3),
        fill='tozeroy', fillcolor='rgba(255,121,198,0.15)'
    ), row=1, col=1)

    # Data points colored by likelihood (green=high, red=low)
    point_pdfs = [stats.norm.pdf(x, mu, sigma) for x in data]
    pdf_max = max(point_pdfs) if max(point_pdfs) > 0 else 1
    point_colors = [p / pdf_max for p in point_pdfs]

    fig.add_trace(go.Scatter(
        x=data, y=[0.02] * n,
        mode='markers', name='Data',
        marker=dict(
            size=12,
            color=point_colors,
            colorscale=[[0, COLORS['red']], [0.5, COLORS['yellow']], [1, COLORS['green']]],
            line=dict(color=COLORS['text'], width=1)
        ),
        hovertemplate='x = %{x:.2f}<extra></extra>'
    ), row=1, col=1)

    # Vertical mean line
    fig.add_trace(go.Scatter(
        x=[float(mu), float(mu)], y=[0, 0.42],
        mode='lines', name='μ',
        line=dict(color=COLORS['cyan'], width=2, dash='dash'),
        showlegend=False
    ), row=1, col=1)

    # --- Bottom left: LL vs mu ---
    mu_vals = np.linspace(mle_mu - 2.5*mle_sigma, mle_mu + 2.5*mle_sigma, 100)
    ll_mu = [_log_likelihood(data, m, sigma) for m in mu_vals]

    fig.add_trace(go.Scatter(
        x=mu_vals.tolist(), y=ll_mu, mode='lines',
        line=dict(color=COLORS['blue'], width=2), showlegend=False
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=[float(mu)], y=[float(ll)], mode='markers',
        marker=dict(size=12, color=COLORS['orange'], symbol='diamond'),
        showlegend=False
    ), row=2, col=1)

    # --- Bottom right: LL vs sigma ---
    sigma_vals = np.linspace(max(0.5, mle_sigma * 0.3), mle_sigma * 2.5, 100)
    ll_sigma = [_log_likelihood(data, mu, s) for s in sigma_vals]

    fig.add_trace(go.Scatter(
        x=sigma_vals.tolist(), y=ll_sigma, mode='lines',
        line=dict(color=COLORS['purple'], width=2), showlegend=False
    ), row=2, col=2)

    fig.add_trace(go.Scatter(
        x=[float(sigma)], y=[float(ll)], mode='markers',
        marker=dict(size=12, color=COLORS['orange'], symbol='diamond'),
        showlegend=False
    ), row=2, col=2)

    # Status
    ratio = np.exp(ll - optimal_ll) if optimal_ll > float('-inf') else 0
    status = "OPTIMAL!" if ratio > 0.99 else "Good" if ratio > 0.7 else "Keep trying..."
    status_color = COLORS['green'] if ratio > 0.7 else COLORS['yellow'] if ratio > 0.3 else COLORS['red']

    layout_kwargs = dict(
        template='plotly_dark',
        paper_bgcolor=COLORS['bg'],
        plot_bgcolor=COLORS['paper'],
        font=dict(family='JetBrains Mono, Consolas, monospace', color=COLORS['text']),
        title=dict(
            text=f"<b>Log-Likelihood: {ll:.2f}</b> | <span style='color:{status_color}'>{status}</span>",
            x=0.5, font=dict(size=16)
        ),
        height=height,
        showlegend=True,
        legend=dict(x=0.85, y=0.98, bgcolor='rgba(0,0,0,0)'),
        margin=dict(t=60, b=40, l=50, r=30)
    )
    if width is not None:
        layout_kwargs['width'] = width
    fig.update_layout(**layout_kwargs) # pyright: ignore[reportArgumentType]

    # Axis styling
    for row, col in [(1, 1), (2, 1), (2, 2)]:
        fig.update_xaxes(gridcolor=COLORS['grid'], row=row, col=col)
        fig.update_yaxes(gridcolor=COLORS['grid'], row=row, col=col)

    fig.update_xaxes(title_text=feature_name, row=1, col=1)
    fig.update_yaxes(title_text="Density (scaled)", row=1, col=1)
    fig.update_xaxes(title_text="μ", row=2, col=1)
    fig.update_yaxes(title_text="ℓ(μ)", row=2, col=1)
    fig.update_xaxes(title_text="σ", row=2, col=2)
    fig.update_yaxes(title_text="ℓ(σ)", row=2, col=2)

    for ann in fig['layout']['annotations']:
        ann['font'] = dict(size=12, color=COLORS['yellow']) # pyright: ignore[reportIndexIssue]

    return fig


def _update_figure(fig_widget, data, mu, sigma, mle_mu, mle_sigma):
    """Update figure widget when sliders change"""

    n = len(data)
    ll = _log_likelihood(data, mu, sigma)
    optimal_ll = _log_likelihood(data, mle_mu, mle_sigma)

    # Precompute
    data_min, data_max = min(data), max(data)
    margin = (data_max - data_min) * 0.3
    x_pdf = np.linspace(data_min - margin, data_max + margin, 200)
    y_pdf = stats.norm.pdf(x_pdf, mu, sigma)
    y_max = max(y_pdf) if max(y_pdf) > 0 else 1
    y_scaled = [y / y_max * 0.4 for y in y_pdf]

    # Point colors
    point_pdfs = [stats.norm.pdf(x, mu, sigma) for x in data]
    pdf_max = max(point_pdfs) if max(point_pdfs) > 0 else 1
    point_colors = [p / pdf_max for p in point_pdfs]

    # LL curves
    mu_vals = np.linspace(mle_mu - 2.5*mle_sigma, mle_mu + 2.5*mle_sigma, 100)
    ll_mu = [_log_likelihood(data, m, sigma) for m in mu_vals]

    sigma_vals = np.linspace(max(0.5, mle_sigma * 0.3), mle_sigma * 2.5, 100)
    ll_sigma = [_log_likelihood(data, mu, s) for s in sigma_vals]

    # Status
    ratio = np.exp(ll - optimal_ll) if optimal_ll > float('-inf') else 0
    status = "OPTIMAL!" if ratio > 0.99 else "Good" if ratio > 0.7 else "Keep trying..."
    status_color = COLORS['green'] if ratio > 0.7 else COLORS['yellow'] if ratio > 0.3 else COLORS['red']

    with fig_widget.batch_update():
        # Trace 0: PDF curve
        fig_widget.data[0].x = x_pdf.tolist()
        fig_widget.data[0].y = y_scaled
        fig_widget.data[0].name = f'N(μ={mu:.1f}, σ={sigma:.1f})'

        # Trace 1: Data points (update colors)
        fig_widget.data[1].marker.color = point_colors

        # Trace 2: Mean vertical line
        fig_widget.data[2].x = [float(mu), float(mu)]

        # Trace 3: LL vs mu curve
        fig_widget.data[3].x = mu_vals.tolist()
        fig_widget.data[3].y = ll_mu

        # Trace 4: Current mu marker
        fig_widget.data[4].x = [float(mu)]
        fig_widget.data[4].y = [float(ll)]

        # Trace 5: LL vs sigma curve
        fig_widget.data[5].x = sigma_vals.tolist()
        fig_widget.data[5].y = ll_sigma

        # Trace 6: Current sigma marker
        fig_widget.data[6].x = [float(sigma)]
        fig_widget.data[6].y = [float(ll)]

        # Title
        fig_widget.layout.title.text = f"<b>Log-Likelihood: {ll:.2f}</b> | <span style='color:{status_color}'>{status}</span>"
