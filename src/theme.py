"""
Shared theme configuration for Back to Basics presentation.
All visualizations use this consistent dark theme.
"""

# Dark theme colors (PyCharm Darcula-inspired)
COLORS = {
    'bg': '#1e1e1e',
    'paper': '#1e1e1e',
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
    'white': '#ffffff',
}

def apply_dark_theme(fig, title=None, height=600):
    """Apply consistent dark theme to any plotly figure."""
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor=COLORS['bg'],
        plot_bgcolor=COLORS['paper'],
        font=dict(
            family='JetBrains Mono, Consolas, monospace',
            color=COLORS['text']
        ),
        title=dict(
            text=title,
            x=0.5,
            font=dict(size=18, color=COLORS['text'])
        ) if title else None,
        height=height,
        legend=dict(
            bgcolor='rgba(0,0,0,0)',
            bordercolor=COLORS['grid']
        ),
        margin=dict(t=80 if title else 40, b=40, l=60, r=40)
    )
    
    # Update all axes
    fig.update_xaxes(
        gridcolor=COLORS['grid'],
        zerolinecolor=COLORS['grid'],
        tickfont=dict(color=COLORS['text'])
    )
    fig.update_yaxes(
        gridcolor=COLORS['grid'],
        zerolinecolor=COLORS['grid'],
        tickfont=dict(color=COLORS['text'])
    )
    
    return fig