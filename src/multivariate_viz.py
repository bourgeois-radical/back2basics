"""
Multivariate Distribution Visualization.
Feature space visualizations in dark theme.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from src.theme import COLORS, apply_dark_theme


def plot_features_2d(features_df, title="🎵 Song Features"):
    """
    2D scatter plot: Tempo vs Duration, colored by Major/Minor.
    
    This is the clean, honest visualization for 2 continuous + 1 binary feature.
    Use this as your primary feature visualization.
    """
    
    tempo = features_df['tempo'].values
    duration = features_df['duration'].values
    is_major = features_df['is_major'].values
    
    fig = go.Figure()
    
    # Plot minor key songs
    minor_mask = is_major == 0
    fig.add_trace(go.Scatter(
        x=tempo[minor_mask],
        y=duration[minor_mask],
        mode='markers',
        name='Minor key',
        marker=dict(
            size=12,
            color=COLORS['pink'],
            opacity=0.8,
            line=dict(color=COLORS['text'], width=1)
        ),
        text=[f"Tempo: {t:.1f} BPM<br>Duration: {d:.0f}s<br>Minor" 
              for t, d in zip(tempo[minor_mask], duration[minor_mask])],
        hovertemplate="%{text}<extra></extra>"
    ))
    
    # Plot major key songs
    major_mask = is_major == 1
    fig.add_trace(go.Scatter(
        x=tempo[major_mask],
        y=duration[major_mask],
        mode='markers',
        name='Major key',
        marker=dict(
            size=12,
            color=COLORS['cyan'],
            opacity=0.8,
            line=dict(color=COLORS['text'], width=1)
        ),
        text=[f"Tempo: {t:.1f} BPM<br>Duration: {d:.0f}s<br>Major" 
              for t, d in zip(tempo[major_mask], duration[major_mask])],
        hovertemplate="%{text}<extra></extra>"
    ))
    
    apply_dark_theme(fig, title=title, height=500)
    
    fig.update_xaxes(title_text='Tempo (BPM)')
    fig.update_yaxes(title_text='Duration (sec)')
    
    fig.update_layout(
        legend=dict(x=0.02, y=0.98)
    )
    
    return fig


def plot_features_2d_with_target(features_df, target, title="🎵 Features → Popularity"):
    """
    2D scatter: Tempo vs Duration, colored by Popularity.
    Marker style indicates Major/Minor.
    
    Clean visualization of the feature-target relationship.
    """
    
    tempo = features_df['tempo'].values
    duration = features_df['duration'].values
    is_major = features_df['is_major'].values
    target = np.asarray(target)
    
    fig = go.Figure()
    
    # Minor key songs (circle markers)
    minor_mask = is_major == 0
    fig.add_trace(go.Scatter(
        x=tempo[minor_mask],
        y=duration[minor_mask],
        mode='markers',
        name='Minor key',
        marker=dict(
            size=12,
            symbol='circle',
            color=target[minor_mask],
            colorscale=[[0, COLORS['red']], [0.5, COLORS['yellow']], [1, COLORS['green']]],
            cmin=target.min(),
            cmax=target.max(),
            opacity=0.8,
            line=dict(color=COLORS['text'], width=1)
        ),
        text=[f"Tempo: {t:.1f} BPM<br>Duration: {d:.0f}s<br>Minor<br>Popularity: {p:.1f}" 
              for t, d, p in zip(tempo[minor_mask], duration[minor_mask], target[minor_mask])],
        hovertemplate="%{text}<extra></extra>"
    ))
    
    # Major key songs (diamond markers)
    major_mask = is_major == 1
    fig.add_trace(go.Scatter(
        x=tempo[major_mask],
        y=duration[major_mask],
        mode='markers',
        name='Major key',
        marker=dict(
            size=12,
            symbol='diamond',
            color=target[major_mask],
            colorscale=[[0, COLORS['red']], [0.5, COLORS['yellow']], [1, COLORS['green']]],
            colorbar=dict(
                title=dict(text='Popularity', font=dict(color=COLORS['text'])),
                tickfont=dict(color=COLORS['text'])
            ),
            cmin=target.min(),
            cmax=target.max(),
            opacity=0.8,
            line=dict(color=COLORS['text'], width=1)
        ),
        text=[f"Tempo: {t:.1f} BPM<br>Duration: {d:.0f}s<br>Major<br>Popularity: {p:.1f}" 
              for t, d, p in zip(tempo[major_mask], duration[major_mask], target[major_mask])],
        hovertemplate="%{text}<extra></extra>"
    ))
    
    apply_dark_theme(fig, title=title, height=500)
    
    fig.update_xaxes(title_text='Tempo (BPM)')
    fig.update_yaxes(title_text='Duration (sec)')
    
    fig.update_layout(
        legend=dict(x=0.02, y=0.98)
    )
    
    # Legend note
    fig.add_annotation(
        x=0.02, y=0.02,
        xref='paper', yref='paper',
        text="● Circle = Minor  ◆ Diamond = Major<br>Color = Popularity",
        showarrow=False,
        font=dict(size=11, color=COLORS['text']),
        bgcolor=COLORS['bg'],
        bordercolor=COLORS['grid'],
        borderwidth=1,
        borderpad=4,
        align='left'
    )
    
    return fig


def plot_feature_space_with_target(features_df, target, title="🎵 Features → Popularity"):
    """
    3D plot where Z-axis is the target (popularity).
    This shows the mapping we want to learn: f(tempo, duration) → popularity
    
    X: Tempo
    Y: Duration
    Z: Popularity (what we want to predict)
    Color: Also popularity for visual clarity
    """
    
    tempo = features_df['tempo'].values
    duration = features_df['duration'].values
    is_major = features_df['is_major'].values
    target = np.asarray(target)
    
    fig = go.Figure()
    
    # Add data points colored by popularity
    fig.add_trace(go.Scatter3d(
        x=tempo,
        y=duration,
        z=target,
        mode='markers',
        marker=dict(
            size=8,
            color=target,
            colorscale=[[0, COLORS['red']], [0.5, COLORS['yellow']], [1, COLORS['green']]],
            colorbar=dict(
                title=dict(text='Popularity', font=dict(color=COLORS['text'])),
                tickfont=dict(color=COLORS['text'])
            ),
            opacity=0.9,
            line=dict(color=COLORS['text'], width=1)
        ),
        text=[f"Song {i+1}<br>Tempo: {t:.1f}<br>Duration: {d:.0f}s<br>{'Major' if m else 'Minor'}<br>Popularity: {p:.1f}" 
              for i, (t, d, m, p) in enumerate(zip(tempo, duration, is_major, target))],
        hovertemplate="%{text}<extra></extra>",
        name='Songs'
    ))
    
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor=COLORS['bg'],
        plot_bgcolor=COLORS['paper'],
        font=dict(family='JetBrains Mono, Consolas, monospace', color=COLORS['text']),
        title=dict(text=title, x=0.5, font=dict(size=18)),
        height=600,
        scene=dict(
            xaxis=dict(
                title='Tempo (BPM)',
                backgroundcolor=COLORS['paper'],
                gridcolor=COLORS['grid'],
                showbackground=True
            ),
            yaxis=dict(
                title='Duration (sec)',
                backgroundcolor=COLORS['paper'],
                gridcolor=COLORS['grid'],
                showbackground=True
            ),
            zaxis=dict(
                title='Popularity',
                backgroundcolor=COLORS['paper'],
                gridcolor=COLORS['grid'],
                showbackground=True
            ),
            bgcolor=COLORS['bg']
        )
    )
    
    return fig


def plot_tempo_vs_popularity(features_df, target, title="🎵 Tempo → Popularity"):
    """
    Simple 2D: just Tempo vs Popularity.
    Good for showing regression relationship.
    """
    
    tempo = features_df['tempo'].values
    is_major = features_df['is_major'].values
    target = np.asarray(target)
    
    fig = go.Figure()
    
    # Minor
    minor_mask = is_major == 0
    fig.add_trace(go.Scatter(
        x=tempo[minor_mask],
        y=target[minor_mask],
        mode='markers',
        name='Minor key',
        marker=dict(
            size=12,
            color=COLORS['pink'],
            opacity=0.8,
            line=dict(color=COLORS['text'], width=1)
        )
    ))
    
    # Major
    major_mask = is_major == 1
    fig.add_trace(go.Scatter(
        x=tempo[major_mask],
        y=target[major_mask],
        mode='markers',
        name='Major key',
        marker=dict(
            size=12,
            color=COLORS['cyan'],
            opacity=0.8,
            line=dict(color=COLORS['text'], width=1)
        )
    ))
    
    apply_dark_theme(fig, title=title, height=450)
    
    fig.update_xaxes(title_text='Tempo (BPM)')
    fig.update_yaxes(title_text='Popularity')
    
    fig.update_layout(legend=dict(x=0.02, y=0.98))
    
    return fig