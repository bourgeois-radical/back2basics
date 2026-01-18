"""
Data Generation for Back to Basics presentation.
The Songwriter's Dilemma: Predicting song popularity.

All data generation is explicit and visible.
"""

import numpy as np
import pandas as pd


def generate_song_features(n=40, seed=42, verbose=True):
    """
    Generate song features for the presentation.
    
    Features:
    - tempo: BPM (beats per minute), continuous
    - duration: song length in seconds, continuous  
    - is_major: whether song is in major key, binary (1=major, 0=minor)
    
    Returns DataFrame with features.
    """
    np.random.seed(seed)
    
    # Tempo: most songs 90-150 BPM, centered around 120
    tempo = np.random.normal(loc=120, scale=15, size=n)
    tempo = np.clip(tempo, 60, 200)
    
    # Duration: most songs 180-240 seconds (3-4 minutes)
    duration = np.random.normal(loc=210, scale=30, size=n)
    duration = np.clip(duration, 120, 360)
    
    # Major/Minor: roughly 60% of pop songs are in major keys
    is_major = np.random.binomial(1, 0.6, size=n)
    
    df = pd.DataFrame({
        'tempo': tempo,
        'duration': duration,
        'is_major': is_major
    })
    
    if verbose:
        print("=" * 60)
        print("🎵 SONG FEATURES GENERATED")
        print("=" * 60)
        print(f"n = {n} songs\n")
        print("Feature Statistics:")
        print("-" * 40)
        print(f"Tempo (BPM):     μ = {tempo.mean():.1f}, σ = {tempo.std():.1f}")
        print(f"Duration (sec):  μ = {duration.mean():.1f}, σ = {duration.std():.1f}")
        print(f"Major key:       {is_major.sum()}/{n} ({100*is_major.mean():.0f}%)")
        print("=" * 60)
    
    return df


def generate_popularity_clean(features_df, seed=42, verbose=True):
    """
    Generate popularity scores (0-100) with CLEAN residuals.
    Residuals are normally distributed - MSE will work well.
    
    True relationship:
    popularity = 30 + 0.3*tempo + 0.05*duration + 10*is_major + noise
    """
    np.random.seed(seed)
    n = len(features_df)
    
    # True coefficients (hidden from audience)
    beta_0 = 30      # intercept
    beta_tempo = 0.3
    beta_duration = 0.05
    beta_major = 10
    
    # Clean normal noise
    noise = np.random.normal(0, 5, size=n)
    
    popularity = (
        beta_0 +
        beta_tempo * features_df['tempo'] +
        beta_duration * features_df['duration'] +
        beta_major * features_df['is_major'] +
        noise
    )
    
    # Clip to valid range
    popularity = np.clip(popularity, 0, 100)
    
    if verbose:
        print("=" * 60)
        print("🎯 TARGET GENERATED: Popularity (Clean)")
        print("=" * 60)
        print(f"Popularity score: μ = {popularity.mean():.1f}, σ = {popularity.std():.1f}")
        print(f"Range: [{popularity.min():.1f}, {popularity.max():.1f}]")
        print(f"Noise: Normal(0, 5)")
        print("=" * 60)
    
    return popularity


def generate_popularity_with_outliers(features_df, seed=42, n_outliers=5, verbose=True):
    """
    Generate popularity scores with OUTLIERS.
    Some songs are viral hits or unexpected flops.
    MAE will be more robust than MSE here.
    
    Same true relationship, but with heavy-tailed noise.
    """
    np.random.seed(seed)
    n = len(features_df)
    
    # True coefficients
    beta_0 = 30
    beta_tempo = 0.3
    beta_duration = 0.05
    beta_major = 10
    
    # Base signal
    signal = (
        beta_0 +
        beta_tempo * features_df['tempo'] +
        beta_duration * features_df['duration'] +
        beta_major * features_df['is_major']
    )
    
    # Normal noise for most points
    noise = np.random.normal(0, 5, size=n)
    
    # Add outliers: some songs go viral or flop unexpectedly
    outlier_idx = np.random.choice(n, size=n_outliers, replace=False)
    outlier_magnitude = np.random.choice([-1, 1], size=n_outliers) * np.random.uniform(25, 40, size=n_outliers)
    noise[outlier_idx] = outlier_magnitude
    
    popularity = signal + noise
    popularity = np.clip(popularity, 0, 100)
    
    if verbose:
        print("=" * 60)
        print("🎯 TARGET GENERATED: Popularity (With Outliers)")
        print("=" * 60)
        print(f"Popularity score: μ = {popularity.mean():.1f}, σ = {popularity.std():.1f}")
        print(f"Range: [{popularity.min():.1f}, {popularity.max():.1f}]")
        print(f"Outliers: {n_outliers} songs with extreme noise (±25-40)")
        print(f"Outlier indices: {sorted(outlier_idx)}")
        print("=" * 60)
    
    return popularity, outlier_idx


def generate_popularity_laplace(features_df, seed=42, verbose=True):
    """
    Generate popularity with Laplace-distributed residuals.
    This is the "true" distribution that MAE assumes.
    """
    np.random.seed(seed)
    n = len(features_df)
    
    beta_0 = 30
    beta_tempo = 0.3
    beta_duration = 0.05
    beta_major = 10
    
    signal = (
        beta_0 +
        beta_tempo * features_df['tempo'] +
        beta_duration * features_df['duration'] +
        beta_major * features_df['is_major']
    )
    
    # Laplace noise (heavier tails than Normal)
    noise = np.random.laplace(0, 4, size=n)
    
    popularity = signal + noise
    popularity = np.clip(popularity, 0, 100)
    
    if verbose:
        print("=" * 60)
        print("🎯 TARGET GENERATED: Popularity (Laplace noise)")
        print("=" * 60)
        print(f"Popularity score: μ = {popularity.mean():.1f}, σ = {popularity.std():.1f}")
        print(f"Range: [{popularity.min():.1f}, {popularity.max():.1f}]")
        print(f"Noise: Laplace(0, 4)")
        print("=" * 60)
    
    return popularity


# =============================================================================
# CONVENIENCE: Get everything at once
# =============================================================================

def get_clean_dataset(n=40, seed=42):
    """Get features + clean target."""
    features = generate_song_features(n, seed, verbose=False)
    target = generate_popularity_clean(features, seed, verbose=False)
    return features, target


def get_outlier_dataset(n=40, seed=42, n_outliers=5):
    """Get features + target with outliers."""
    features = generate_song_features(n, seed, verbose=False)
    target, outlier_idx = generate_popularity_with_outliers(features, seed, n_outliers, verbose=False)
    return features, target, outlier_idx