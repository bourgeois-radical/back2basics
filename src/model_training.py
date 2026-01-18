"""
Model Training for Back to Basics presentation.
Train regression models with different loss functions.

MSE  → assumes Normal residuals
MAE  → assumes Laplace residuals  
Huber → hybrid (Normal near zero, Laplace in tails)
"""

import numpy as np
from sklearn.linear_model import LinearRegression, HuberRegressor, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error


class RegressionExperiment:
    """
    Container for training multiple models on the same data.
    Makes comparison explicit and visual.
    """
    
    def __init__(self, features_df, target, feature_names=None):
        """
        Initialize experiment with data.
        
        Parameters
        ----------
        features_df : DataFrame
            Feature matrix
        target : array-like
            Target values (popularity)
        """
        self.X = features_df.values
        self.y = np.asarray(target)
        self.feature_names = feature_names or list(features_df.columns)
        self.n_samples, self.n_features = self.X.shape
        
        # Scale features for stable training
        self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(self.X)
        
        # Storage for trained models
        self.models = {}
        self.predictions = {}
        self.residuals = {}
        self.metrics = {}
    
    def train_mse(self, verbose=True):
        """
        Train with MSE loss (Ordinary Least Squares).
        Assumes: residuals ~ Normal(0, σ²)
        """
        model = LinearRegression()
        model.fit(self.X_scaled, self.y)
        
        pred = model.predict(self.X_scaled)
        resid = self.y - pred
        
        self.models['MSE'] = model
        self.predictions['MSE'] = pred
        self.residuals['MSE'] = resid
        self.metrics['MSE'] = {
            'mse': mean_squared_error(self.y, pred),
            'rmse': np.sqrt(mean_squared_error(self.y, pred)),
            'mae': mean_absolute_error(self.y, pred)
        }
        
        if verbose:
            self._print_results('MSE', model, resid)
        
        return model
    
    def train_mae(self, verbose=True):
        """
        Train with MAE loss (L1 regression via iterative reweighting).
        Assumes: residuals ~ Laplace(0, b)
        
        Note: sklearn doesn't have pure LAD, so we use HuberRegressor
        with very small epsilon to approximate it.
        """
        # Approximate LAD with Huber (epsilon→0 gives LAD)
        model = HuberRegressor(epsilon=1.01, max_iter=1000)
        model.fit(self.X_scaled, self.y)
        
        pred = model.predict(self.X_scaled)
        resid = self.y - pred
        
        self.models['MAE'] = model
        self.predictions['MAE'] = pred
        self.residuals['MAE'] = resid
        self.metrics['MAE'] = {
            'mse': mean_squared_error(self.y, pred),
            'rmse': np.sqrt(mean_squared_error(self.y, pred)),
            'mae': mean_absolute_error(self.y, pred)
        }
        
        if verbose:
            self._print_results('MAE', model, resid)
        
        return model
    
    def train_huber(self, epsilon=1.35, verbose=True):
        """
        Train with Huber loss (hybrid MSE/MAE).
        - Behaves like MSE for small residuals (|r| < epsilon)
        - Behaves like MAE for large residuals (|r| > epsilon)
        
        Default epsilon=1.35 is the standard choice.
        """
        model = HuberRegressor(epsilon=epsilon, max_iter=1000)
        model.fit(self.X_scaled, self.y)
        
        pred = model.predict(self.X_scaled)
        resid = self.y - pred
        
        self.models['Huber'] = model
        self.predictions['Huber'] = pred
        self.residuals['Huber'] = resid
        self.metrics['Huber'] = {
            'mse': mean_squared_error(self.y, pred),
            'rmse': np.sqrt(mean_squared_error(self.y, pred)),
            'mae': mean_absolute_error(self.y, pred)
        }
        
        if verbose:
            self._print_results('Huber', model, resid)
        
        return model
    
    def train_all(self, verbose=True):
        """Train all three models."""
        if verbose:
            print("=" * 70)
            print("🔧 TRAINING REGRESSION MODELS")
            print("=" * 70)
            print(f"n = {self.n_samples} samples, {self.n_features} features")
            print(f"Features: {self.feature_names}")
            print("=" * 70 + "\n")
        
        self.train_mse(verbose)
        self.train_mae(verbose)
        self.train_huber(verbose)
        
        if verbose:
            self._print_comparison()
    
    def _print_results(self, name, model, resid):
        """Print training results for one model."""
        print(f"\n{'─' * 50}")
        print(f"📊 {name} Model")
        print(f"{'─' * 50}")
        
        # Coefficients
        if hasattr(model, 'coef_'):
            print("Coefficients (scaled features):")
            for fname, coef in zip(self.feature_names, model.coef_):
                print(f"  {fname:12}: {coef:+.4f}")
            if hasattr(model, 'intercept_'):
                print(f"  {'intercept':12}: {model.intercept_:+.4f}")
        
        # Residual stats
        print(f"\nResiduals:")
        print(f"  Mean:   {resid.mean():+.4f} (should be ≈0)")
        print(f"  Std:    {resid.std():.4f}")
        print(f"  Min:    {resid.min():.4f}")
        print(f"  Max:    {resid.max():.4f}")
    
    def _print_comparison(self):
        """Print comparison table of all models."""
        print("\n" + "=" * 70)
        print("📈 MODEL COMPARISON")
        print("=" * 70)
        print(f"{'Model':<10} {'MSE':>12} {'RMSE':>12} {'MAE':>12}")
        print("-" * 50)
        for name in ['MSE', 'MAE', 'Huber']:
            if name in self.metrics:
                m = self.metrics[name]
                print(f"{name:<10} {m['mse']:>12.4f} {m['rmse']:>12.4f} {m['mae']:>12.4f}")
        print("=" * 70)
    
    def get_coefficients_comparison(self):
        """Get coefficients from all models as a dict."""
        coefs = {}
        for name, model in self.models.items():
            if hasattr(model, 'coef_'):
                coefs[name] = {
                    'coef': model.coef_,
                    'intercept': getattr(model, 'intercept_', 0)
                }
        return coefs


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def quick_train(features_df, target, verbose=True):
    """
    One-liner to train all models and return the experiment.
    
    Usage:
        exp = quick_train(features, popularity)
        exp.residuals['MSE']  # get residuals
        exp.models['MAE']     # get trained model
    """
    exp = RegressionExperiment(features_df, target)
    exp.train_all(verbose=verbose)
    return exp


def compare_on_clean_vs_outliers(features, target_clean, target_outliers, verbose=True):
    """
    Train models on both datasets and compare.
    Shows why MAE is robust to outliers.
    """
    if verbose:
        print("\n" + "🟢" * 30)
        print("CLEAN DATA (Normal residuals)")
        print("🟢" * 30)
    
    exp_clean = quick_train(features, target_clean, verbose=verbose)
    
    if verbose:
        print("\n" + "🔴" * 30)
        print("DATA WITH OUTLIERS")
        print("🔴" * 30)
    
    exp_outliers = quick_train(features, target_outliers, verbose=verbose)
    
    return exp_clean, exp_outliers