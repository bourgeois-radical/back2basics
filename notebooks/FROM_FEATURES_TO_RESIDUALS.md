# Speaker Notes: From Features to Residuals

## Section 2: Multivariate Feature Space

### The Setup

**SAY: "We looked at one feature — tempo. But songs have many features. Let's add duration and whether the song is in a major or minor key."**

Show the data generation:
```python
features = generate_song_features(n=40, seed=42)
```

**SAY: "40 songs. Three features. Can we visualize this?"**

Show 3D plot.

**SAY: "Each point is a song in feature space. Cyan dots are major key, pink are minor."**

---

### The Trap

**SAY: "Now, could we find MLE parameters for this multivariate distribution? Fit a multivariate Normal?"**

Pause. Let them think.

**SAY: "Technically yes. But here's the thing — that's not what we want."**

**SAY: "We don't want to describe where songs live in feature space. We want to PREDICT something. Given a song's features, what's its popularity?"**

This is the pivot point.

---

## Section 3: Learning a Function

### The Goal

**SAY: "We want to learn a function f that maps features to a target."**

$$f: \mathbb{R}^d \to \mathbb{R}$$

$$f(\text{tempo}, \text{duration}, \text{is\_major}) \to \text{popularity}$$

**SAY: "The domain is feature space — could be hundreds of dimensions. The range is our prediction. Not the true y, but our estimate ŷ."**

Show the 3D plot with popularity as Z-axis.

**SAY: "See the pattern? Higher popularity tends to cluster in certain regions. We want to find that pattern."**

---

### What is Regression?

**SAY: "Regression finds the function f that best predicts y from X. But what does 'best' mean?"**

This is where loss functions enter.

**SAY: "We need to measure how wrong our predictions are. That measurement is the loss function. And here's the key insight..."**

---

## Section 4: The Loss Function IS the Distributional Assumption

### The Reveal

**SAY: "Remember MLE? We assumed data came from a Normal distribution and found the best parameters."**

**SAY: "In regression, we're doing the same thing — but on RESIDUALS, not features."**

$$\text{residual} = y - f(x) = y - \hat{y}$$

| Loss Function | Assumed Distribution of Residuals |
|--------------|-----------------------------------|
| MSE (Mean Squared Error) | Normal |
| MAE (Mean Absolute Error) | Laplace |
| Huber | Normal near zero, Laplace in tails |

**SAY: "When you minimize MSE, you're implicitly assuming residuals are normally distributed. When you minimize MAE, you're assuming Laplace. The choice of loss IS the choice of distribution."**

---

### Why Does This Matter?

Show Normal vs Laplace plot.

**SAY: "Laplace has heavier tails. It 'expects' outliers. Normal thinks outliers are very unlikely — so it panics when it sees one."**

**SAY: "If your data has outliers, MAE is more robust. If your residuals are truly Normal, MSE is more efficient."**

---

## Section 5: Training Models

### The Experiment

**SAY: "Let's train three models on the same data. Same features, same target. Only the loss function changes."**

```python
exp = quick_train(features, popularity)
```

Show comparison table.

**SAY: "On clean data, MSE often wins. It's the optimal estimator when residuals are Normal."**

---

### The Outlier Test

**SAY: "Now let's add some outliers. Five songs that went unexpectedly viral or flopped."**

```python
popularity_outliers, outlier_idx = generate_popularity_with_outliers(features)
```

Train again. Show comparison.

**SAY: "Look at the difference. MSE gets pulled toward the outliers. MAE stays robust. Huber is in between."**

---

## Section 6: Visualizing Residuals

### Residual Distributions

Show residual histogram with Normal/Laplace overlay.

**SAY: "Here are the residuals from each model. Notice how MSE residuals look more Normal — because that's what MSE optimizes for."**

**SAY: "MAE residuals have heavier tails. It's not trying to eliminate outliers — it accepts them."**

---

### The Loss Landscape

Show 1D loss landscape with clean vs outlier data.

**SAY: "Look at this. Clean data — MSE and MAE find almost the same minimum."**

**SAY: "Add one outlier — MSE minimum shifts dramatically. MAE barely moves. The outlier has huge leverage on MSE because it squares the error."**

Key insight:
- MSE: error of 10 contributes 100 to loss
- MAE: error of 10 contributes 10 to loss

**SAY: "Squaring amplifies large errors. That's why MSE is sensitive to outliers."**

---

## Section 7: Gradient Descent

### How Do We Find the Minimum?

**SAY: "We can't always solve for the optimal parameters analytically. We use gradient descent."**

Show gradient intuition plot.

**SAY: "The gradient tells us which way is uphill. We go the opposite direction — downhill — toward the minimum."**

$$\theta_{\text{new}} = \theta_{\text{old}} - \eta \cdot \nabla L(\theta)$$

Where:
- $\eta$ is the learning rate (step size)
- $\nabla L$ is the gradient (slope)

**SAY: "At the minimum, gradient is zero. We've found our MLE estimate — for the residual distribution we assumed."**

---

### MSE vs MAE Gradients

**SAY: "MSE gradient is smooth — proportional to the residual. Easy to optimize."**

$$\frac{\partial}{\partial \theta} \text{MSE} \propto (y - \hat{y})$$

**SAY: "MAE gradient is constant magnitude — just the sign of the residual. Can be jumpy near zero."**

$$\frac{\partial}{\partial \theta} \text{MAE} \propto \text{sign}(y - \hat{y})$$

**SAY: "Huber gives you the best of both: smooth near zero, bounded far away."**

---

## Section 8: The Punchline

### Bringing It All Together

**SAY: "We started by fitting a distribution to features. But that was a trap."**

**SAY: "In supervised learning, we fit a distribution to RESIDUALS. The loss function encodes our assumption about that distribution."**

| What You're Doing | What You're Really Assuming |
|-------------------|----------------------------|
| Minimizing MSE | Residuals ~ Normal(0, σ²) |
| Minimizing MAE | Residuals ~ Laplace(0, b) |
| Minimizing Cross-Entropy | Labels ~ Bernoulli(p) |

**SAY: "Every loss function is MLE in disguise. You're always assuming a distribution — you just might not realize it."**

**SAY: "Choose your loss function wisely. It's not just a number to minimize. It's a statement about how you believe the world works."**
