# Key Formulas: From MLE to Loss Functions

## 1. Maximum Likelihood Estimation

### Likelihood
$$L(\theta) = \prod_{i=1}^{n} f(x_i \mid \theta)$$

### Log-Likelihood
$$\ell(\theta) = \sum_{i=1}^{n} \log f(x_i \mid \theta)$$

### MLE Estimator
$$\hat{\theta}_{MLE} = \arg\max_{\theta} \ell(\theta)$$

---

## 2. Normal Distribution MLE

### PDF
$$f(x \mid \mu, \sigma) = \frac{1}{\sigma\sqrt{2\pi}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$

### Log-Likelihood
$$\ell(\mu, \sigma) = -\frac{n}{2}\log(2\pi) - n\log(\sigma) - \frac{1}{2\sigma^2}\sum_{i=1}^{n}(x_i - \mu)^2$$

### MLE Solutions
$$\hat{\mu} = \frac{1}{n}\sum_{i=1}^{n} x_i = \bar{x}$$

$$\hat{\sigma}^2 = \frac{1}{n}\sum_{i=1}^{n}(x_i - \bar{x})^2$$

---

## 3. Regression: The Residual Perspective

### Model
$$y = f(x) + \varepsilon$$

### Residual
$$r_i = y_i - \hat{y}_i = y_i - f(x_i)$$

### Key Insight
**Minimizing a loss function = MLE under an assumed residual distribution**

---

## 4. Loss Functions and Their Distributions

### MSE (Mean Squared Error)

$$\text{MSE} = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$$

**Assumes:** $\varepsilon \sim \mathcal{N}(0, \sigma^2)$

**Derivation:**
$$\log f(\varepsilon) = -\frac{\varepsilon^2}{2\sigma^2} + \text{const}$$
$$\max \ell \Leftrightarrow \min \sum \varepsilon_i^2 \Leftrightarrow \min \text{MSE}$$

---

### MAE (Mean Absolute Error)

$$\text{MAE} = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|$$

**Assumes:** $\varepsilon \sim \text{Laplace}(0, b)$

**Laplace PDF:**
$$f(\varepsilon) = \frac{1}{2b}\exp\left(-\frac{|\varepsilon|}{b}\right)$$

**Derivation:**
$$\log f(\varepsilon) = -\frac{|\varepsilon|}{b} + \text{const}$$
$$\max \ell \Leftrightarrow \min \sum |\varepsilon_i| \Leftrightarrow \min \text{MAE}$$

---

### Huber Loss

$$L_\delta(r) = \begin{cases} 
\frac{1}{2}r^2 & \text{if } |r| \leq \delta \\[0.5em]
\delta |r| - \frac{1}{2}\delta^2 & \text{if } |r| > \delta
\end{cases}$$

**Assumes:** Hybrid distribution — Normal near zero, Laplace in tails.

---

### Cross-Entropy (Binary Classification)

$$\text{BCE} = -\frac{1}{n}\sum_{i=1}^{n}\left[y_i \log(p_i) + (1-y_i)\log(1-p_i)\right]$$

**Assumes:** $y \sim \text{Bernoulli}(p)$

**Derivation:**
$$P(y \mid p) = p^y(1-p)^{1-y}$$
$$\log P(y \mid p) = y\log(p) + (1-y)\log(1-p)$$
$$\max \ell \Leftrightarrow \min (-\ell) \Leftrightarrow \min \text{BCE}$$

---

## 5. Gradient Descent

### Update Rule
$$\theta^{(t+1)} = \theta^{(t)} - \eta \nabla_\theta L(\theta^{(t)})$$

Where:
- $\eta$ = learning rate
- $\nabla_\theta L$ = gradient of loss w.r.t. parameters

### Gradients of Common Losses

**MSE gradient:**
$$\frac{\partial}{\partial \theta} \text{MSE} = -\frac{2}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)\frac{\partial \hat{y}_i}{\partial \theta}$$

**MAE subgradient:**
$$\frac{\partial}{\partial \theta} \text{MAE} = -\frac{1}{n}\sum_{i=1}^{n}\text{sign}(y_i - \hat{y}_i)\frac{\partial \hat{y}_i}{\partial \theta}$$

---

## 6. Summary Table

| Loss | Formula | Residual Distribution | Optimal Estimator |
|------|---------|----------------------|-------------------|
| MSE | $\frac{1}{n}\sum r_i^2$ | Normal | Mean |
| MAE | $\frac{1}{n}\sum \|r_i\|$ | Laplace | Median |
| Huber | (piecewise) | Hybrid | Robust mean |
| Cross-Entropy | $-\sum y\log p$ | Bernoulli | MLE probability |

---

## 7. The Punchline

$$\boxed{\text{Choosing a loss function} = \text{Choosing a distributional assumption on residuals}}$$

Every time you write `loss='mse'` or `loss='mae'`, you're making a statement about how you believe errors are distributed in your problem.
