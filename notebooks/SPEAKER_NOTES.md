# MLE Speaker Notes

## The Story: The Songwriter's Dilemma

**SAY: "You're a musician analyzing your catalog. You have 40 songs and you've measured their tempo in BPM."**

**SAY: "The question: what distribution do these tempos follow?"**

We assume Normal. But which one? There are infinitely many, each defined by μ (center) and σ (spread).

**SAY: "Your task: find the μ and σ that best explain your data. This is Maximum Likelihood Estimation."**

---

## Dataset Overview

We're using **song tempo (BPM)** as our single feature for the MLE demo.

Later in the presentation, you can expand to a full prediction problem:
- **Features:** tempo, duration, loudness (keep it simple — 2-3 features max)
- **Regression target:** popularity score (0-100)
- **Classification target:** hit or not-hit

But for MLE, one feature is enough. The point is to show the *principle*, not build a real model.

**Sample data:**
```python
np.random.seed(42)
tempos = np.random.normal(loc=120, scale=15, size=40)
```

True parameters (hidden): μ = 120, σ = 15  
They'll discover: μ̂ ≈ 116.7, σ̂ ≈ 14.1 (sample statistics)

---

## The Critical Assumption

**SAY: "We have a sample. We don't know where it came from. We *assume* it came from a Normal distribution."**

This is the key insight that connects to loss functions later:
- MLE requires an **assumed distribution**
- Different assumptions → different estimators
- For residuals: assume Normal → get MSE; assume Laplace → get MAE

**SAY: "MLE doesn't tell us *which* distribution is correct. It tells us: *if* we assume Normal, what are the best parameters?"**

---

## Why Log-Likelihood?

The likelihood is the product of densities:

$$L(\mu, \sigma) = \prod_{i=1}^{n} f(x_i \mid \mu, \sigma)$$

**SAY: "For 40 data points, you're multiplying 40 numbers each less than 1. This gets astronomically small — computers can't handle it."**

### Underflow: The Technical Details

| Type | Bits | Smallest Number |
|------|------|-----------------|
| float32 | 32 (1 sign, 8 exp, 23 mantissa) | ~10⁻³⁸ |
| float64 | 64 (1 sign, 11 exp, 52 mantissa) | ~10⁻³⁰⁸ |

**Why not just use float64?**
- 2× memory, 2× slower on GPUs
- Deep learning uses float32 or even float16
- Still fails: n=1000 → likelihood can be 10⁻⁵⁰⁰

**SAY: "Likelihood shrinks exponentially with sample size. No floating point format saves you."**

**Solution:** Log turns products into sums:

$$\ell(\mu, \sigma) = \log L = \sum_{i=1}^{n} \log f(x_i \mid \mu, \sigma)$$

**SAY: "Log is strictly increasing, so maximizing log-likelihood gives us the same answer as maximizing likelihood."**

### Why is Log-Likelihood Always Negative?

Density < 1 → log(density) < 0 → sum of negatives = negative

| Density | Log |
|---------|-----|
| 0.05 | -3.0 |
| 0.01 | -4.6 |
| 0.001 | -6.9 |

**SAY: "Yes, log-likelihood is negative. That's normal. Think of it like golf — lower magnitude is better. -150 beats -167. The best log-likelihood is closest to zero."**

**Should we take absolute value?** No!
- We only care about comparisons: -150 > -167
- Absolute value would flip the optimization
- Derivative doesn't care about sign

*Note:* Deep learning uses **negative log-likelihood (NLL)** and *minimizes* it. Same math, different convention.

### Why Negative? Why Minimize?

**SAY: "Optimizers are built to minimize. Gradient descent goes downhill. So instead of maximizing log-likelihood, we flip the sign and minimize."**

$$\text{NLL} = -\ell(\theta) = -\log L(\theta)$$

| Approach | Sign | Goal | Value |
|----------|------|------|-------|
| Log-likelihood | negative (e.g., -150) | maximize | less negative = better |
| **Negative** log-likelihood | positive (e.g., +150) | minimize | smaller = better |

Mathematically identical. Just convention.

### Classification: Cross-Entropy is NLL

Yes, even simple classification uses NLL! Here's how:

**Binary classification:** Assume $y \sim \text{Bernoulli}(p)$ where $p = \sigma(f(x))$ is your model's predicted probability.

The Bernoulli PMF:
$$P(y \mid p) = p^y (1-p)^{1-y}$$

Take log:
$$\log P(y \mid p) = y \log(p) + (1-y) \log(1-p)$$

Flip sign (NLL):
$$-\log P(y \mid p) = -y \log(p) - (1-y) \log(1-p)$$

**SAY: "This is binary cross-entropy. It's just negative log-likelihood of a Bernoulli distribution."**

For n samples, sum them up:
$$\text{BCE} = -\frac{1}{n}\sum_{i=1}^{n} \left[ y_i \log(p_i) + (1-y_i) \log(1-p_i) \right]$$

### Multi-class: Same idea

For $K$ classes with one-hot labels and softmax outputs:
$$\text{Cross-Entropy} = -\sum_{i=1}^{n} \sum_{k=1}^{K} y_{ik} \log(p_{ik})$$

This is NLL of a **Categorical distribution** (generalization of Bernoulli to K outcomes).

**SAY: "Every standard loss function is secretly MLE. Cross-entropy assumes Bernoulli/Categorical. MSE assumes Gaussian. MAE assumes Laplace. The loss function IS your distributional assumption."**

---

## What is Likelihood? (Density vs. Probability)

### The Word "Likelihood"

**SAY: "How likely — or how plausible — are these parameters given what we observed?"**

In everyday English, "likely" and "probable" are synonyms. In statistics, they're different:
- **Probability:** P(data | parameters) — fixed parameters, varying data
- **Likelihood:** L(parameters | data) — fixed data, varying parameters

The formula $f(x \mid \theta)$ can be read two ways:

| Interpretation | What's fixed? | What varies? | Question |
|---------------|---------------|--------------|----------|
| **Probability/Density** | parameters θ | data x | "Given θ, how dense is x?" |
| **Likelihood** | data x | parameters θ | "Given x, how plausible is θ?" |

**SAY: "Same formula, different question. We observed the data — it's fixed. Now we're asking: which parameters make that observation most plausible?"**

Notation:
- $f(x \mid \theta)$ — density of x given θ
- $L(\theta) = f(x \mid \theta)$ — likelihood of θ (data x is implicit, it's what we observed)

### Density vs. Probability

| Discrete (coins, dice) | Continuous (tempo, height) |
|------------------------|---------------------------|
| P(X = x) is probability | f(x) is **density** |
| Always 0 to 1 | Can exceed 1! |
| Sum = 1 | Integral = 1 |

**SAY: "Density can be greater than 1. If the distribution is very narrow, the peak can be huge — but the area under the curve is still 1."**

**SAY: "Think of density as probability per unit — like speed is distance per time. Speed can be 100 km/h even though you haven't traveled 100 km."**

### What the Widget Shows

$$\ell(\mu, \sigma) = -\frac{n}{2}\log(2\pi) - n\log(\sigma) - \frac{1}{2\sigma^2}\sum_{i=1}^{n}(x_i - \mu)^2$$

**SAY: "Each vertical line from a point to the curve represents the density at that point. Green means high — plausible. Red means low — unlikely. The log-likelihood sums all these contributions."**

---

## Why Derivatives?

**SAY: "To find the peak of a hill, we look for where the slope is zero. At the top — slope is zero."**

Strategy:
1. Write $\ell(\mu, \sigma)$
2. Take ∂ℓ/∂μ, set = 0 → solve for μ̂
3. Take ∂ℓ/∂σ, set = 0 → solve for σ̂

They discover empirically (sliders) that μ̂ = x̄ and σ̂ = sample std. Then we prove it.

---

## Line-by-Line Derivation Commentary

*Use this when walking through the derivation. **Bold** = say aloud.*

### Setup

**PDF:** $f(x_i \mid \mu, \sigma) = \frac{1}{\sigma\sqrt{2\pi}} \exp\left(-\frac{(x_i - \mu)^2}{2\sigma^2}\right)$

**SAY: "This is the bell curve formula. The fraction is normalization — makes area equal 1. The exponential creates the bell shape."**

**Likelihood:** $L = \prod_{i=1}^{n} f(x_i)$

**SAY: "Since observations are independent, the joint density is the product. Big Pi means multiply all n terms."**

### Taking the Log

**SAY: "Key property of logarithms: the log of a product is the sum of the logs."**

Write: $\log(a \cdot b) = \log a + \log b$

**SAY: "So our product of n terms becomes a sum of n terms. Much nicer."**

### Log Properties (write these on board)

| Property | Formula |
|----------|---------|
| Log of product | $\log(ab) = \log a + \log b$ |
| Log of exp | $\log(e^x) = x$ |
| Log of reciprocal | $\log(1/a) = -\log(a)$ |
| Log of square root | $\log(\sqrt{a}) = \frac{1}{2}\log(a)$ |

**Final form:**
$$\ell = -n\log(\sigma) - \frac{n}{2}\log(2\pi) - \frac{1}{2\sigma^2}\sum_{i=1}^{n}(x_i - \mu)^2$$

**SAY: "Terms that don't depend on i get multiplied by n when we sum."**

### Finding μ̂

$$\frac{\partial \ell}{\partial \mu} = \frac{\partial}{\partial \mu}\left[-n\log(\sigma) - \frac{n}{2}\log(2\pi) - \frac{1}{2\sigma^2}\sum(x_i - \mu)^2\right]$$

**SAY: "First term — no μ. Constant. Zero. Gone."**

**SAY: "Second term — no μ. Constant. Zero. Gone."**

**SAY: "Third term — this is where μ lives."**

Chain rule on $(x_i - \mu)^2$:

**SAY: "Outer function is 'square', inner function is (xᵢ - μ). Derivative of inner: xᵢ is constant, derivative of -μ is -1."**

$$\frac{d}{d\mu}(x_i - \mu)^2 = 2(x_i - \mu) \cdot (-1) = -2(x_i - \mu)$$

Result: $\frac{\partial \ell}{\partial \mu} = \frac{1}{\sigma^2} \sum(x_i - \mu)$

**SAY: "The minus one-half times minus two gives plus one. Negatives cancel."**

Set = 0 and solve:

$$\sum(x_i - \mu) = 0 \quad\Rightarrow\quad \sum x_i = n\mu \quad\Rightarrow\quad \boxed{\hat{\mu} = \bar{x}}$$

**SAY: "The MLE for μ is the sample mean. Exactly what you found with the slider!"**

### Finding σ̂

**SAY: "Now differentiate with respect to σ."**

First term: **SAY: "Derivative of log is one over the argument."**
$$\frac{d}{d\sigma}(-n\log\sigma) = -\frac{n}{\sigma}$$

Second term: **SAY: "No σ here. Zero."**

Third term: **SAY: "Power rule. σ⁻² becomes -2σ⁻³."**
$$\frac{d}{d\sigma}\sigma^{-2} = -2\sigma^{-3}$$

Result: $\frac{\partial \ell}{\partial \sigma} = -\frac{n}{\sigma} + \frac{\sum(x_i-\mu)^2}{\sigma^3}$

Set = 0, multiply by σ³:
$$\sigma^2 = \frac{1}{n}\sum(x_i - \bar{x})^2 \quad\Rightarrow\quad \boxed{\hat{\sigma} = \sqrt{\frac{1}{n}\sum(x_i - \bar{x})^2}}$$

**SAY: "The MLE for σ is the sample standard deviation — with n, not n-1. Exactly what the slider told us!"**

### Note on Bias

**SAY: "You might notice this divides by n, but stats class taught n-1. The n-1 version is unbiased. MLE is slightly biased but doesn't care — it optimizes likelihood, not bias."**

---

## Pedagogical Flow

1. **Show data** → "These 40 tempos came from some distribution"
2. **Assumption** → "Let's *assume* Normal. What μ and σ?"
3. **Interactive** → Let them discover the optimum with sliders
4. **Reveal** → "You found sample mean and sample std!"
5. **Prove** → Derivatives confirm analytically
6. **Punchline setup** → "So we learned the distribution of... what exactly?"

---

## The Trap You're Setting

Right now they think: "We're fitting the distribution of **features**."

**Later reveal:** In regression, MLE operates on **residuals**, not features!

| Loss | Assumed Residual Distribution |
|------|------------------------------|
| MSE | Normal |
| MAE | Laplace |
| Cross-entropy | Bernoulli |

**SAY: "The model learns f(x) such that y - f(x) follows our assumed distribution. The choice of loss function *is* the choice of distribution."**