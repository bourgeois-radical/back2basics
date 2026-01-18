# Bias-Variance Tradeoff and the Connection to Regularization

## Speaker Notes

---

## The Setup: What Are We Measuring?

**SAY: "We've been talking about MLE as a way to fit models. Now we need to ask: how do we evaluate how good our fitted model is?"**

We have:

A true function $f_{\text{true}}(x)$ that we don't know.

Observed data: $y = f_{\text{true}}(x) + \varepsilon$, where $\varepsilon \sim \mathcal{N}(0, \sigma^2)$.

A fitted model $\hat{f}(x)$ that we obtain by training on data.

We want to understand the expected prediction error: $\mathbb{E}[(y - \hat{f}(x))^2]$.

---

## Critical Question: What Does $\mathbb{E}[\cdot]$ Mean Here?

**SAY: "This is where most people get confused. The expectation is taken over different possible training datasets."**

Imagine you could:

1. Sample a training set from the true distribution
2. Train your model, get $\hat{f}_1(x)$
3. Repeat: sample another training set, train, get $\hat{f}_2(x)$
4. Repeat again: get $\hat{f}_3(x)$, $\hat{f}_4(x)$, and so on...

Each training set gives you a different fitted model.

$\mathbb{E}[\hat{f}(x)]$ is the **average prediction** your model would make at point $x$ across all these possible training sets.

**SAY: "The model itself is a random variable — it depends on which training data you happened to get."**

---

## Step 1: Decomposing the Residual

Start with a single residual at point $x$:

$$y - \hat{f}(x)$$

The true data generating process is $y = f_{\text{true}}(x) + \varepsilon$, so:

$$y - \hat{f}(x) = \underbrace{(y - f_{\text{true}}(x))}_{\varepsilon \text{ (irreducible noise)}} + \underbrace{(f_{\text{true}}(x) - \hat{f}(x))}_{\text{model error}}$$

**SAY: "We split the residual into two parts: noise that we can never eliminate, and model error that we might be able to reduce."**

---

## Step 2: Decomposing the Model Error

y moved to the right and reduced 

-f_true moved to left 

$\mathbb{E}[\hat{f}(x)] - \mathbb{E}[\hat{f}(x)] = 0$


The model error can be further split by adding and subtracting $\mathbb{E}[\hat{f}(x)]$:

$$f_{\text{true}}(x) - \hat{f}(x) = \underbrace{(f_{\text{true}}(x) - \mathbb{E}[\hat{f}(x)])}_{\text{Bias}} + \underbrace{(\mathbb{E}[\hat{f}(x)] - \hat{f}(x))}_{\text{Variance}}$$

**Bias:** How far is the *average* model prediction from the truth? This measures systematic error — if you trained infinitely many models, would they center on the right answer?

**Variance:** How far is *this particular* model from the average model? This measures how much your model fluctuates depending on which training set you happened to get.

**SAY: "Bias is about being wrong on average. Variance is about being unstable across different training sets."**

---

## Step 3: The MSE Decomposition (Full Derivation)

Now we compute $\mathbb{E}[(y - \hat{f}(x))^2]$. The expectation is over both noise $\varepsilon$ and training set randomness.

Substitute $y = f_{\text{true}}(x) + \varepsilon$:

$$\mathbb{E}[(y - \hat{f}(x))^2] = \mathbb{E}[(\varepsilon + f_{\text{true}}(x) - \hat{f}(x))^2]$$

Expand the square using $(a + b)^2 = a^2 + 2ab + b^2$:

$$= \mathbb{E}[\varepsilon^2] + 2\mathbb{E}[\varepsilon(f_{\text{true}}(x) - \hat{f}(x))] + \mathbb{E}[(f_{\text{true}}(x) - \hat{f}(x))^2]$$

**SAY: "The cross-term vanishes. Why? Because the noise $\varepsilon$ is independent of the model $\hat{f}(x)$, and $\mathbb{E}[\varepsilon] = 0$."**

$$= \sigma^2 + \mathbb{E}[(f_{\text{true}}(x) - \hat{f}(x))^2]$$

---

## Step 4: Decomposing the Model Error Term

Now we work on $\mathbb{E}[(f_{\text{true}}(x) - \hat{f}(x))^2]$.

Add and subtract $\mathbb{E}[\hat{f}(x)]$ inside:

$$\mathbb{E}[(f_{\text{true}}(x) - \hat{f}(x))^2] = \mathbb{E}[(f_{\text{true}}(x) - \mathbb{E}[\hat{f}(x)] + \mathbb{E}[\hat{f}(x)] - \hat{f}(x))^2]$$

Define:

$B = f_{\text{true}}(x) - \mathbb{E}[\hat{f}(x)]$ — this is the bias, a constant (no randomness).

$V = \mathbb{E}[\hat{f}(x)] - \hat{f}(x)$ — this is the variance term, random (depends on training set).

So we have:

$$\mathbb{E}[(B + V)^2] = B^2 + 2B \cdot \mathbb{E}[V] + \mathbb{E}[V^2]$$

**SAY: "What is $\mathbb{E}[V]$? Let's compute it."**

$$\mathbb{E}[V] = \mathbb{E}[\mathbb{E}[\hat{f}(x)] - \hat{f}(x)] = \mathbb{E}[\hat{f}(x)] - \mathbb{E}[\hat{f}(x)] = 0$$

**SAY: "The cross-term vanishes again because $\mathbb{E}[V] = 0$. The variance term is centered at zero by construction."**

We are left with:

$$= B^2 + \mathbb{E}[V^2] = \underbrace{(f_{\text{true}}(x) - \mathbb{E}[\hat{f}(x)])^2}_{\text{Bias}^2} + \underbrace{\mathbb{E}[(\hat{f}(x) - \mathbb{E}[\hat{f}(x)])^2]}_{\text{Variance}}$$

---

## The Final Result

$$\boxed{\mathbb{E}[(y - \hat{f}(x))^2] = \sigma^2 + \text{Bias}^2 + \text{Variance}}$$

| Term | Meaning | Can We Reduce It? |
|------|---------|-------------------|
| $\sigma^2$ | Irreducible noise in the data | No — this is a property of the world |
| $\text{Bias}^2$ | Systematic error of the model | Yes — use more flexible models |
| $\text{Variance}$ | Instability across training sets | Yes — use simpler models or more data |

**SAY: "This is the fundamental tradeoff. We can only control bias and variance, and they typically pull in opposite directions."**

---

## The Tradeoff in Practice

**High Bias, Low Variance (Underfitting):**

Simple models (e.g., linear regression for nonlinear data).

Wrong on average, but consistent across training sets.

**Low Bias, High Variance (Overfitting):**

Complex models (e.g., high-degree polynomials, deep networks).

Right on average, but wildly different depending on which training set you get.

**SAY: "A model that memorizes the training data has low bias (fits training data perfectly) but high variance (won't generalize to new data)."**

---

## Connection to MLE

**SAY: "How does this connect to what we learned about MLE?"**

When you minimize MSE (which is MLE under Gaussian noise assumption), you're implicitly minimizing $\text{Bias}^2 + \text{Variance}$. You cannot touch $\sigma^2$ — it's a property of the data.

The problem: with finite data and a flexible model, minimizing training MSE tends to **reduce bias aggressively** at the cost of **increasing variance**.

**SAY: "MLE says: make the data as probable as possible. With a flexible model, this means fitting the training data very closely — including the noise. This is overfitting."**

---

## From MLE to MAP: Adding Regularization

**SAY: "What if we could tell the model: don't just maximize likelihood — also stay simple?"**

**MLE (Maximum Likelihood Estimation):**

$$\hat{\theta}_{\text{MLE}} = \arg\max_{\theta} P(\text{data} \mid \theta)$$

Find the parameters that make the observed data most probable.

**MAP (Maximum A Posteriori):**

$$\hat{\theta}_{\text{MAP}} = \arg\max_{\theta} P(\theta \mid \text{data})$$

Find the parameters that are most probable given the data.

Using Bayes' theorem:

$$P(\theta \mid \text{data}) = \frac{P(\text{data} \mid \theta) \cdot P(\theta)}{P(\text{data})}$$

Since $P(\text{data})$ doesn't depend on $\theta$:

$$\hat{\theta}_{\text{MAP}} = \arg\max_{\theta} \left[ P(\text{data} \mid \theta) \cdot P(\theta) \right]$$

**SAY: "MAP adds a prior $P(\theta)$ — our belief about what parameter values are plausible before seeing data."**

---

## Taking the Negative Log

Taking the negative log (to convert maximization to minimization):

$$\hat{\theta}_{\text{MAP}} = \arg\min_{\theta} \left[ -\log P(\text{data} \mid \theta) - \log P(\theta) \right]$$

The first term is the negative log-likelihood — this is our loss function (MSE for Gaussian).

The second term is the negative log-prior — this becomes the regularization term.

$$\hat{\theta}_{\text{MAP}} = \arg\min_{\theta} \left[ \underbrace{\text{Loss}}_{\text{fit the data}} + \underbrace{\text{Regularization}}_{\text{stay simple}} \right]$$

---

## Gaussian Prior → Ridge Regression (L2)

Assume a Gaussian prior on parameters: $P(\theta) \sim \mathcal{N}(0, \tau^2)$.

$$P(\theta) = \frac{1}{\sqrt{2\pi\tau^2}} \exp\left(-\frac{\theta^2}{2\tau^2}\right)$$

Taking the negative log:

$$-\log P(\theta) = \frac{\theta^2}{2\tau^2} + \text{const}$$

For a vector of parameters: $-\log P(\theta) \propto \lVert \theta \rVert_2^2 = \sum_j \theta_j^2$.

**SAY: "The Gaussian prior says: I believe parameters should be close to zero. Large parameters are improbable."**

The MAP objective becomes:

$$\hat{\theta}_{\text{MAP}} = \arg\min_{\theta} \left[ \text{MSE} + \lambda \lVert \theta \rVert_2^2 \right]$$

**This is Ridge Regression.**

---

## Laplace Prior → Lasso Regression (L1)

Assume a Laplace prior on parameters: $P(\theta) \sim \text{Laplace}(0, b)$.

$$P(\theta) = \frac{1}{2b} \exp\left(-\frac{\lvert \theta \rvert}{b}\right)$$

Taking the negative log:

$$-\log P(\theta) = \frac{\lvert \theta \rvert}{b} + \text{const}$$

For a vector of parameters: $-\log P(\theta) \propto \lVert \theta \rVert_1 = \sum_j \lvert \theta_j \rvert$.

**SAY: "The Laplace prior also says parameters should be small, but it's more tolerant of a few large values and more aggressive about pushing small values to exactly zero."**

The MAP objective becomes:

$$\hat{\theta}_{\text{MAP}} = \arg\min_{\theta} \left[ \text{MSE} + \lambda \lVert \theta \rVert_1 \right]$$

**This is Lasso Regression.**

---

## Summary: The Full Picture

| Method | Formula | Prior Assumption | Effect |
|--------|---------|------------------|--------|
| MLE | $\min \ \text{Loss}$ | None (uniform) | Fit data as closely as possible |
| Ridge (L2) | $\min \ \text{Loss} + \lambda \lVert \theta \rVert_2^2$ | Gaussian prior | Shrink all parameters toward zero |
| Lasso (L1) | $\min \ \text{Loss} + \lambda \lVert \theta \rVert_1$ | Laplace prior | Shrink parameters; some become exactly zero |

**SAY: "Regularization is not a hack. It's Bayesian inference — you're encoding prior beliefs about what reasonable parameters look like."**

---

## The Bias-Variance Interpretation of Regularization

**SAY: "What does regularization do to bias and variance?"**

By constraining parameters to be small, regularization:

**Increases bias** — the model is prevented from fitting the training data perfectly.

**Decreases variance** — the model is more stable across different training sets.

The hyperparameter $\lambda$ controls this tradeoff:

$\lambda = 0$: Pure MLE, low bias, high variance.

$\lambda \to \infty$: All parameters shrink to zero, high bias, low variance.

**SAY: "You choose $\lambda$ to find the sweet spot — the point where the sum of bias squared and variance is minimized."**

---

## Key Takeaways

**SAY: "Let me summarize the key ideas."**

1. **Expected prediction error decomposes into three terms:** irreducible noise, bias squared, and variance.

2. **Bias and variance trade off:** More flexible models reduce bias but increase variance. Simpler models do the opposite.

3. **MLE minimizes training error:** With flexible models, this leads to overfitting (low bias, high variance).

4. **Regularization is MAP with a prior:** Adding a penalty term is equivalent to assuming a prior distribution on parameters.

5. **Gaussian prior → L2 (Ridge); Laplace prior → L1 (Lasso):** The choice of regularization encodes your belief about parameter distributions.

6. **Regularization trades bias for variance:** You intentionally accept some bias to gain stability.