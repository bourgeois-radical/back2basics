# MLE Derivation for Laplace Distribution — Speaker Notes

---

## The Setup: Why Laplace?

**SAY: "We just derived that minimizing MSE is equivalent to MLE under a Normal assumption. But what if we use MAE instead? What distribution does that assume?"**

**SAY: "The answer is the Laplace distribution. Let's derive the MLE and see what estimator emerges."**

The Laplace distribution is sometimes called the "double exponential" — two exponential distributions glued back-to-back. Compared to Normal, it has a sharper peak and heavier tails. This shape difference causes mathematical complications we need to understand.

---

## Quick Comparison: Normal vs Laplace PDFs

| Property | Normal | Laplace |
|----------|--------|---------|
| PDF | $\frac{1}{\sigma\sqrt{2\pi}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$ | $\frac{1}{2b} \exp\left(-\frac{\lvert x-\mu \rvert}{b}\right)$ |
| Key term inside exp | $(x-\mu)^2$ (squared) | $\lvert x-\mu \rvert$ (absolute value) |
| Shape at center | Smooth, rounded | Sharp, pointed |
| Tails | Light (decay fast) | Heavy (decay slower) |

**SAY: "The crucial difference is squared vs absolute value. This seemingly small change has profound consequences for the calculus."**

---

## Setup

We have observations $x_1, x_2, \ldots, x_n$ assumed to be i.i.d. from a Laplace distribution with location parameter $\mu$ and scale parameter $b$.

The probability density function (PDF) of a single observation is:

$$f(x_i \mid \mu, b) = \frac{1}{2b} \exp\left(-\frac{\lvert x_i - \mu \rvert}{b}\right)$$

**SAY: "The 1/2b is normalization. The exponential of negative absolute deviation creates the tent shape."**

---

## Step 1: Write the Likelihood Function

Since observations are independent, the joint density is the product:

$$L(\mu, b) = \prod_{i=1}^{n} f(x_i \mid \mu, b) = \prod_{i=1}^{n} \frac{1}{2b} \exp\left(-\frac{\lvert x_i - \mu \rvert}{b}\right)$$

**SAY: "Same logic as Normal — independence means we multiply densities."**

---

## Step 2: Take the Logarithm

$$\ell(\mu, b) = \log L(\mu, b) = \sum_{i=1}^{n} \log\left[\frac{1}{2b} \exp\left(-\frac{\lvert x_i - \mu \rvert}{b}\right)\right]$$

Using $\log(ab) = \log a + \log b$:

$$\ell = \sum_{i=1}^{n} \left[\log\left(\frac{1}{2b}\right) + \log\exp\left(-\frac{\lvert x_i - \mu \rvert}{b}\right)\right]$$

Since $\log(e^x) = x$ and $\log(1/a) = -\log(a)$:

$$\ell = \sum_{i=1}^{n} \left[-\log(2b) - \frac{\lvert x_i - \mu \rvert}{b}\right]$$

The first term doesn't depend on $i$, so it sums to $n$ times itself:

$$\boxed{\ell(\mu, b) = -n\log(2b) - \frac{1}{b}\sum_{i=1}^{n}\lvert x_i - \mu \rvert}$$

**SAY: "Look at that second term. The log-likelihood contains the sum of absolute deviations. Maximizing log-likelihood means minimizing this sum. This IS the MAE loss, times n."**

**SAY: "This confirms: MAE loss equals MLE under Laplace assumption."**

---

## Step 3: Attempting to Find $\hat{\mu}$ — Derivative w.r.t. $\mu$

Now we try to do what worked for the Normal distribution: take the derivative and set it to zero.

$$\frac{\partial \ell}{\partial \mu} = \frac{\partial}{\partial \mu}\left[-n\log(2b) - \frac{1}{b}\sum_{i=1}^{n}\lvert x_i - \mu \rvert\right]$$

**SAY: "First term — no μ. Constant. Zero. Gone."**

$$\frac{\partial \ell}{\partial \mu} = -\frac{1}{b} \cdot \frac{\partial}{\partial \mu}\sum_{i=1}^{n}\lvert x_i - \mu \rvert = -\frac{1}{b} \sum_{i=1}^{n} \frac{\partial}{\partial \mu}\lvert x_i - \mu \rvert$$

**SAY: "And here we hit a problem. We need to compute the derivative of absolute value. But absolute value is not differentiable everywhere."**

---

## 🛑 CRITICAL INTERLUDE: Why is $\lvert x \rvert$ Not Differentiable at Zero?

**SAY: "This is the heart of the matter. Let's understand what differentiable means and why absolute value fails."**

### What Does "Differentiable" Mean?

A function $f(x)$ is differentiable at a point $a$ if the following limit exists:

$$f'(a) = \lim_{h \to 0} \frac{f(a+h) - f(a)}{h}$$

**SAY: "This limit must give the same value whether h approaches zero from the positive side or the negative side. If the two one-sided limits disagree, the derivative does not exist."**

### Checking $f(x) = \lvert x \rvert$ at $x = 0$

**Approaching from the right** ($h > 0$, so $h \to 0^+$):

$$\lim_{h \to 0^+} \frac{\lvert 0 + h \rvert - \lvert 0 \rvert}{h} = \lim_{h \to 0^+} \frac{\lvert h \rvert}{h} = \lim_{h \to 0^+} \frac{h}{h} = +1$$

**SAY: "When h is positive, absolute value of h equals h. We get positive one."**

**Approaching from the left** ($h < 0$, so $h \to 0^-$):

$$\lim_{h \to 0^-} \frac{\lvert 0 + h \rvert - \lvert 0 \rvert}{h} = \lim_{h \to 0^-} \frac{\lvert h \rvert}{h} = \lim_{h \to 0^-} \frac{-h}{h} = -1$$

**SAY: "When h is negative, absolute value of h equals negative h. We get negative one."**

**The two limits do not match.** We get $+1$ from the right and $-1$ from the left.

Therefore, $\lvert x \rvert$ is **not differentiable at $x = 0$**.

### Visual Intuition

**SAY: "Think about what the derivative represents: the slope of the tangent line."**

For $x > 0$: the function $\lvert x \rvert = x$ has slope $+1$

For $x < 0$: the function $\lvert x \rvert = -x$ has slope $-1$

At $x = 0$: there is a sharp corner, a kink

**SAY: "At a corner, there is no single tangent line. You could draw infinitely many lines that touch the corner. Hence, no unique derivative exists."**

### Contrast with $x^2$

The function $x^2$ has no such problem:

The derivative is $2x$

At $x = 0$: derivative is $2(0) = 0$ ✓

**SAY: "The parabola has a smooth, rounded bottom. A unique tangent line exists everywhere. That's why the Normal distribution gave us clean calculus."**

---

## Step 4: The Derivative of $\lvert x_i - \mu \rvert$ (Where It Exists)

Even though $\lvert x_i - \mu \rvert$ is not differentiable everywhere, we can compute the derivative where it does exist.

Let $u = x_i - \mu$. Then $\lvert x_i - \mu \rvert = \lvert u \rvert$.

**Case 1:** If $x_i > \mu$, then $u = x_i - \mu > 0$, so $\lvert u \rvert = u = x_i - \mu$.

$$\frac{\partial}{\partial \mu}(x_i - \mu) = -1$$

**Case 2:** If $x_i < \mu$, then $u = x_i - \mu < 0$, so $\lvert u \rvert = -u = -(x_i - \mu) = \mu - x_i$.

$$\frac{\partial}{\partial \mu}(\mu - x_i) = +1$$

**Case 3:** If $x_i = \mu$, then $u = 0$, and the derivative does not exist (we are at the kink).

### Summary: The Derivative as a Piecewise Function

$$\frac{\partial}{\partial \mu}\lvert x_i - \mu \rvert = \begin{cases} -1 & \text{if } x_i > \mu \\ +1 & \text{if } x_i < \mu \\ \text{undefined} & \text{if } x_i = \mu \end{cases}$$

This can be written compactly using the sign function:

$$\frac{\partial}{\partial \mu}\lvert x_i - \mu \rvert = -\text{sign}(x_i - \mu)$$

where $\text{sign}(z) = \begin{cases} +1 & \text{if } z > 0 \\ 0 & \text{if } z = 0 \\ -1 & \text{if } z < 0 \end{cases}$

**SAY: "The derivative of absolute value gives us plus or minus one — just the sign, not the magnitude. This is fundamentally different from the squared function, which gave us the actual distance times two."**

---

## Step 5: Setting the Derivative to Zero

Despite the non-differentiability issue, let us proceed and see what happens. Assuming $\mu$ does not exactly equal any data point:

$$\frac{\partial \ell}{\partial \mu} = -\frac{1}{b} \sum_{i=1}^{n} \left[-\text{sign}(x_i - \mu)\right] = \frac{1}{b} \sum_{i=1}^{n} \text{sign}(x_i - \mu)$$

Setting this equal to zero:

$$\frac{1}{b} \sum_{i=1}^{n} \text{sign}(x_i - \mu) = 0$$

Since $b > 0$, we can multiply both sides by $b$:

$$\sum_{i=1}^{n} \text{sign}(x_i - \mu) = 0$$

---

## Step 6: Interpreting the Condition

**SAY: "What does this equation mean? Let's unpack it."**

Each term $\text{sign}(x_i - \mu)$ contributes:

$+1$ if $x_i > \mu$ (the observation is above $\mu$)

$-1$ if $x_i < \mu$ (the observation is below $\mu$)

For the sum to equal zero, the $+1$s and $-1$s must cancel out. This means:

$$\text{(number of observations above } \mu\text{)} = \text{(number of observations below } \mu\text{)}$$

**SAY: "This is exactly the definition of the median. The median is the value that splits the data in half — with equal counts above and below."**

$$\boxed{\hat{\mu} = \text{median}(x_1, x_2, \ldots, x_n)}$$

---

## The Profound Difference: Mean vs Median

### For Normal Distribution (MSE)

Setting $\frac{\partial \ell}{\partial \mu} = 0$ gave us:

$$\sum_{i=1}^{n}(x_i - \mu) = 0 \implies \sum_{i=1}^{n} x_i = n\mu \implies \hat{\mu} = \frac{1}{n}\sum_{i=1}^{n} x_i = \bar{x}$$

**SAY: "The condition involves the values of the deviations. This leads to an arithmetic solution — add them up, divide by n."**

### For Laplace Distribution (MAE)

Setting $\frac{\partial \ell}{\partial \mu} = 0$ gave us:

$$\sum_{i=1}^{n} \text{sign}(x_i - \mu) = 0$$

**SAY: "The condition involves only the signs of the deviations, not their magnitudes. This leads to a counting solution — find where equal numbers lie above and below."**

### Why This Happens

| Distribution | Term in Exponent | Derivative | Type of Equation | Solution |
|--------------|------------------|------------|------------------|----------|
| Normal | $(x_i - \mu)^2$ | $2(x_i - \mu)$ | Linear in data values | Algebraic (mean) |
| Laplace | $\lvert x_i - \mu \rvert$ | $\pm 1$ (signs only) | Linear in signs | Order statistic (median) |

**SAY: "The squared function remembers how far each point is from μ — the derivative is proportional to the distance. The absolute value function only remembers which side each point is on — the derivative is plus or minus one regardless of distance."**

---

## Step 7: Finding $\hat{b}$ — Derivative w.r.t. $b$

The scale parameter $b$ does have a closed-form MLE, similar to $\sigma$ in the Normal case.

Starting from:

$$\ell(\mu, b) = -n\log(2b) - \frac{1}{b}\sum_{i=1}^{n}\lvert x_i - \mu \rvert$$

Let $S = \sum_{i=1}^{n}\lvert x_i - \mu \rvert$ (treating this as a constant w.r.t. $b$).

$$\ell = -n\log(2b) - \frac{S}{b}$$

Using $\log(2b) = \log 2 + \log b$:

$$\ell = -n\log 2 - n\log b - \frac{S}{b}$$

**SAY: "Now we differentiate with respect to b."**

**First term:** $\frac{\partial}{\partial b}(-n\log 2) = 0$ (constant)

**Second term:** $\frac{\partial}{\partial b}(-n\log b) = -\frac{n}{b}$

**SAY: "Derivative of log is one over the argument."**

**Third term:** We need $\frac{\partial}{\partial b}\left(-\frac{S}{b}\right) = -S \cdot \frac{\partial}{\partial b}(b^{-1})$

Using power rule: $\frac{d}{db}b^{-1} = -b^{-2} = -\frac{1}{b^2}$

$$-S \cdot \left(-\frac{1}{b^2}\right) = \frac{S}{b^2}$$

**Combining all terms:**

$$\frac{\partial \ell}{\partial b} = -\frac{n}{b} + \frac{S}{b^2} = -\frac{n}{b} + \frac{1}{b^2}\sum_{i=1}^{n}\lvert x_i - \mu \rvert$$

---

## Step 8: Solving for $\hat{b}$

Set the derivative to zero:

$$-\frac{n}{b} + \frac{S}{b^2} = 0$$

Multiply both sides by $b^2$:

$$-nb + S = 0$$

$$S = nb$$

$$b = \frac{S}{n} = \frac{1}{n}\sum_{i=1}^{n}\lvert x_i - \mu \rvert$$

Substituting $\hat{\mu} = \text{median}(x_1, \ldots, x_n)$:

$$\boxed{\hat{b} = \frac{1}{n}\sum_{i=1}^{n}\lvert x_i - \text{median} \rvert}$$

**SAY: "This is called the Mean Absolute Deviation from the median, or MAD. It is the Laplace analogue of the sample variance."**

---

## Beautiful Parallel: Normal vs Laplace

| Quantity | Normal Distribution | Laplace Distribution |
|----------|---------------------|----------------------|
| Location MLE | $\hat{\mu} = \bar{x}$ (mean) | $\hat{\mu} = \text{median}$ |
| Scale MLE | $\hat{\sigma}^2 = \frac{1}{n}\sum_{i=1}^{n}(x_i - \bar{x})^2$ | $\hat{b} = \frac{1}{n}\sum_{i=1}^{n}\lvert x_i - \text{median} \rvert$ |
| Scale interpretation | Mean Squared Deviation from Mean | Mean Absolute Deviation from Median |
| Loss function | MSE (Mean Squared Error) | MAE (Mean Absolute Error) |

**SAY: "Everything pairs up perfectly. The Laplace distribution is like the absolute value version of the Normal distribution, and all the estimators follow suit."**

---

## Why Does the Median Minimize MAE? An Intuition

**SAY: "Here is another way to see it. Imagine you are standing at position μ on a number line, and there are people standing at positions x₁ through xₙ. Your goal is to minimize the total walking distance if everyone walks to you."**

$$\text{Total distance} = \sum_{i=1}^{n}\lvert x_i - \mu \rvert$$

If you take one step to the right:

Everyone to your right gets one step closer (saves distance)

Everyone to your left gets one step farther (adds distance)

**SAY: "If there are more people on one side, you should move toward that side. You stop moving when there are equal numbers on both sides. That is the median."**

**Contrast with MSE:** With squared distances, it is not just about how many people are on each side, but how far they are. The mean balances the total pull from each side, weighted by distance.

---

## Robustness to Outliers

**SAY: "This explains why the median is robust to outliers."**

Consider the derivative: $\text{sign}(x_i - \mu) = \pm 1$

Whether $x_i$ is slightly away from $\mu$ or extremely far away, the derivative contribution is still just $\pm 1$. Outliers do not get extra voting power.

**SAY: "With squared loss, an outlier 100 units away contributes 200 to the derivative. With absolute loss, it contributes just 1. The median treats all deviations equally in terms of direction, ignoring magnitude."**

---

## Technical Note: Subgradients

For readers interested in the rigorous mathematical treatment:

The absolute value function $\lvert x \rvert$ is convex but not differentiable at $x = 0$. In convex optimization, we handle this using subgradients. The subdifferential of $\lvert x \rvert$ at $x = 0$ is the interval $[-1, +1]$:

$$\partial \lvert x \rvert \big|_{x=0} = [-1, +1]$$

A minimum occurs where $0 \in \partial f(x)$, which for our problem means:

$$0 \in \sum_{i=1}^{n} \partial \lvert x_i - \mu \rvert$$

This condition is satisfied at the median, even handling the edge cases rigorously.

---

## Summary

| Parameter | MLE Estimator |
|-----------|---------------|
| $\mu$ | $\hat{\mu} = \text{median}(x_1, \ldots, x_n)$ |
| $b$ | $\hat{b} = \frac{1}{n}\sum_{i=1}^{n} \lvert x_i - \text{median} \rvert$ |

---

## Key Takeaways

**SAY: "Let me summarize the key points."**

**1. MAE loss equals MLE under Laplace assumption** — just as MSE loss equals MLE under Normal assumption.

**2. The absolute value function is not differentiable at zero** because the left and right derivatives do not match ($-1$ vs $+1$). Geometrically, there is a kink with no unique tangent line.

**3. The derivative gives signs, not distances** — so the optimization condition counts points above versus below, rather than summing distances.

**4. The median emerges as the solution** — because it is the point with equal counts on each side.

**5. The median is robust to outliers** — because the derivative contribution is $\pm 1$ regardless of how far away the point is.

---

## Connecting to Machine Learning

**SAY: "When you choose a loss function, you are making an implicit assumption about error distribution."**

| Loss Function | Assumed Distribution | Optimal Prediction |
|---------------|---------------------|-------------------|
| MSE | Normal (Gaussian) | Mean |
| MAE | Laplace | Median |
| Quantile Loss | Asymmetric Laplace | Quantile |
| Huber Loss | Hybrid | Robust mean |

**SAY: "The median's robustness to outliers is why MAE is often preferred when data contains extreme values. Now you understand the probabilistic justification for this choice."**


# On mu notation

This is a really important conceptual question, and I think the confusion arises because we use the same symbol $\mu$ for both distributions but it has slightly different interpretations.

## What $\mu$ Represents in Each Distribution

In both cases, $\mu$ is the **location parameter** of the distribution — it tells you where the distribution is centered. But "centered" means different things depending on the distribution's shape.

**For the Normal distribution:** The distribution is symmetric, and it turns out that the mean, median, and mode are all the same value. So $\mu$ is simultaneously the population mean, the population median, and the mode. When we do MLE, we discover that the best estimate $\hat{\mu}$ equals the sample mean $\bar{x}$.

**For the Laplace distribution:** The distribution is also symmetric, so again the mean, median, and mode all coincide at $\mu$. The parameter $\mu$ is the population mean AND the population median. But when we do MLE, we discover that the best estimate $\hat{\mu}$ equals the sample median.

## The Subtle Point

Here's what might be causing confusion: In both cases, we're trying to estimate the same thing — the center of the distribution, which we call $\mu$. The difference is in **how we estimate it**:

| Distribution | Parameter $\mu$ is... | Best estimator $\hat{\mu}$ is... |
|--------------|----------------------|----------------------------------|
| Normal | Population mean (= population median) | Sample mean |
| Laplace | Population mean (= population median) | Sample median |

So the Laplace $\mu$ is indeed the population mean. It just happens that the most efficient way to estimate this population mean is to use the sample median, not the sample mean.

## Why Does This Happen?

This seems paradoxical at first: we want to estimate the population mean, but we use the sample median? The reason is that the sample median is a **more efficient estimator** of the Laplace location parameter than the sample mean would be.

If you used the sample mean to estimate $\mu$ in a Laplace distribution, you would get an unbiased estimate (on average, you'd be right), but it would have higher variance — your estimates would be more spread out. The sample median gives you tighter, more reliable estimates because it's not thrown off by the heavy tails of the Laplace distribution.

## An Analogy

Think of it this way: Suppose you want to know the "center" of a city (maybe defined as the average location of all buildings). You could:

1. Calculate the arithmetic average of all building coordinates (like the sample mean)
2. Find the point that minimizes total travel distance to all buildings (like the sample median)

Both approaches are trying to find "the center," but they use different methods. For a Laplace-distributed city (one with some buildings very far from the core), method 2 gives you a more stable answer that isn't yanked around by outlying buildings.

## To Directly Answer Your Question

The $\mu$ in Laplace is the population mean (which equals the population median due to symmetry). We "assume different values of $\mu$" in the same way as for Normal — we're searching over possible locations for the center of the distribution. The MLE procedure tells us that the sample median is the best way to estimate this center, even though the center itself is also the population mean.

Does this distinction make sense? The key is separating "what parameter are we estimating" (the population center $\mu$) from "what formula do we use to estimate it" (sample mean vs sample median).

# Why Not Use Median for Normal and Mean for Laplace?

## The Question

If $\mu$ is both the population mean and the population median for both Normal and Laplace distributions (due to symmetry), why does MLE give us different estimators? Why can't we use either one for either distribution?

---

## The Short Answer

You **can** use either estimator for either distribution. Both the sample mean and sample median are valid, unbiased estimators of $\mu$ in both cases. The difference is **efficiency** — how precise your estimate is. MLE automatically finds the most efficient estimator for your assumed distribution.

---

## What Is Efficiency?

Two estimators can both be unbiased (correct on average) but have different variances. The one with lower variance is more **efficient** — it gives you tighter, more reliable estimates.

**SAY: "Think of two archers both aiming at the bullseye. Both hit the center on average — they're unbiased. But one archer's shots are tightly clustered, while the other's are spread out. The first archer is more efficient."**

---

## The Variance Comparison

For data from a **Normal distribution**:

$$\text{Var}(\bar{x}) = \frac{\sigma^2}{n}$$

$$\text{Var}(\text{median}) \approx \frac{\pi}{2} \cdot \frac{\sigma^2}{n} \approx 1.57 \cdot \frac{\sigma^2}{n}$$

**SAY: "For Normal data, the sample mean is about 57% more precise than the sample median. Using the median means you're throwing away information."**

For data from a **Laplace distribution**, the situation reverses: the sample median has lower variance than the sample mean.

---

## Why Does MLE Find the Optimal Estimator?

When you take the derivative of the log-likelihood and set it to zero, you're implicitly asking: "How should I weight each observation to extract maximum information?"

**For Normal — derivative gives** $\sum(x_i - \mu) = 0$:

Each observation is weighted by its distance from $\mu$. Points far from the center contribute more to the equation. This makes sense because under Normal assumptions, extreme observations are genuinely informative — the thin tails mean an extreme value really did come from far out, so it should influence our estimate.

**For Laplace — derivative gives** $\sum \text{sign}(x_i - \mu) = 0$:

Each observation contributes only $\pm 1$, regardless of distance. Extreme observations get no extra weight. This makes sense because under Laplace assumptions, the heavy tails mean extreme values are common even when the center is not near them — they're less informative about location.

**SAY: "The shape of the log-likelihood encodes how much to trust each observation. MLE reads this information and gives you the optimal weighting automatically."**

---

## Concrete Example

Consider the sample: $\{1, 2, 3, 4, 100\}$

Sample mean: $\bar{x} = \frac{1 + 2 + 3 + 4 + 100}{5} = 22$

Sample median: $\tilde{x} = 3$

**SAY: "Which estimate is better? It depends on where you think that 100 came from."**

**If the data is Normal:**

A value of 100 when the true center is 3 would be astronomically improbable — it would be many standard deviations away. So perhaps the true center really is higher, closer to 22, which would make 100 less extreme. The mean "listens" to the outlier because under Normal assumptions, outliers are informative.

**If the data is Laplace:**

The heavy tails make a value of 100 reasonably probable even if the true center is only 3. The outlier is not strong evidence that the center is far from 3. The median "ignores" the outlier because under Laplace assumptions, outliers are not very informative about location.

---

## The Information-Theoretic View

There is a deep result in statistics: the MLE is asymptotically efficient. This means that as sample size grows, no other estimator can have lower variance than the MLE.

The Cramér-Rao bound gives the minimum possible variance for an unbiased estimator. The MLE achieves this bound (for large $n$). Using a different estimator — like the median for Normal data — means you're leaving precision on the table.

**SAY: "MLE isn't just one method among many. It's provably optimal in terms of extracting information from your data, given your assumed distribution."**

---

## Intuition: What Are You Assuming About Outliers?

The choice between mean and median ultimately comes down to: **how informative are extreme observations about the center?**

| Assumption | Implication | Optimal Estimator |
|------------|-------------|-------------------|
| Outliers are informative (Normal) | Weight observations by distance | Mean |
| Outliers are noise (Laplace) | Weight observations equally (by sign) | Median |

**SAY: "When you choose MSE loss, you're saying outliers are informative. When you choose MAE loss, you're saying outliers are noise. The loss function encodes your belief about the data-generating process."**

---

## Summary

**SAY: "To answer the original question: yes, you can use the median for Normal or the mean for Laplace. They're valid estimators. But MLE tells you which estimator is optimal — which one extracts maximum information from the data. The mean is optimal for Normal; the median is optimal for Laplace. Using the 'wrong' estimator means you're being inefficient with your data."**

| Distribution | Valid Estimators | MLE (Optimal) Estimator | Why Optimal? |
|--------------|------------------|-------------------------|--------------|
| Normal | Mean, Median, others | Mean | Minimum variance; uses distance information |
| Laplace | Mean, Median, others | Median | Minimum variance; robust to heavy tails |