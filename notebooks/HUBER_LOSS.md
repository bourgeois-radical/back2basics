# Speaker Notes: Huber Loss

## What is Huber Loss?

**SAY: "Huber is the compromise. It asks: what if we want MSE's smoothness but MAE's robustness?"**

### The Formula

$$L_\delta(r) = \begin{cases} 
\frac{1}{2}r^2 & \text{if } |r| \leq \delta \\
\delta |r| - \frac{1}{2}\delta^2 & \text{if } |r| > \delta
\end{cases}$$

Where:
- $r = y - \hat{y}$ is the residual
- $\delta$ is the threshold (typically 1.35)

**SAY: "For small residuals — less than delta — it's quadratic, like MSE. For large residuals — greater than delta — it's linear, like MAE."**

---

## Why δ = 1.35?

**SAY: "The standard choice is 1.35. Why? It gives 95% efficiency compared to MSE when residuals are truly Normal."**

In other words: if your data is clean, Huber loses only 5% efficiency compared to MSE. But if you have outliers, it's much more robust.

**SAY: "It's the safe default. You give up a little efficiency for a lot of robustness."**

---

## Visual Intuition

Show the Huber loss plot.

**SAY: "See how Huber smoothly transitions from quadratic to linear? That's the key."**

- Near zero: behaves like MSE (smooth gradient, efficient)
- Far from zero: behaves like MAE (bounded influence, robust)

**SAY: "An outlier with residual 100 contributes 10,000 to MSE, but only about 100δ to Huber. The outlier can't hijack the optimization."**

---

## When to Use Huber

| Situation | Recommendation |
|-----------|----------------|
| Clean data, Normal residuals | MSE (most efficient) |
| Known outliers, heavy tails | MAE (most robust) |
| Uncertain, want safety | Huber (balanced) |

**SAY: "Huber is the pragmatist's choice. When you're not sure about your data, Huber hedges your bets."**

---

## Tuning δ

**SAY: "You can tune delta. Smaller delta means more MAE-like. Larger delta means more MSE-like."**

| δ | Behavior |
|---|----------|
| Small (0.1) | Almost pure MAE |
| Standard (1.35) | Balanced |
| Large (10) | Almost pure MSE |

In practice, 1.35 works well for most cases. Only tune if you have domain knowledge about your residual distribution.
