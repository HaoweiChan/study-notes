---
title: "A/B Testing & Experiment Design"
date: "2025-12-27"
category: "machine-learning"
tags: ["statistics", "experimentation", "data-science", "hypothesis-testing"]
related: ["adtech-metrics-auc-logloss-lift"]
slug: "ab-testing-experiment-design"
flashcards:
  - q: "What is the Minimum Detectable Effect (MDE)?"
    a: "The smallest improvement in the metric that the test is designed to detect with a given power and significance level."
  - q: "What is the Peeking Problem in A/B testing?"
    a: "Checking results continuously and stopping early when significance is reached, which inflates the Type I error rate."
  - q: "What is Type I error (Alpha)?"
    a: "False Positive: Rejecting the null hypothesis when it is actually true."
  - q: "What is Type II error (Beta)?"
    a: "False Negative: Failing to reject the null hypothesis when the alternative is true."
  - q: "What is Network Interference (Spillover Effect)?"
    a: "When the treatment of one user affects the behavior of the control group (common in social networks or two-sided marketplaces)."
quizzes:
  - q: "You want to detect a 1% lift in CTR with 80% power. If you decrease the MDE to 0.1%, what happens to the required sample size?"
    options: ["It increases significantly", "It decreases significantly", "It stays the same", "It depends on the p-value"]
    answers: [0]
    explanation: "Sample size is inversely proportional to the square of MDE. Smaller effects require much larger samples to detect."
  - q: "Which metric is best suited as a 'Guardrail Metric' in an AdTech latency experiment?"
    options: ["CTR", "Revenue per User", "99th Percentile Latency (P99)", "Conversion Rate"]
    answers: [2]
    explanation: "Guardrail metrics ensure that optimizing for the primary metric (e.g., CTR) doesn't degrade system health (e.g., Latency)."
---

# A/B Testing & Experiment Design

## Summary
A/B Testing (Split Testing) is the gold standard for causal inference in product development. It involves comparing two versions (Control A vs. Treatment B) to determine which performs better on a specific metric. In AdTech, this is crucial for testing new ranking algorithms, UI changes, or bidding strategies.

## Details

### 1. Core Concepts
*   **Null Hypothesis ($H_0$):** There is no difference between A and B ($\delta = 0$).
*   **Alternative Hypothesis ($H_1$):** There is a difference ($\delta \neq 0$).
*   **Significance Level ($\alpha$):** Probability of rejecting $H_0$ when it is true (Type I Error, typically 0.05).
*   **Power ($1 - \beta$):** Probability of correctly rejecting $H_0$ when $H_1$ is true (typically 0.80).

### 2. Sample Size Calculation
The required sample size ($n$) per group depends on:
1.  **Baseline Conversion Rate ($p$):** Variance depends on $p(1-p)$.
2.  **Minimum Detectable Effect (MDE):** The smallest lift you care about.
3.  **Significance ($\alpha$) & Power ($1-\beta$).**

$$ n \propto \frac{\sigma^2}{\delta^2} $$
*   Where $\delta$ is the MDE. Halving the MDE requires **4x** the sample size.

### 3. Metrics Selection
*   **OEC (Overall Evaluation Criterion):** The single metric you want to optimize (e.g., Revenue, CTR).
*   **Guardrail Metrics:** Metrics that must not degrade (e.g., Page Load Time, App Crashes, Unsubscribe Rate).

### 4. Common Pitfalls
*   **Peeking:** Repeatedly checking the p-value and stopping when significant. Fix: Use **Sequential Testing** (e.g., SPRT) or fixed horizon.
*   **Novelty Effect:** Users click because it's new, not better. Fix: Run the test longer to let behavior stabilize.
*   **Simpson's Paradox:** aggregated trends reverse when data is split by subgroups. Fix: Stratified sampling or randomized block design.
*   **Interference (SUTVA violation):** In two-sided markets (Uber, Ad Bidding), treating one user affects others. Fix: **Switchback Testing** (randomize by time) or **Cluster Randomization** (randomize by city).

## Examples / snippets

### Power Analysis in Python

```python
from statsmodels.stats.power import TTestIndPower

# Parameters
effect_size = 0.05  # Cohen's d (Standardized Mean Difference)
alpha = 0.05        # Significance Level
power = 0.80        # Power

analysis = TTestIndPower()
sample_size = analysis.solve_power(effect_size=effect_size, power=power, alpha=alpha)

print(f"Required Sample Size per Group: {sample_size:.0f}")
# Output: Required Sample Size per Group: 6280
```

### Checking for Significance (T-Test)

```python
from scipy import stats
import numpy as np

# Synthetic Data
control_ctr = np.random.beta(10, 1000, 10000)
treatment_ctr = np.random.beta(12, 1000, 10000)

t_stat, p_val = stats.ttest_ind(control_ctr, treatment_ctr)

print(f"P-value: {p_val:.5f}")
if p_val < 0.05:
    print("Result is Statistically Significant!")
else:
    print("Fail to reject Null Hypothesis.")
```

## Learning Sources
- [Trustworthy Online Controlled Experiments (Kohavi)](https://experimentguide.com/) - The bible of A/B testing.
- [Evan Miller's Sample Size Calculator](https://www.evanmiller.org/ab-testing/sample-size.html) - Standard tool for calculations.
- [Netflix Tech Blog: Experimentation](https://netflixtechblog.com/tagged/experimentation) - Industry case studies.
