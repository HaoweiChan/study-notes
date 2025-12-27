---
title: "Causal Inference & Uplift Modeling"
date: "2025-12-27"
tags: ["adtech", "statistics", "machine-learning"]
related: []
slug: "causal-inference-uplift-modeling"
category: "machine-learning"
---

# Causal Inference & Uplift Modeling

## Summary
In AdTech, predicting "who will click" (CTR) is not enough; we need to know "who will click *because* of the ad" (Incremental Lift). **Causal Inference** provides frameworks to estimate the **Average Treatment Effect (ATE)** and **Conditional ATE (CATE)**, preventing wasted spend on users who would have converted anyway ("Sure Things").

## Details

### 1. The Problem: Correlation != Causation
- **Standard ML**: Predicts $P(Y=1|X)$. "If I see a user with these features, will they buy?"
- **Causal ML**: Predicts $P(Y=1|do(T=1)) - P(Y=1|do(T=0))$. "How much *more* likely are they to buy if I show the ad vs. if I don't?"
- **Use Case**:
    - **Sure Things**: Users who buy regardless of ads. (Don't show ad -> Save money).
    - **Lost Causes**: Users who never buy. (Don't show ad).
    - **Persuadables**: Users who buy *only if* shown the ad. (Target these!).
    - **Sleeping Dogs**: Users who buy *less* if shown the ad (annoyed). (Avoid!).

### 2. Uplift Modeling Approaches

#### A. T-Learner (Two-Model)
- Train two separate models:
    1. $\mu_1(x) = E[Y|X, T=1]$ (Treatment Group)
    2. $\mu_0(x) = E[Y|X, T=0]$ (Control Group)
- **Prediction**: $\tau(x) = \mu_1(x) - \mu_0(x)$.
- **Pros**: Simple, uses standard libraries (XGBoost).
- **Cons**: If sample size is small or treatment effect is small, errors in $\mu_1$ and $\mu_0$ can compound.

#### B. S-Learner (Single-Model)
- Train one model including Treatment indicator $T$ as a feature: $Y \sim f(X, T)$.
- **Prediction**: $\tau(x) = f(X, 1) - f(X, 0)$.
- **Pros**: Handles data efficiency better.
- **Cons**: Model might ignore $T$ if it's not a strong predictor (regularization sets coefficient to 0).

#### C. X-Learner (Meta-Learner)
- A multi-stage approach designed for imbalanced treatment groups (e.g., control group is very small).
- Estimates propensity scores and "imputed" treatment effects.

### 3. Evaluation: Qini Curve
- Since we can't observe the counterfactual (can't see both $Y(1)$ and $Y(0)$ for the same user), we can't calculate MSE.
- **Qini Curve / AUUC (Area Under Uplift Curve)**:
    - Sort users by predicted uplift.
    - Plot cumulative incremental gains.
    - A good model sorts "Persuadables" to the top.

## Examples / snippets

### T-Learner Implementation (Python)

```python
from sklearn.linear_model import LogisticRegression
import numpy as np

# Data: X (features), T (treatment 0/1), Y (outcome 0/1)
# Assume we have collected data from a Randomized Controlled Trial (RCT)

def t_learner(X_train, T_train, Y_train, X_test):
    # Split data by treatment
    mask_treatment = (T_train == 1)
    mask_control = (T_train == 0)
    
    # Train Treatment Model
    m1 = LogisticRegression()
    m1.fit(X_train[mask_treatment], Y_train[mask_treatment])
    
    # Train Control Model
    m0 = LogisticRegression()
    m0.fit(X_train[mask_control], Y_train[mask_control])
    
    # Predict Probabilities
    p1 = m1.predict_proba(X_test)[:, 1]
    p0 = m0.predict_proba(X_test)[:, 1]
    
    # Uplift = P(Buy|Ad) - P(Buy|No Ad)
    uplift = p1 - p0
    return uplift
```

## Learning Sources
- [Causal Inference for the Brave and True](https://matheusfacure.github.io/python-causality-handbook/) - Excellent interactive book.
- [Uber Engineering: Causal Inference](https://www.uber.com/blog/causal-inference-at-uber/) - How Uber uses it for pricing/marketing.
- [EconML (Microsoft)](https://github.com/microsoft/EconML) - Library for Causal ML.
- [CausalML (Uber)](https://github.com/uber/causalml) - Uplift modeling toolkit.
