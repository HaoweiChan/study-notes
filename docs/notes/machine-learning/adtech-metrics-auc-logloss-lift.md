---
title: "AdTech Metrics (AUC, LogLoss, Lift)"
date: "2025-12-27"
tags: ["metrics", "adtech", "machine-learning"]
related: []
slug: "adtech-metrics-auc-logloss-lift"
category: "machine-learning"
---

# AdTech Metrics (AUC, LogLoss, Lift)

## Summary
In AdTech, standard classification metrics like Accuracy are misleading due to extreme class imbalance (CTR is often < 1%). The industry relies on **AUC-ROC** for ranking quality and **LogLoss** for probability calibration, along with **Normalized Entropy** and **Lift** for business impact.

## Details

### 1. The Problem with Accuracy
If the average Click-Through Rate (CTR) is 0.1%, a dummy model that predicts "No Click" for every impression has **99.9% accuracy**. This model is useless. We need metrics that measure ranking ability and probability reliability.

### 2. AUC-ROC (Area Under the ROC Curve)
- **Definition**: The probability that a randomly chosen **positive** instance (click) is ranked higher than a randomly chosen **negative** instance (no-click).
- **Range**: 0.5 (Random) to 1.0 (Perfect).
- **Use Case**: Evaluating the **ranking** quality. Can the model distinguish between a user who will click and one who won't?
- **Pros**: Insensitive to class imbalance.
- **Cons**: Doesn't tell you if the predicted *probability* is correct (e.g., predicting 0.9 for a click vs 0.51 for a click yields same rank order but different bidding implications).

### 3. LogLoss (Binary Cross-Entropy)
- **Formula**: $LogLoss = - \frac{1}{N} \sum_{i=1}^N \[y_i \log(p_i) + (1-y_i) \log(1-p_i)\]$
- **Definition**: Measures the divergence between predicted probabilities and actual labels. Penalizes confident wrong predictions heavily.
- **Use Case**: Evaluating **calibration**. In Real-Time Bidding (RTB), we bid $Bid = p(click) \times Value$. If $p(click)$ is off by 2x, we overpay by 2x. AUC doesn't catch this; LogLoss does.
- **Goal**: Minimize LogLoss.

### 4. Normalized Entropy (NE) / Normalized LogLoss
- **Definition**: LogLoss divided by the entropy of the background average CTR.
- **Formula**: $NE = \frac{LogLoss}{Entropy(p_{avg})}$ where $Entropy(p) = -\[p \log p + (1-p) \log(1-p)\]$
- **Use Case**: Comparing model performance across different datasets/campaigns with different average CTRs. A LogLoss of 0.05 is good if average CTR is 0.1, but terrible if average CTR is 0.0001. NE normalizes this.

### 5. Calibration (Observed/Predicted Ratio)
- **Definition**: Ratio of (Average Predicted CTR) / (Average Actual CTR).
- **Ideal**: 1.0.
- **Significance**: If Ratio > 1.0, the model over-estimates (over-bidding). If < 1.0, it under-estimates (losing opportunities).

## Examples / snippets

### Calculating Metrics with Scikit-Learn

```python
from sklearn.metrics import log_loss, roc_auc_score
import numpy as np

# Ground truth (y_true) and Predicted probabilities (y_pred)
y_true = [0, 0, 0, 1, 0, 1]
y_pred = [0.1, 0.2, 0.05, 0.8, 0.3, 0.6]

# 1. AUC - Measures Ranking Order
auc = roc_auc_score(y_true, y_pred)
print(f"AUC: {auc:.4f}") 
# Result: 1.0 (Perfect ranking: all 1s have higher probs than all 0s)

# 2. LogLoss - Measures Probability Value
ll = log_loss(y_true, y_pred)
print(f"LogLoss: {ll:.4f}")

# 3. Normalized Entropy (NE)
mean_ctr = np.mean(y_true)
entropy = -(mean_ctr * np.log(mean_ctr) + (1 - mean_ctr) * np.log(1 - mean_ctr))
ne = ll / entropy
print(f"Normalized Entropy: {ne:.4f}")
```

## Learning Sources
- [Google: Machine Learning Crash Course - Classification](https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc) - Basics of ROC and AUC.
- [Scikit-Learn Documentation: Log Loss](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html) - Mathematical definition and usage.
- [Facebook: Normalized Cross Entropy](https://research.facebook.com/publications/) - (Search for "Normalized Entropy" in ad tech papers).
- [AdTech Explained: Calibration vs Ranking](https://towardsdatascience.com/) - Search for articles distinguishing ranking metrics from calibration metrics.
