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

## Flashcards

- Why is Accuracy a bad metric for CTR prediction? ::: Because of extreme **class imbalance** (CTR is very low), a model predicting all zeros would have high accuracy but zero value.
- What does AUC measure in the context of CTR? ::: It measures the probability that a random **click** is ranked higher than a random **non-click**. It evaluates **ranking quality**.
- What does LogLoss measure that AUC does not? ::: LogLoss measures **calibration** (how accurate the predicted probability values are), which is critical for calculating correct bids.
- What is the benefit of Normalized Entropy (NE) over LogLoss? ::: NE allows for comparison of model performance across datasets with **different average CTRs**.
- If a model has high AUC but high LogLoss, what does it mean? ::: The model ranks items correctly (good for sorting), but the probability values are incorrectly scaled (bad for pricing/bidding).

## Quizzes

### Metric Selection for Bidding
Q: You are building a Real-Time Bidding (RTB) bidder where the bid price is calculated as `CTR * Value`. You have two models:
- Model A: AUC 0.85, LogLoss 0.4
- Model B: AUC 0.80, LogLoss 0.2
Which model should you choose?
Options:
- A) Model A because it ranks ads better.
- B) Model B because it has better probability calibration.
- C) Neither, use Accuracy.
- D) Model A because higher AUC implies better conversion.
Answers: B
Explanation: For bidding, the *value* of the probability matters directly for the price calculation. Lower LogLoss indicates the predicted probabilities are closer to reality, reducing the risk of over/under-bidding. Model A might rank well but output calibrated probabilities (e.g., predicting 0.99 for a 0.5 event).

### Understanding AUC
Q: An AUC of 0.5 indicates what?
Options:
- A) Perfect prediction
- B) Random guessing
- C) Inverse prediction (predicting 0 for 1 and vice versa)
- D) 50% Accuracy
Answers: B
Explanation: An AUC of 0.5 means the model cannot distinguish between positive and negative classes better than a random coin flip.

## Learning Sources
- [Google: Machine Learning Crash Course - Classification](https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc) - Basics of ROC and AUC.
- [Scikit-Learn Documentation: Log Loss](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html) - Mathematical definition and usage.
- [Facebook: Normalized Cross Entropy](https://research.facebook.com/publications/) - (Search for "Normalized Entropy" in ad tech papers).
- [AdTech Explained: Calibration vs Ranking](https://towardsdatascience.com/) - Search for articles distinguishing ranking metrics from calibration metrics.
