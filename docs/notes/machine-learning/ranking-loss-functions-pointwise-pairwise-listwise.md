---
title: "Ranking Loss Functions (Pointwise, Pairwise, Listwise)"
date: "2025-12-27"
tags: ["ranking", "loss-functions", "recommender-systems"]
related: []
slug: "ranking-loss-functions-pointwise-pairwise-listwise"
category: "machine-learning"
---

# Ranking Loss Functions (Pointwise, Pairwise, Listwise)

## Summary
In "Learning to Rank" (LTR), loss functions define how the model optimizes the order of items. Approaches are categorized into **Pointwise** (independent items), **Pairwise** (relative order of pairs), and **Listwise** (optimizing the entire list metric).

## Details

### 1. Pointwise Approach
- **Concept**: Treat ranking as a standard regression or classification problem.
- **Input**: Single item + Query.
- **Loss**: MSE (for rating prediction) or LogLoss (for click prediction).
- **Pros**: Simple, standard, fast.
- **Cons**: Ignores the *relative* order.
    - Example: Predicting 0.9 vs 0.8 is the same error as 0.2 vs 0.1, but in ranking, only the top matters. It penalizes errors at the bottom of the list just as much as the top.

### 2. Pairwise Approach
- **Concept**: The goal is to correctly order pairs of items. If Item A > Item B, the model should score $f(A) > f(B)$.
- **Input**: Pair of items (Positive $i$, Negative $j$).
- **Loss**: Minimize inversions.
    - **BPR (Bayesian Personalized Ranking)**: $Loss = - \ln \sigma(x_{uij})$ where $x_{uij} = \hat{y}_{ui} - \hat{y}_{uj}$.
    - **RankNet**: Uses Cross-Entropy on the probability that $i$ is ranked higher than $j$.
    - **LambdaRank**: Adjusts the gradients of RankNet by the change in NDCG gained by swapping the pair.
- **Pros**: Directly optimizes order.
- **Cons**: $O(N^2)$ pairs (though usually sampled).

### 3. Listwise Approach
- **Concept**: Optimize the evaluation metric (NDCG, MAP) over the entire list directly.
- **Input**: Entire list of items for a query.
- **Loss**:
    - **SoftRank**: Smooth approximation of rank to make it differentiable.
    - **LambdaMART**: Gradient Boosted Trees variant using LambdaRank gradients. (State-of-the-art for tabular ranking).
- **Pros**: Optimizes what you actually care about (NDCG).
- **Cons**: Complex implementation, computationally expensive.

## Examples / snippets

### Pairwise BPR Loss (PyTorch)

```python
import torch
import torch.nn as nn

class BPRLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.logsigmoid = nn.LogSigmoid()

    def forward(self, positive_scores, negative_scores):
        """
        positive_scores: Model output for items user interacted with
        negative_scores: Model output for sampled non-interacted items
        """
        # We want pos > neg, so difference > 0
        diff = positive_scores - negative_scores
        
        # Maximize log(sigmoid(diff)) -> Minimize -log(sigmoid(diff))
        loss = -self.logsigmoid(diff)
        
        return loss.mean()
```

## Learning Sources
- [Microsoft Learning to Rank](https://www.microsoft.com/en-us/research/project/mslr/) - Classic papers on RankNet, LambdaRank, LambdaMART.
- [CatBoost/XGBoost Documentation](https://catboost.ai/docs/concepts/loss-functions-ranking.html) - Practical guide to using ranking objectives (YetiRank, PairLogit).
- [TensorFlow Ranking](https://github.com/tensorflow/ranking) - Library for standard ranking losses.
- [BPR: Bayesian Personalized Ranking from Implicit Feedback](https://arxiv.org/abs/1205.2618) - The BPR paper.
