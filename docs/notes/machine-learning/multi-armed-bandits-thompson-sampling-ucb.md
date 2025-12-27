---
title: "Multi-Armed Bandits (Thompson Sampling, UCB)"
date: "2025-12-27"
tags: ["reinforcement-learning", "adtech", "ab-testing"]
related: []
slug: "multi-armed-bandits-thompson-sampling-ucb"
category: "machine-learning"
---

# Multi-Armed Bandits (Thompson Sampling, UCB)

## Summary
Multi-Armed Bandits (MAB) solve the **Explore-Exploit** dilemma: how to balance testing new options (exploration) with choosing the best known option (exploitation) to maximize total reward. Key algorithms include **Thompson Sampling** (Bayesian) and **UCB** (Frequentist).

## Details

### 1. The Problem
You have $K$ slot machines ("arms"). Each pays out with a different, unknown probability. You want to maximize your total winnings over $N$ pulls.
- **Exploit**: If you only pick the current winner, you might miss a better arm you haven't tried enough.
- **Explore**: If you only random, you waste money on bad arms.
- **AdTech Application**: Choosing which ad creative to show. (Arm = Creative, Reward = Click).

### 2. Algorithms

#### A. Epsilon-Greedy ($\epsilon$-greedy)
- Flip a coin.
- With probability $\epsilon$ (e.g., 0.1): Choose a random arm (Explore).
- With probability $1-\epsilon$: Choose the arm with the highest current average reward (Exploit).
- **Pros**: Simple.
- **Cons**: Constant exploration (continues to explore bad arms forever unless $\epsilon$ decays).

#### B. Upper Confidence Bound (UCB)
- **Principle**: "Optimism in the face of uncertainty."
- Calculate the Confidence Interval for each arm's expected reward.
- Pick the arm with the highest **Upper Bound**.
- **Logic**:
    - If an arm is good (high average), the upper bound is high.
    - If an arm is unknown (few samples), the confidence interval is wide, so the upper bound is high.
- **Formula**: $Score_i = \bar{\mu}_i + \sqrt{\frac{2 \ln t}{n_i}}$

#### C. Thompson Sampling (Bayesian)
- **Principle**: Probability Matching. Choose an arm according to the probability that it is the optimal arm.
- **Mechanism**:
    - Maintain a posterior distribution (e.g., Beta) for each arm's reward.
    - Sample a random value from each arm's distribution.
    - Pick the arm with the highest **Sampled** value.
- **Logic**:
    - If we are uncertain (few data), the distribution is wide $\rightarrow$ high chance of sampling a high value (Explore).
    - If we are certain it's bad, distribution is narrow and low $\rightarrow$ low chance of selection.
- **Performance**: Generally outperforms UCB in industry practice.

### 3. Contextual Bandits
Standard MAB assumes the best arm is the same for everyone. **Contextual Bandits** use "Context" (User features) to decide.
- Instead of keeping a simple average, we fit a linear model (or Neural Net) to predict Reward given Context + Arm.
- Used heavily in News Feed recommendation (Yahoo Front Page).

## Examples / snippets

### Thompson Sampling for Bernoulli Bandits (Python)

```python
import numpy as np

class ThompsonSampling:
    def __init__(self, n_arms):
        self.n_arms = n_arms
        # Beta parameters: alpha (successes + 1), beta (failures + 1)
        self.alpha = np.ones(n_arms)
        self.beta = np.ones(n_arms)

    def select_arm(self):
        # Sample from Beta(alpha, beta) for each arm
        samples = [np.random.beta(self.alpha[i], self.beta[i]) for i in range(self.n_arms)]
        return np.argmax(samples)

    def update(self, arm, reward):
        # Reward is 1 (Click) or 0 (No Click)
        if reward == 1:
            self.alpha[arm] += 1
        else:
            self.beta[arm] += 1

# Simulation
bandit = ThompsonSampling(n_arms=3)
# Arm 0: 10% CTR, Arm 1: 50% CTR, Arm 2: 20% CTR
true_probs = [0.1, 0.5, 0.2]

for _ in range(1000):
    arm = bandit.select_arm()
    reward = 1 if np.random.rand() < true_probs[arm] else 0
    bandit.update(arm, reward)

print(f"Alphas: {bandit.alpha}")
print(f"Betas: {bandit.beta}")
# Expect Arm 1 (Index 1) to have high Alpha
```

## Learning Sources
- [A Tutorial on Thompson Sampling (arXiv)](https://arxiv.org/abs/1707.02038) - Comprehensive tutorial.
- [Sutton & Barto: Reinforcement Learning](http://incompleteideas.net/book/the-book-2nd.html) - Chapter 2 covers Bandits in detail.
- [Netflix Tech Blog: Selecting the best artwork](https://netflixtechblog.com/) - Search for how Netflix uses bandits for thumbnails.
- [Vowpal Wabbit Contextual Bandits](https://vowpalwabbit.org/) - Industry standard library for Contextual Bandits.
