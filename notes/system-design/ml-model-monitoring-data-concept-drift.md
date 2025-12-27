---
title: "ML Model Monitoring (Data & Concept Drift)"
date: "2025-12-27"
tags: ["mlops", "system-design", "production-ml"]
related: []
slug: "ml-model-monitoring-data-concept-drift"
category: "system-design"
---

# ML Model Monitoring (Data & Concept Drift)

## Summary
In production, ML models degrade over time not because code changes, but because **data changes**. Monitoring systems track **Data Drift** (input changes) and **Concept Drift** (relationship changes) to trigger retraining or alerting.

## Details

### 1. Types of Drift

#### A. Data Drift (Covariate Shift)
-   **Definition**: The distribution of input features $P(X)$ changes, but the relationship to the target $P(Y|X)$ stays the same.
-   **Example**: Your model was trained on users aged 20-30. Suddenly, a marketing campaign brings in users aged 50-60. The model has never seen this age group and fails.
-   **Detection**: Compare training distribution vs. serving distribution.

#### B. Concept Drift
-   **Definition**: The relationship between inputs and target $P(Y|X)$ changes.
-   **Example**: "Buying masks" was a niche behavior in 2019. In 2020 (COVID), it became mainstream. The *meaning* of the input features changed.
-   **Detection**: Harder. Requires ground truth labels (which might be delayed). Monitor accuracy/loss over time.

#### C. Label Drift (Prior Probability Shift)
-   **Definition**: The distribution of the target variable $P(Y)$ changes.
-   **Example**: In spam detection, normally 10% is spam. Suddenly, a botnet attack makes 90% of traffic spam.

### 2. Detection Metrics

#### A. Statistical Tests (Univariate)
-   **KS Test (Kolmogorov-Smirnov)**: Measures maximum difference between two cumulative distribution functions (CDFs). Good for numerical data.
-   **PSI (Population Stability Index)**:
    -   Bin the data (e.g., deciles).
    -   Compare % of population in each bin for Training (Expected) vs Serving (Actual).
    -   $PSI = \sum (Actual\% - Expected\%) \times \ln(Actual\% / Expected\%)$.
    -   Rule of thumb: PSI < 0.1 (Stable), PSI > 0.2 (Significant Drift).
-   **Chi-Square Test**: For categorical data.

#### B. Model Performance (Lagged)
-   If you get labels instantly (e.g., CTR), monitor LogLoss/AUC hourly.
-   If labels are delayed (e.g., Loan Default takes 1 year), rely on Data Drift proxies.

### 3. Monitoring Architecture
1.  **Inference Service**: Logs inputs (X) and predictions ($\hat{y}$) to a stream (Kafka).
2.  **Drift Calculation Job**:
    -   Consumes Kafka.
    -   Fetches a "Reference Distribution" (from Training Set) from Feature Store.
    -   Calculates PSI/KS every hour.
3.  **Alerting**: PagerDuty if PSI > 0.2.

## Examples / snippets

### Calculating PSI (Population Stability Index)

```python
import numpy as np

def calculate_psi(expected, actual, buckettype='bins', buckets=10, axis=0):
    '''Calculate the PSI (population stability index) across all variables
    Args:
       expected: numpy matrix of original values
       actual: numpy matrix of new values
       buckettype: type of strategy for creating buckets
       buckets: number of buckets
    '''
    
    def psi(expected_array, actual_array, buckets):
        def scale_range (input, min, max):
            input += -(np.min(input))
            input /= np.max(input) / (max - min)
            input += min
            return input

        breakpoints = np.arange(0, buckets + 1) / (buckets) * 100

        if buckettype == 'bins':
            breakpoints = scale_range(breakpoints, np.min(expected_array), np.max(expected_array))
        elif buckettype == 'quantiles':
            breakpoints = np.stack([np.percentile(expected_array, b) for b in breakpoints])

        expected_percents = np.histogram(expected_array, breakpoints)[0] / len(expected_array)
        actual_percents = np.histogram(actual_array, breakpoints)[0] / len(actual_array)

        # Avoid division by zero
        expected_percents = np.where(expected_percents == 0, 0.0001, expected_percents)
        actual_percents = np.where(actual_percents == 0, 0.0001, actual_percents)

        psi_value = np.sum((actual_percents - expected_percents) * np.log(actual_percents / expected_percents))
        return psi_value

    return psi(expected, actual, buckets)

# Usage
train_ages = np.random.normal(30, 5, 1000)
prod_ages = np.random.normal(35, 5, 1000) # Mean shifted -> Drift

psi_score = calculate_psi(train_ages, prod_ages, buckettype='quantiles', buckets=10)
print(f"PSI Score: {psi_score:.4f}")
# PSI > 0.2 indicates significant drift
```

## Flashcards

- What is Data Drift (Covariate Shift)? ::: A change in the distribution of **input features** $P(X)$, while the underlying relationship to the target stays the same.
- What is Concept Drift? ::: A change in the **relationship** between inputs and the target variable $P(Y|X)$ (e.g., user preferences change).
- What is PSI (Population Stability Index)? ::: A metric used to quantify how much a variable's distribution has shifted between two time periods (e.g., Training vs. Production).
- Why is monitoring Concept Drift harder than Data Drift? ::: Because Concept Drift detection usually requires **ground truth labels**, which may be delayed or unavailable in production, whereas Data Drift only looks at inputs.
- What is the typical threshold for PSI to indicate "Significant Drift"? ::: A PSI value greater than **0.2**.

## Quizzes

### Diagnosis
Q: Your fraud detection model's accuracy drops suddenly. You check the input feature distributions (PSI), and they are all stable (PSI < 0.1). What is the most likely cause?
Options:
- A) Data Drift.
- B) Concept Drift.
- C) Bug in the PSI calculation.
- D) The server is down.
Answers: B
Explanation: If inputs $P(X)$ are stable (no Data Drift), but accuracy drops, it implies the relationship $P(Y|X)$ has changed. For example, fraudsters invented a new technique that looks "normal" based on old features. This is Concept Drift.

### Architecture
Q: You have a credit risk model where defaults are known only after 12 months. How do you monitor this model effectively in the short term?
Options:
- A) Wait 12 months to calculate accuracy.
- B) Monitor Data Drift (PSI) on key features daily. If features shift, retrain or investigate.
- C) Assume the model is perfect.
- D) Use the model's own predictions as ground truth.
Answers: B
Explanation: Since you have a long "Label Delay," you cannot monitor accuracy directly. You must rely on leading indicators like Data Drift. If the applicant population changes (e.g., income drops), the model is likely invalid, even if you don't have the default labels yet.

## Learning Sources
- [Evidently AI: Data Drift detection](https://docs.evidentlyai.com/) - Popular open-source tool for ML monitoring.
- [Google Cloud: MLOps Drift Detection](https://cloud.google.com/blog/products/ai-machine-learning/detecting-feature-drift-in-tensorflow-extended) - Implementation in TFX.
- [Fiddler AI: The difference between Data Drift and Concept Drift](https://www.fiddler.ai/blog) - Conceptual guide.
- [Population Stability Index (PSI) Explained](https://towardsdatascience.com/) - Search for "PSI math explained".
