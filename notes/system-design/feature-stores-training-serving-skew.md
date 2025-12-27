---
title: "Feature Stores & Training-Serving Skew"
date: "2025-12-27"
tags: ["mlops", "system-design", "data-engineering"]
related: []
slug: "feature-stores-training-serving-skew"
category: "system-design"
---

# Feature Stores & Training-Serving Skew

## Summary
A **Feature Store** (e.g., Feast, Tecton) solves the **Training-Serving Skew** problem by providing a centralized repository for ML features. It ensures that the feature values used to train a model (offline) are mathematically identical to the features available at inference time (online), and handles **Point-in-Time** correctness.

## Details

### 1. The Problem: Training-Serving Skew
Skew occurs when the data used for inference differs from the data used for training.
-   **Logic Skew**: The Data Scientist writes `pandas` code for training features, but the Backend Engineer rewrites it in `Java/SQL` for production. Subtle bugs (e.g., handling nulls differently) cause model degradation.
-   **Data Latency Skew**: Training used yesterday's batch data, but production uses real-time stream data.
-   **Time Travel / Leakage**: Training data accidentally includes "future" information (e.g., calculating "avg spend" using transactions that happened *after* the prediction event).

### 2. What is a Feature Store?
A system that manages feature engineering pipelines and serves them to two targets:
1.  **Offline Store** (e.g., BigQuery, S3, Parquet): For generating historical training datasets. Supports "Time Travel" queries.
2.  **Online Store** (e.g., Redis, DynamoDB): For low-latency (<10ms) retrieval of the *latest* feature values during real-time inference.

### 3. Key Concepts
-   **Feature Registry**: Single source of truth for feature definitions (YAML/Python).
-   **Materialization**: The process of computing features and pushing them to the Online Store.
-   **Point-in-Time (PIT) Correctness**:
    -   When generating training data, we join labels (events) with features.
    -   Crucial Rule: For a label at time $T$, we must join feature values known at time $t < T$.
    -   Feature Stores automate this complex "As-Of Join".

## Examples / snippets

### Defining a Feature in Feast (Python)
Instead of ad-hoc SQL, we define features declaratively.

```python
from feast import Entity, FeatureView, Field
from feast.types import Float32, Int64
from datetime import timedelta

# Define the entity (primary key)
user = Entity(name="user", join_keys=["user_id"])

# Define the source and features
user_stats_view = FeatureView(
    name="user_transaction_stats",
    entities=[user],
    ttl=timedelta(days=1),
    schema=[
        Field(name="avg_daily_transactions", dtype=Float32),
        Field(name="total_spend_7d", dtype=Float32),
    ],
    online=True,  # Materialize to Redis
    source=transaction_source, # Points to a Batch or Stream source
)
```

### Retrieving Features

**Training (Historical Retrieval):**
```python
# Returns a dataframe with features joined at the correct timestamp for each event
training_df = store.get_historical_features(
    entity_df=events_df, 
    features=["user_transaction_stats:total_spend_7d"]
).to_df()
```

**Inference (Online Retrieval):**
```python
# Returns the latest value from Redis (<10ms)
features = store.get_online_features(
    features=["user_transaction_stats:total_spend_7d"],
    entity_rows=[{"user_id": 1001}]
).to_dict()
```

## Flashcards

- What is Training-Serving Skew? ::: The performance degradation caused when production data/logic differs from training data/logic.
- What is the main responsibility of a Feature Store? ::: To serve consistent feature values to both **offline training** and **online inference** environments.
- What is Point-in-Time (PIT) Correctness? ::: Ensuring that historical training data only uses feature values that were actually available **before** the event timestamp (no future leakage).
- Which database type is typically used for the "Online Store"? ::: Low-latency Key-Value stores (Redis, DynamoDB, Cassandra).
- Which database type is typically used for the "Offline Store"? ::: Scalable Data Warehouses or Data Lakes (BigQuery, Snowflake, S3/Parquet).

## Quizzes

### Debugging Skew
Q: Your model has 90% accuracy in the notebook but 60% in production. You suspect "Logic Skew." What is the most likely cause?
Options:
- A) The model is overfitting.
- B) The production code implements the feature calculation (e.g., rolling average) differently than the Python training code.
- C) The production server is too slow.
- D) The learning rate was too high.
Answers: B
Explanation: Logic skew happens when the implementation of feature engineering differs between environments (e.g., Python vs. SQL/Java). A Feature Store solves this by using a single definition.

### Point-in-Time Correctness
Q: You are predicting if a user will click an ad at 10:00 AM. You have a feature "Number of clicks today." Why is a simple SQL join dangerous?
Options:
- A) SQL is slow.
- B) A simple join might include clicks that happened at 10:05 AM (after the prediction), causing Data Leakage.
- C) It uses too much memory.
- D) SQL cannot calculate sums.
Answers: B
Explanation: This is "Future Leakage." The model learns to predict using data it won't have in reality. You must use an "As-Of" join to get the count *strictly before* 10:00 AM.

## Learning Sources
- [Feast Official Documentation](https://feast.dev/) - The leading open-source feature store.
- [Tecton Blog: What is a Feature Store?](https://www.tecton.ai/blog/what-is-a-feature-store/) - Excellent conceptual overview.
- [Google Cloud: MLOps: Continuous delivery and automation pipelines in machine learning](https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning) - Overview of the MLOps lifecycle.
- [Uber Michelangelo Palette](https://eng.uber.com/michelangelo-palette/) - One of the first industry feature stores.
