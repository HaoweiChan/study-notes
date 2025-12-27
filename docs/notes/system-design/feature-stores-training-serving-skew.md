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
- **Logic Skew**: The Data Scientist writes `pandas` code for training features, but the Backend Engineer rewrites it in `Java/SQL` for production. Subtle bugs (e.g., handling nulls differently) cause model degradation.
- **Data Latency Skew**: Training used yesterday's batch data, but production uses real-time stream data.
- **Time Travel / Leakage**: Training data accidentally includes "future" information (e.g., calculating "avg spend" using transactions that happened *after* the prediction event).

### 2. What is a Feature Store?
A system that manages feature engineering pipelines and serves them to two targets:
1. **Offline Store** (e.g., BigQuery, S3, Parquet): For generating historical training datasets. Supports "Time Travel" queries.
2. **Online Store** (e.g., Redis, DynamoDB): For low-latency (<10ms) retrieval of the *latest* feature values during real-time inference.

### 3. Key Concepts
- **Feature Registry**: Single source of truth for feature definitions (YAML/Python).
- **Materialization**: The process of computing features and pushing them to the Online Store.
- **Point-in-Time (PIT) Correctness**:
    - When generating training data, we join labels (events) with features.
    - Crucial Rule: For a label at time $T$, we must join feature values known at time $t < T$.
    - Feature Stores automate this complex "As-Of Join".

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

## Learning Sources
- [Feast Official Documentation](https://feast.dev/) - The leading open-source feature store.
- [Tecton Blog: What is a Feature Store?](https://www.tecton.ai/blog/what-is-a-feature-store/) - Excellent conceptual overview.
- [Google Cloud: MLOps: Continuous delivery and automation pipelines in machine learning](https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning) - Overview of the MLOps lifecycle.
- [Uber Michelangelo Palette](https://eng.uber.com/michelangelo-palette/) - One of the first industry feature stores.
