---
title: "Handling Sparse Data (Embeddings, Hashing Trick)"
date: "2025-12-27"
tags: ["feature-engineering", "adtech", "deep-learning"]
related: []
slug: "handling-sparse-data-embeddings-hashing-trick"
category: "machine-learning"
---

# Handling Sparse Data (Embeddings, Hashing Trick)

## Summary
In domains like AdTech and NLP, data is often extremely sparse and high-dimensional (e.g., millions of User IDs or words). Standard One-Hot Encoding is computationally infeasible. Solutions include **Embeddings** (dense vector representations) and the **Hashing Trick** (fixed-size vector mapping) to handle scale and memory constraints.

## Details

### 1. The Challenge: High Cardinality
If you have a feature `user_id` with 10 million unique values:
- **One-Hot Encoding**: Creates a vector of size 10M.
    - Memory: $10^7$ floats per sample $\rightarrow$ Impossible.
    - Computation: Matrix multiplication is huge.
    - Generalization: No relationship between User A and User B.

### 2. Embeddings (Dense Representation)
Map each categorical ID to a low-dimensional dense vector (e.g., size 64).
- **Lookup Table**: A matrix of size $N \times D$ (e.g., $10M \times 64$).
- **Learning**: The vectors are learnable parameters. Similar users (who click similar ads) will have similar vectors (close in Euclidean space).
- **Memory**: Much smaller than One-Hot, but still large if $N$ is huge.

### 3. The Hashing Trick (Feature Hashing)
Used when the vocabulary size $N$ is unknown or too large to store an Embedding Table.
- **Mechanism**: Apply a hash function to the raw feature value and modulo by a fixed size $K$.
    - $index = hash("user\_12345") \% K$
    - $x\[index\] = 1$
- **Pros**:
    - Fixed memory usage ($K$).
    - No need to maintain a vocabulary dictionary (stateless).
    - Handles new values automatically.
- **Cons**: **Collisions**. Different users might hash to the same index.
    - *Impact*: Surprisingly low impact on model performance if $K$ is sufficiently large (e.g., $2^{20}$). Deep networks are robust to some noise.

### 4. Crossed Features
Combining two sparse features (e.g., `City` and `Job`) creates an even sparser feature.
- Cardinality: $|City| \times |Job|$.
- Hashing is almost mandatory here.

## Examples / snippets

### Feature Hashing in Scikit-Learn

```python
from sklearn.feature_extraction import FeatureHasher

# Raw data (dictionaries)
data = [
    {'user': 'u1', 'city': 'NY', 'device': 'mobile'},
    {'user': 'u2', 'city': 'SF', 'device': 'desktop'},
    {'user': 'u3', 'city': 'NY', 'device': 'tablet'},
]

# Hash to a fixed vector size of 10
h = FeatureHasher(n_features=10, input_type='dict')
f = h.transform(data)

print(f.toarray())
# Output: Sparse matrix with values at hashed indices
```

### Hashing Trick Implementation (Concept)

```python
def get_hashed_index(feature_string, vector_size=1024):
    """
    Returns the index for a feature string using hashing.
    """
    # Use a fast hash function like MurmurHash (simulated here with hash())
    h = hash(feature_string)
    # Modulo to fit in vector size
    return abs(h) % vector_size

# Example
idx1 = get_hashed_index("user_id=12345")
idx2 = get_hashed_index("user_id=67890")
# These indices are fed into an EmbeddingBag or Sparse Linear Layer
```

## Learning Sources
- [Weinberger et al.: Feature Hashing for Large Scale Multitask Learning](https://arxiv.org/abs/0902.2206) - The seminal paper on the hashing trick.
- [Google Crash Course: Embeddings](https://developers.google.com/machine-learning/crash-course/embeddings/video-lecture) - Visual guide to embeddings.
- [TensorFlow: Feature Columns](https://www.tensorflow.org/api_docs/python/tf/feature_column) - Documentation on `hashed_categorical_column`.
- [Scikit-Learn: Feature Hashing](https://scikit-learn.org/stable/modules/feature_extraction.html#feature-hashing) - Practical guide.
