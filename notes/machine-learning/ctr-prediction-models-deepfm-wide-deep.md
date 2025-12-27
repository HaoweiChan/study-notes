---
title: "CTR Prediction Models (DeepFM, Wide&Deep)"
date: "2025-12-27"
tags: ["recommender-systems", "deep-learning", "adtech"]
related: []
slug: "ctr-prediction-models-deepfm-wide-deep"
category: "machine-learning"
---

# CTR Prediction Models (DeepFM, Wide&Deep)

## Summary
CTR (Click-Through Rate) prediction models are designed to handle high-dimensional, sparse categorical data (like User IDs and Item IDs) common in AdTech. Key architectures include **Wide & Deep** (combining memorization and generalization) and **DeepFM** (automating feature interaction learning).

## Details

### The Challenge: High Cardinality & Sparsity
In online advertising, input features are often categorical with millions of possible values (e.g., `user_id`, `ad_id`, `site_domain`).
- One-hot encoding creates massive, sparse vectors.
- Linear models (Logistic Regression) scale well but fail to capture complex non-linear interactions without manual feature engineering (e.g., `AND(user_gender=male, ad_category=sports)`).
- Standard DNNs capture interactions but can over-generalize and miss specific "memorized" rules.

### Wide & Deep Learning (Google, 2016)
Combines two components trained jointly:
1.  **Wide Component (Linear Model)**:
    - Uses raw sparse features and manually engineered **Cross-Product Transformations** (e.g., "User installed App X AND Impression is App Y").
    - **Goal**: Memorization. Good for specific, frequent co-occurrences.
2.  **Deep Component (Feed-Forward NN)**:
    - Uses low-dimensional, dense **Embeddings** for sparse features.
    - Passes embeddings through hidden layers (ReLU).
    - **Goal**: Generalization. Good for unseen combinations.

**Formula**: $P(Y=1|x) = \sigma(w_{wide}^T \[x, \phi(x)\] + w_{deep}^T a^{(lf)} + b)$

### DeepFM (Huawei, 2017)
Improves on Wide & Deep by removing the need for manual feature engineering in the "Wide" part.
- Replaces the Linear/Wide component with a **Factorization Machine (FM)**.
- **FM Component**: Learns 2nd-order feature interactions using dot products of latent vectors.
- **Deep Component**: Standard DNN for high-order interactions.
- **Key Innovation**: The FM and Deep parts **share the same input embeddings**. This means the embeddings are trained to serve both low-order interaction (FM) and high-order non-linear abstraction (Deep).

### Other Notable Architectures
- **DCN (Deep & Cross Network)**: Uses a specific "Cross Network" layer to explicitly apply feature crossing at each layer, learning bounded-degree interactions efficiently without manual engineering.
- **Two-Tower (DSSM)**: Primarily for **Retrieval** (finding top K candidates from millions), not Ranking. Independent User Tower and Item Tower output embeddings, similarity is calculated via Dot Product.

## Examples / snippets

### Simplified Wide & Deep Structure (PyTorch-style pseudo-code)

```python
import torch
import torch.nn as nn

class WideAndDeep(nn.Module):
    def __init__(self, num_features, embedding_dims, hidden_units):
        super().__init__()
        # Wide part: Linear layer for sparse features (or cross-products)
        self.wide = nn.Linear(num_features, 1)
        
        # Deep part: Embeddings + MLP
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_feat, dim) for num_feat, dim in embedding_dims
        ])
        
        input_dim = sum([dim for _, dim in embedding_dims])
        layers = []
        for hidden in hidden_units:
            layers.append(nn.Linear(input_dim, hidden))
            layers.append(nn.ReLU())
            input_dim = hidden
        self.deep_mlp = nn.Sequential(*layers)
        self.deep_out = nn.Linear(input_dim, 1)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_wide, x_deep_indices):
        # x_wide: binary features or cross products
        wide_out = self.wide(x_wide)
        
        # x_deep_indices: categorical indices for embeddings
        embedded = [emb(x_deep_indices[:, i]) for i, emb in enumerate(self.embeddings)]
        deep_in = torch.cat(embedded, dim=1)
        deep_out = self.deep_out(self.deep_mlp(deep_in))
        
        # Combine logits
        return self.sigmoid(wide_out + deep_out)
```

## Flashcards

- What is the main goal of the "Wide" component in Wide & Deep? ::: To **memorize** specific feature co-occurrences (e.g., specific rules).
- What is the main goal of the "Deep" component in Wide & Deep? ::: To **generalize** to unseen feature combinations using embeddings.
- How does DeepFM improve upon Wide & Deep? ::: It replaces the manual feature engineering of the Wide part with a **Factorization Machine (FM)** to learn 2nd-order interactions automatically.
- What is the shared input mechanism in DeepFM? ::: The FM component and the Deep component share the same **Embedding vectors** for the raw features.
- Which architecture is typically used for the Retrieval stage (candidate generation)? ::: **Two-Tower** (or DSSM) architecture.

## Quizzes

### Model Selection
Q: You have a dataset where specific combinations of features (e.g., "City=Paris" AND "Language=French") are highly predictive, but you also want the model to recommend relevant items to users with little history. Which architecture helps balance these needs?
Options:
- A) Logistic Regression
- B) Matrix Factorization
- C) Wide & Deep
- D) K-Nearest Neighbors
Answers: C
Explanation: Wide & Deep is explicitly designed to balance memorization (Wide part for specific rules like City+Language) and generalization (Deep part for new users/items via embeddings).

### DeepFM Architecture
Q: In DeepFM, how are 2nd-order feature interactions handled?
Options:
- A) By manually creating cross-product features
- B) By a Factorization Machine component using dot products of embeddings
- C) By the MLP (Multi-Layer Perceptron) hidden layers only
- D) By a Recurrent Neural Network
Answers: B
Explanation: DeepFM uses a Factorization Machine layer to explicitly model 2nd-order interactions ($<v_i, v_j> x_i x_j$) using the shared embeddings, without manual engineering.

## Learning Sources
- [Wide & Deep Learning for Recommender Systems (arXiv)](https://arxiv.org/abs/1606.07792) - The original paper by Google (2016).
- [DeepFM: A Factorization-Machine based Neural Network (arXiv)](https://arxiv.org/abs/1703.04247) - The original paper by Huawei/IJCAI (2017).
- [TensorFlow Recommenders](https://www.tensorflow.org/recommenders) - Official library and tutorials for building retrieval and ranking models.
- [System Design for Recommendations (Eugene Yan)](https://eugeneyan.com/writing/system-design-for-discovery/) - High-level overview of where these models fit in a production system.
