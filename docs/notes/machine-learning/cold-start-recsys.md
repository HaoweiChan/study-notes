---
title: "Cold Start Strategies in RecSys"
date: "2025-12-27"
category: "machine-learning"
tags: ["recommender-systems", "cold-start", "machine-learning", "bandits"]
related: ["multi-armed-bandits-thompson-sampling-ucb", "ctr-prediction-models-deepfm-wide-deep"]
slug: "cold-start-recsys"
flashcards:
  - q: "What is the Cold Start Problem in Recommender Systems?"
    a: "The inability of the system to make accurate recommendations for new users or new items due to a lack of interaction history."
  - q: "What is the difference between User Cold Start and Item Cold Start?"
    a: "User Cold Start: New user with no history. Item Cold Start: New item that no one has interacted with."
  - q: "How does Content-Based Filtering help with Cold Start?"
    a: "It relies on item/user metadata (e.g., genre, age) rather than interaction history, allowing predictions for new entities."
  - q: "What is the 'DropoutNet' technique for cold start?"
    a: "A training technique where input embeddings are randomly dropped out to force the model to learn from content features alone."
quizzes:
  - q: "Which algorithm is most effective for quickly learning the quality of a new ad creative (Item Cold Start)?"
    options: ["Matrix Factorization", "Multi-Armed Bandit (e.g., Thompson Sampling)", "Collaborative Filtering", "K-Means Clustering"]
    answers: [1]
    explanation: "Bandits explicitly balance exploration (showing the new ad to gather data) and exploitation (showing best ads), minimizing regret."
  - q: "For a brand new user with NO data, what is the most robust fallback strategy?"
    options: ["User-Based CF", "Item-Based CF", "Global Popularity / Trending", "Matrix Factorization"]
    answers: [2]
    explanation: "Without any signal, recommending globally popular items provides the highest probability of engagement."
---

# Cold Start Strategies in RecSys

## Summary
The **Cold Start Problem** occurs when a recommender system lacks sufficient interaction data (clicks, views) to infer preferences for new users or items. In AdTech, handling **Item Cold Start** (new ads) is critical for campaign performance, often solved via Exploration (Bandits) or Content-Based methods.

## Details

### 1. Types of Cold Start
*   **User Cold Start:** A new user arrives. We don't know their preferences.
*   **Item Cold Start:** A new item/ad is added. We don't know its quality (CTR).
*   **System Cold Start:** A new platform with no data at all.

### 2. Strategies & Solutions

#### A. Heuristics & Baselines
*   **Global Popularity:** Recommend "Top 10" items. Robust but not personalized.
*   **Demographic/Geographic:** If we know user is from "US" and uses "iOS", recommend items popular in that segment.

#### B. Content-Based Filtering (CBF)
*   Use **Metadata** instead of interactions.
*   *Example:* For a new movie, use its genre, director, and actors to map it to similar movies in the embedding space.
*   *Architecture:* **Two-Tower Models** can take (User Features) and (Item Features) as inputs. If ID embedding is missing, the model relies on the feature tower.

#### C. Multi-Armed Bandits (Exploration)
*   Crucial for **Item Cold Start** (New Ads).
*   The system must "pay" a cost (potential low CTR) to "explore" the new item.
*   **Thompson Sampling** or **UCB** assigns a probabilistic score to new items to ensure they get impressions.

#### D. Model-Based Techniques
*   **DropoutNet:** During training, randomly zero out the User/Item ID embeddings. This forces the neural network to learn weights for the *content features* (metadata), making it robust when IDs are missing at inference time.
*   **Transfer Learning:** Pre-train embeddings on a dense dataset (e.g., Search history) and fine-tune on the sparse target (e.g., Video Watch).

## Examples / snippets

### DropoutNet Concept (PyTorch)

```python
import torch
import torch.nn as nn

class DropoutNet(nn.Module):
    def __init__(self, id_dim, feature_dim, hidden_dim):
        super().__init__()
        self.id_embedding = nn.Embedding(1000, id_dim)
        self.feature_encoder = nn.Linear(feature_dim, id_dim)
        self.fc = nn.Linear(id_dim * 2, 1)
        self.dropout = nn.Dropout(p=0.5) # The key: dropout

    def forward(self, user_id, item_features, use_id=True):
        # 1. Get ID embedding (or zero if cold start)
        if use_id:
            id_emb = self.id_embedding(user_id)
            # Apply dropout specifically to input embedding to simulate cold start during training
            id_emb = self.dropout(id_emb) 
        else:
            id_emb = torch.zeros_like(self.id_embedding(torch.tensor(0)))

        # 2. Get Feature embedding
        feat_emb = self.feature_encoder(item_features)

        # 3. Combine
        combined = torch.cat([id_emb, feat_emb], dim=1)
        return torch.sigmoid(self.fc(combined))
```

## Learning Sources
*   [Google Engineers: DropoutNet Paper](https://www.cs.cornell.edu/~volkovs/nips2017_dropoutnet.pdf) - Addressing cold start in Recommender Systems.
*   [Eugene Yan: System Design for Recommendations](https://eugeneyan.com/writing/system-design-for-discovery/) - Practical guide including cold start.
*   [Netflix Tech Blog: Warm Start](https://netflixtechblog.com/) - How Netflix handles new users.
