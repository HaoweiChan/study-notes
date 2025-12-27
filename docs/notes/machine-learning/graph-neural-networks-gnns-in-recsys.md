---
title: "Graph Neural Networks (GNNs) in RecSys"
date: "2025-12-27"
tags: ["graph-learning", "recommender-systems", "deep-learning"]
related: []
slug: "graph-neural-networks-gnns-in-recsys"
category: "machine-learning"
---

# Graph Neural Networks (GNNs) in RecSys

## Summary
Recommender Systems naturally model User-Item interactions as a Bipartite Graph. **GNNs** (like **GraphSAGE** and **PinSage**) outperform traditional Matrix Factorization by explicitly aggregating information from a user's neighbors (items they liked) and neighbors-of-neighbors to generate rich embeddings.

## Details

### 1. From Matrix Factorization to Graphs
- **MF**: Learns a static embedding for User $U$ and Item $I$ based on direct interaction.
- **Limitation**: Ignores higher-order connectivity. (e.g., User A and B bought the same Item I; they are related).
- **GNN**: Propagates information. User embedding is an aggregation of the Items they interacted with.

### 2. Core Mechanism: Message Passing
1.  **Message**: Neighbors send their feature vectors to the target node.
2.  **Aggregation**: Target node sums/averages these messages.
3.  **Update**: Target node updates its own embedding using the aggregated message + its previous state.
4.  **Layers**: 2 layers = information from 2-hop neighbors (Friends of Friends).

### 3. Key Architectures

#### A. GraphSAGE (Sample and AggreGatE)
- Instead of training an embedding for every node (transductive), train an **Aggregator Function** (inductive).
- Can generate embeddings for *new* nodes (unseen users) using their features and neighbors.
- **Sampling**: Aggregates from a fixed-size sample of neighbors (e.g., 10) to keep compute constant.

#### B. PinSage (Pinterest)
- Scaled GNNs to billions of nodes.
- **Random Walks**: Instead of 1-hop neighbors, uses Random Walks to define "Importance" (visit count) of neighbors.
- **Importance Pooling**: Aggregates only the most "important" neighbors found by random walks.
- **Hard Negatives**: Curriculum learning with hard negatives is crucial for performance.

## Examples / snippets

### GraphSAGE Aggregation (Pseudo-code)

```python
import torch
import torch.nn as nn

class GraphSAGE(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim * 2, out_dim)

    def forward(self, node_feats, adj_list):
        # 1. Aggregate Neighbors
        # For each node, get features of neighbors and average them
        # (Simplified: assume fixed size neighbors tensor)
        neighbor_feats = get_neighbors(node_feats, adj_list) # [N, samples, D]
        agg_neighbor = torch.mean(neighbor_feats, dim=1)     # [N, D]
        
        # 2. Concat with Self
        combined = torch.cat([node_feats, agg_neighbor], dim=1) # [N, 2D]
        
        # 3. Update
        return torch.relu(self.linear(combined))
```

## Learning Sources
- [GraphSAGE Paper](https://arxiv.org/abs/1706.02216) - "Inductive Representation Learning on Large Graphs".
- [PinSage Paper (KDD 2018)](https://arxiv.org/abs/1806.01973) - "Graph Convolutional Neural Networks for Web-Scale Recommender Systems".
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/) - The standard library for GNNs.
- [Standford CS224W](http://web.stanford.edu/class/cs224w/) - The best course on Machine Learning with Graphs.
