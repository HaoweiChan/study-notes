---
title: "Embedding Search Engines (Vector DBs)"
date: "2025-12-27"
tags: ["system-design", "search", "genai", "vectors"]
related: []
slug: "embedding-search-engines-vector-dbs"
category: "system-design"
---

# Embedding Search Engines (Vector DBs)

## Summary
Embedding Search Engines (or Vector Databases) enable efficient **Approximate Nearest Neighbor (ANN)** search over millions of high-dimensional vectors. They are the backbone of semantic search, RAG, and recommendation systems, using algorithms like **HNSW** and **IVF** to trade accuracy for speed.

## Details

### 1. The Problem: Exact NN is Slow
To find the most similar vector to a query $q$ in a dataset of $N$ vectors ($d$ dimensions), exact search (Brute Force KNN) requires calculating distance to *every* point.
- Complexity: $O(N \cdot d)$.
- For $N=10M, d=1024$, this is too slow (<100ms requirement).

### 2. ANN Algorithms (Indexing)
To speed this up, we build an index that allows us to visit only a fraction of the data.

#### A. HNSW (Hierarchical Navigable Small World)
- **Structure**: A multi-layered graph.
    - Top layers: "Highways" with long-range links (fast traversal across the graph).
    - Bottom layers: "Local roads" for fine-grained search.
- **Mechanism**: Start at the top layer, greedily move to the neighbor closest to the query. Drop down a layer, repeat.
- **Pros**: Extremely fast, high recall.
- **Cons**: Memory intensive (stores the graph structure).

#### B. IVF (Inverted File Index)
- **Structure**: Partition the vector space into $K$ clusters (Voronoi cells) using K-Means.
- **Mechanism**:
    1. Find the closest cluster center to the query.
    2. Search only vectors inside that cluster (and maybe 2-3 neighboring clusters).
- **Pros**: Lower memory usage.
- **Cons**: Accuracy can drop if the query falls on a cluster boundary (mitigated by probing multiple clusters).

### 3. Similarity Metrics
- **Cosine Similarity**: Measures the angle. Good for normalized vectors. $A \cdot B / (\|A\| \|B\|)$.
- **Dot Product**: Measures magnitude and angle. $A \cdot B$. Used when magnitude matters (e.g., Matrix Factorization).
- **Euclidean Distance (L2)**: Measures straight-line distance. $\|A - B\|^2$.

### 4. System Components
- **Faiss (Facebook AI Similarity Search)**: The engine (library) that implements HNSW/IVF. Runs on CPU/GPU.
- **Vector DBs (Pinecone, Milvus, Weaviate)**: Full managed systems wrapping Faiss-like engines with CRUD, filtering, replication, and sharding.

## Examples / snippets

### Faiss Indexing (Python)

```python
import faiss
import numpy as np

# 1. Generate Data
d = 64                           # Dimension
nb = 100000                      # Database size
nq = 10000                       # Queries
xb = np.random.random((nb, d)).astype('float32')
xq = np.random.random((nq, d)).astype('float32')

# 2. Build Index (IVF)
nlist = 100  # Number of clusters (cells)
quantizer = faiss.IndexFlatL2(d)  # Defines distance metric (L2)
index = faiss.IndexIVFFlat(quantizer, d, nlist)

index.train(xb)  # Train K-Means
index.add(xb)    # Add vectors to index

# 3. Search
k = 5
index.nprobe = 10  # Search 10 closest clusters (Trade-off: Speed vs Recall)
D, I = index.search(xq, k) # D: Distances, I: Indices

print(I[:5]) # Top 5 results for first 5 queries
```

## Learning Sources
- [Faiss Wiki](https://github.com/facebookresearch/faiss/wiki) - The bible of vector search.
- [Pinecone: What is a Vector Database?](https://www.pinecone.io/learn/vector-database/) - High-level overview.
- [Hierarchical Navigable Small Worlds (HNSW) Paper](https://arxiv.org/abs/1603.09320) - The algorithm behind most modern vector DBs.
- [Weaviate: Distance Metrics](https://weaviate.io/developers/weaviate/config-refs/distance) - Visualization of Cosine vs L2 vs Dot Product.
