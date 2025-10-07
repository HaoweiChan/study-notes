---
title: "Union-Find DSU"
date: "2025-10-07"
tags: []
related: []
slug: "union-find-dsu"
category: "algo"
---

# Union-Find DSU

## Summary
Union-Find (Disjoint Set Union/DSU) is a data structure that efficiently manages disjoint sets with near-constant time operations for finding set representatives and merging sets, commonly used for connected components, cycle detection, and Kruskal's MST algorithm.

## Details
### 🧠 What is Union-Find?

Union-Find (also called Disjoint Set Union, or DSU) is a data structure used to efficiently manage a collection of disjoint sets, supporting two main operations:
- **find(x):** Find the representative (root) of the set that x belongs to.
- **union(x, y):** Merge the sets that contain x and y.

With path compression and union by rank/size, both operations run in nearly constant time: O(α(n)), where α is the inverse Ackermann function (very slow-growing).

### ✅ When to Use Union-Find
- You need to keep track of connected components in a graph
- You want to detect cycles in an undirected graph
- You're merging elements under equivalence relations
- Optimizing Kruskal's algorithm for Minimum Spanning Tree (MST)

### 📂 LeetCode Problems Using Union-Find / DSU

| Problem | Description |
|---------|-------------|
| **547. Number of Provinces** | Count connected components in a graph |
| **684. Redundant Connection** | Find edge that creates a cycle |
| **1319. Number of Operations to Make Network Connected** | Count extra edges and components |
| **128. Longest Consecutive Sequence** | Can be solved using Union-Find |
| **990. Satisfiability of Equality Equations** | Merge variables with equal constraints |
| **839. Similar String Groups** | Treat strings as nodes and group similar ones |
| **952. Largest Component Size by Common Factor** | Union nodes based on common prime factor |

## Examples / snippets

### Basic Union-Find Implementation
```python
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [1] * n  # Alternatively use size

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]

    def union(self, x, y):
        rootX = self.find(x)
        rootY = self.find(y)

        if rootX == rootY:
            return False  # already in the same set

        # Union by rank
        if self.rank[rootX] < self.rank[rootY]:
            self.parent[rootX] = rootY
        elif self.rank[rootX] > self.rank[rootY]:
            self.parent[rootY] = rootX
        else:
            self.parent[rootY] = rootX
            self.rank[rootX] += 1

        return True
```

### Union-Find with Size Tracking
```python
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.size = [1] * n
        self.components = n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        rootX = self.find(x)
        rootY = self.find(y)

        if rootX == rootY:
            return False

        # Union by size
        if self.size[rootX] < self.size[rootY]:
            self.parent[rootX] = rootY
            self.size[rootY] += self.size[rootX]
        else:
            self.parent[rootY] = rootX
            self.size[rootX] += self.size[rootY]

        self.components -= 1
        return True

    def get_size(self, x):
        return self.size[self.find(x)]

    def get_components(self):
        return self.components
```

### Example Usage - Connected Components
```python
# LeetCode 547. Number of Provinces
def find_circle_num(is_connected):
    n = len(is_connected)
    uf = UnionFind(n)

    for i in range(n):
        for j in range(i + 1, n):
            if is_connected[i][j]:
                uf.union(i, j)

    return uf.get_components()

# Example
is_connected = [[1,1,0],[1,1,0],[0,0,1]]
result = find_circle_num(is_connected)
print(result)  # Expected: 2
```

### Example Usage - Cycle Detection
```python
# LeetCode 684. Redundant Connection
def find_redundant_connection(edges):
    n = len(edges)
    uf = UnionFind(n)

    for u, v in edges:
        if not uf.union(u - 1, v - 1):  # 0-indexed
            return [u, v]
    return []

# Example
edges = [[1,2],[1,3],[2,3]]
result = find_redundant_connection(edges)
print(result)  # Expected: [2,3]
```

### Kruskal's MST with Union-Find
```python
def kruskal_mst(n, edges):
    """
    edges: list of [u, v, weight]
    Returns total weight of MST or -1 if disconnected
    """
    uf = UnionFind(n)
    edges.sort(key=lambda x: x[2])  # Sort by weight

    total_weight = 0
    edges_used = 0

    for u, v, weight in edges:
        if uf.union(u, v):
            total_weight += weight
            edges_used += 1
            if edges_used == n - 1:
                return total_weight

    return -1  # Graph is not connected

# Example
n = 4
edges = [[0,1,10],[0,2,6],[0,3,5],[1,3,15],[2,3,4]]
result = kruskal_mst(n, edges)
print(result)  # Expected: 19 (edges 0-3, 2-3, 0-2)
```

### Union-Find with Path Compression (Optimized)
```python
class OptimizedUnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.size = [1] * n

    def find(self, x):
        # Path compression with recursion
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        rootX = self.find(x)
        rootY = self.find(y)

        if rootX == rootY:
            return False

        # Union by rank (smaller rank tree attaches to larger)
        if self.rank[rootX] < self.rank[rootY]:
            self.parent[rootX] = rootY
            self.size[rootY] += self.size[rootX]
        elif self.rank[rootX] > self.rank[rootY]:
            self.parent[rootY] = rootX
            self.size[rootX] += self.size[rootY]
        else:
            self.parent[rootY] = rootX
            self.size[rootX] += self.size[rootY]
            self.rank[rootX] += 1

        return True

    def get_size(self, x):
        return self.size[self.find(x)]
```
```
