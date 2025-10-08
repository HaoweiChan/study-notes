---
title: "1135. Connecting Cities with Minimum Cost"
date: "2025-10-08"
tags: ["Union Find", "Graph", "Minimum Spanning Tree", "Greedy"]
related: []
slug: "1135-connecting-cities-with-minimum-cost"
category: "leetcode"
leetcode_url: "https://leetcode.com/problems/connecting-cities-with-minimum-cost/description/?envType=weekly-question&envId=2025-10-01"
leetcode_difficulty: "Medium"
leetcode_topics: ["Union Find", "Graph", "Minimum Spanning Tree", "Greedy"]
---

# 1135. Connecting Cities with Minimum Cost

## Summary
Connect n cities with minimum total connection cost using Kruskal's algorithm and Union-Find, ensuring all cities are connected in a single component with exactly n-1 connections.

## Problem Description
There are `n` cities numbered from `1` to `n`. You are given `connections`, an array where `connections[i] = [city1, city2, cost]` represents a bidirectional connection that can be established between `city1` and `city2` with a cost of `cost`.

Return the minimum cost to connect all the `n` cities. If it is impossible to connect all the `n` cities, return `-1`.

**Constraints:**
- `1 <= n <= 10^4`
- `1 <= connections.length <= 10^5`
- `1 <= city1, city2 <= n`
- `city1 != city2`
- `0 <= cost <= 10^5`

## Solution Approach
This is a classic **Minimum Spanning Tree (MST)** problem solved using **Kruskal's algorithm** with **Union-Find**.

### Kruskal's Algorithm Approach
**Key Insight:** Connect cities with smallest cost edges first, ensuring no cycles are formed.

**Algorithm:**
1. Sort all connections by cost in ascending order
2. Initialize Union-Find with n cities (each city is its own component)
3. Initialize total cost and edges used counter
4. Iterate through sorted connections:
   - If two cities are not in the same component, connect them
   - Add the connection cost to total
   - Increment edges used counter
5. If edges used == n-1, return total cost, otherwise return -1

### Why Kruskal's Algorithm Works
- We want to connect all cities with minimum total cost
- A tree with n nodes has exactly n-1 edges
- By always choosing the smallest available edge that doesn't form a cycle, we ensure minimum total cost
- If we can't form a tree (some cities remain disconnected), it's impossible

## Time & Space Complexity
- **Time Complexity:** O(E log E + E * α(n)) where E is number of connections, α(n) is inverse Ackermann function (nearly constant)
- **Space Complexity:** O(n + E) for Union-Find and connections storage

## Key Insights
- **MST property**: The minimum spanning tree connects all nodes with minimum total edge weight
- **Kruskal's greedy choice**: Always pick the smallest edge that doesn't form a cycle
- **Union-Find efficiency**: Nearly constant time per operation with path compression and union by rank
- **Disconnected graph**: If final components > 1, return -1
- **Self-loops and duplicates**: Should be handled gracefully (though constraints prevent self-loops)

## Examples / snippets

### Solution Code (Kruskal's Algorithm with Union-Find)
```python
from typing import List

class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.components = n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]

    def union(self, x, y):
        rootX, rootY = self.find(x), self.find(y)
        if rootX == rootY:
            return False

        # Union by rank
        if self.rank[rootX] < self.rank[rootY]:
            self.parent[rootX] = rootY
        elif self.rank[rootX] > self.rank[rootY]:
            self.parent[rootY] = rootX
        else:
            self.parent[rootY] = rootX
            self.rank[rootX] += 1

        self.components -= 1
        return True

class Solution:
    def minimumCost(self, n: int, connections: List[List[int]]) -> int:
        # Sort connections by cost
        connections.sort(key=lambda x: x[2])

        uf = UnionFind(n)
        total_cost = 0
        edges_used = 0

        for city1, city2, cost in connections:
            # Convert to 0-based indexing
            city1 -= 1
            city2 -= 1

            # If cities are not connected, connect them
            if uf.union(city1, city2):
                total_cost += cost
                edges_used += 1

                # Early termination: if we have n-1 edges, we're done
                if edges_used == n - 1:
                    return total_cost

        # Check if all cities are connected
        return total_cost if uf.components == 1 else -1
```

### Example Walkthrough
**Example 1:**
```
Input: n = 3, connections = [[1,2,5],[1,3,6],[2,3,1]]
Output: 6

Explanation:
- Sorted connections: [[2,3,1], [1,2,5], [1,3,6]]
- Add [2,3] with cost 1 (edges_used = 1)
- Add [1,2] with cost 5 (edges_used = 2)
- All cities connected, total cost = 6 ✓
```

**Example 2:**
```
Input: n = 4, connections = [[1,2,3],[3,4,4]]
Output: -1

Explanation:
- Sorted connections: [[1,2,3], [3,4,4]]
- Add [1,2] with cost 3 (edges_used = 1)
- Add [3,4] with cost 4 (edges_used = 2)
- Cities 1,2 and 3,4 are in separate components
- Cannot connect all cities, return -1 ✓
```

## Edge Cases & Validation
- **Already connected**: n=1 should return 0 (no connections needed)
- **Minimum connections**: Need exactly n-1 connections for a tree
- **Disconnected graph**: If connections don't form a spanning tree, return -1
- **Duplicate connections**: Should be handled (though constraints prevent duplicates)
- **High cost connections**: Should still work if they form a valid MST
- **Single connection**: n=2 with one connection should return the connection cost

## Related Problems
- [1584. Min Cost to Connect All Points](https://leetcode.com/problems/min-cost-to-connect-all-points/) - Similar MST but with coordinate points
- [1168. Optimize Water Distribution in a Village](https://leetcode.com/problems/optimize-water-distribution-in-a-village/) - MST with virtual node
- [1135. Connecting Cities with Minimum Cost](https://leetcode.com/problems/connecting-cities-with-minimum-cost/) - This problem
- [1192. Critical Connections in a Network](https://leetcode.com/problems/critical-connections-in-a-network/) - Finding bridges in graph
```
