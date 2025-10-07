---
title: "Graph Search Algorithms"
date: "2025-10-07"
tags: []
related: []
slug: "graph-search-algorithms"
category: "algo"
---

# Graph Search Algorithms

## Summary
Comprehensive overview of essential graph search algorithms including BFS, DFS, 0-1 BFS, Dijkstra's algorithm, Union-Find (DSU), and A* search, covering their implementations, use cases, and time complexities.

## Details
### BFS (Breadth-First Search)

**What is BFS?**

BFS is a graph traversal that explores nodes level by level from a start node. It is mainly used for shortest path in unweighted graphs and multi-source spreading, achieving efficiency by processing a FIFO frontier.

- **Time Complexity:** O(V + E)
- **Space Complexity:** O(V) (frontier + visited)

**How it Works**
1. Initialize queue & distance - Put start node(s) in a queue; mark distance 0
2. Expand by layers - Pop from the queue; push unvisited neighbors; set dist\[neighbor] = dist\[u] + 1
3. Early exit for targets - If a target is found, its first discovery distance is minimal (by edges)

**When to Use BFS**
- Use when the graph is unweighted and you need fewest edges / minimum steps
- Use for multi-source flood/fill by seeding the queue with multiple starts
- Not ideal when edges have weights > 1

**Common Problem Categories**
- Shortest path (unweighted), minimum steps, nearest target
- Multi-source propagation (rotting, walls/gates)
- Level-order traversal in graphs/trees

### DFS (Depth-First Search)

**What is DFS?**

DFS is a graph traversal that explores as far as possible along a branch before backtracking. It is mainly used for component counting, cycle detection, topological sort (postorder), and backtracking enumeration, leveraging a stack/recursion.

- **Time Complexity:** O(V + E)
- **Space Complexity:** O(V) (recursion/stack)

**How it Works**
1. Go deep first - From the start, visit an unvisited neighbor and recurse until you can't
2. Mark & backtrack - Mark nodes as visited; when stuck, backtrack to explore alternatives
3. Postorder aggregation - Combine results from children after recursive returns (useful on DAGs/trees)

**When to Use DFS**
- Use for counting components/areas, cycle detection, topological sort, SCCs, backtracking
- Prefer iterative DFS in Python for very deep graphs to avoid recursion limits
- Not ideal for shortest path in unweighted graphs (use BFS) or weighted graphs (use Dijkstra/A*)

**Common Problem Categories**
- Number of Islands, Max Area, connected components
- Topological sort, cycle detection, SCC (Tarjan/Kosaraju)
- Backtracking (paths, permutations, subsets)

### 0–1 BFS

**What is 0–1 BFS?**

0–1 BFS solves shortest paths on graphs whose edge weights are only 0 or 1. It is mainly used for "min flips/breaks" style problems and achieves efficiency by using a deque as a two-level priority queue.

- **Time Complexity:** O(V + E)
- **Space Complexity:** O(V)

**How it Works**
1. Deque with two priorities - Keep nodes in a deque; current layer at the front (distance d), next layer at the back (d+1)
2. Relax 0-edges to front - If an improved edge has weight 0, push the neighbor to the front (same distance layer)
3. Relax 1-edges to back - If weight is 1, push to the back (distance +1)

**When to Use 0–1 BFS**
- Use when all edge weights are 0 or 1
- Great for grids with "free move" vs "costly move" edges
- Not valid if any edge has weight > 1 (use Dijkstra)

**Common Problem Categories**
- Minimum walls to break / flips to make
- Teleport (0) vs walk (1) problems
- Toggle/bitwise operations modeled as 0/1 costs

### Dijkstra's Algorithm

**What is Dijkstra's Algorithm?**

Dijkstra computes shortest paths from a source in graphs with non-negative edge weights. It is mainly used for general weighted shortest paths, using a min-heap to always expand the smallest tentative distance.

- **Time Complexity:** O((V + E) log V) (binary heap)
- **Space Complexity:** O(V)

**How it Works**
1. Initialize distances & heap - Set dist\[src]=0, others ∞; push (0, src) into a min-heap
2. Extract-min & relax - Pop (d,u). If stale (d != dist\[u]), skip. For each (u→v,w≥0), relax: if d+w < dist\[v], update and push
3. Early exit (optional) - If you only need the distance to a single target, you can stop when the target is popped

**When to Use Dijkstra**
- Use when edges have arbitrary non-negative weights (integers or reals)
- Early exit is fine for single-target shortest path
- Not valid if negative weights exist (use Bellman–Ford/Johnson)

**Common Problem Categories**
- Network delays, routing, travel time/cost
- Path With Minimum Effort (weighted grid)
- Road networks, maps (sometimes with A* heuristic)

### Union–Find (Disjoint Set Union, DSU)

**What is Union–Find?**

Union–Find (DSU) maintains a partition of elements into disjoint sets, supporting quick merge (union) and find representative operations. It is mainly used for connectivity queries and cycle detection and achieves efficiency via path compression + union by size/rank.

- **Time Complexity:** ~O(α(N)) per op (inverse Ackermann; effectively constant)
- **Space Complexity:** O(N)

**How it Works**
1. Each node starts alone - parent\[x] = x, size\[x] = 1
2. Find with compression - find(x) returns the set representative and flattens paths for speed
3. Union by size/rank - Attach the smaller tree under the larger; update sizes and component count

**When to Use Union–Find**
- Use for "Are u and v connected?" queries (many queries)
- Kruskal's MST, cycle detection in undirected graphs
- Dynamic edge additions (no deletions). Not for shortest paths

**Common Problem Categories**
- Connected components, merging accounts, equations with equality
- Offline connectivity under thresholds (sort edges + union)
- MST building (Kruskal)

### A* Search

**What is A* Search?**

A* finds shortest paths like Dijkstra but uses a heuristic to guide the search toward the goal. It is mainly used for single-source single-target routing on spatial graphs, achieving speedups by prioritizing nodes with f = g + h.

- **Time Complexity:** O((V + E) log V) worst-case (often much faster in practice)
- **Space Complexity:** O(V)

**How it Works**
1. g-score and heuristic h - g\[u] = best-known distance from source; h\[u] = admissible heuristic estimate to target
2. Priority by f = g + h - Expand nodes in increasing f; if h never overestimates, first pop of target is optimal
3. Relax neighbors - Standard relax like Dijkstra, but priority uses g + h

**When to Use A***
- Use when you have a good admissible heuristic and a specific target
- Particularly useful in maps/grids (use straight-line or Manhattan distance)
- Not ideal if no meaningful heuristic is available (falls back to Dijkstra)

**Common Problem Categories**
- Pathfinding on maps/grids
- Game AI navigation
- Robot motion planning

### Quick Chooser Recap
- **Unweighted shortest steps?** → BFS
- **Only 0/1 weights?** → 0–1 BFS
- **General non-negative weights?** → Dijkstra (or A* if good heuristic + single target)
- **Connectivity / many union queries?** → Union–Find
- **Counting/aggregation/topo/backtracking?** → DFS

## Examples / snippets

### BFS Implementation
```python
from collections import deque

def bfs(start, is_target, neighbors):
    q = deque([start])
    dist = {start: 0}
    while q:
        u = q.popleft()
        if is_target(u):
            return dist[u]
        for v in neighbors(u):
            if v not in dist:
                dist[v] = dist[u] + 1
                q.append(v)
    return -1  # not found

# Example usage
start = 0
is_target = lambda x: x == 7
d = bfs(start, is_target, neighbors)
print(d)  # Expected: minimal number of edges from start to node 7
```

### DFS Implementation
```python
def dfs_iter(start, neighbors):
    stack = [start]
    seen = {start}
    order = []
    while stack:
        u = stack.pop()
        order.append(u)
        for v in neighbors(u):
            if v not in seen:
                seen.add(v)
                stack.append(v)
    return order

# Example usage
order = dfs_iter(0, neighbors)
print(order)  # Expected: a depth-first visitation order
```

### 0-1 BFS Implementation
```python
from collections import deque
INF = 10**18

def zero_one_bfs(n, adj, src):
    # adj[u] = list of (v, w) with w in {0, 1}
    dist = [INF] * n
    dist[src] = 0
    dq = deque([src])
    while dq:
        u = dq.popleft()
        du = dist[u]
        for v, w in adj[u]:
            nd = du + w
            if nd < dist[v]:
                dist[v] = nd
                if w == 0:
                    dq.appendleft(v)
                else:
                    dq.append(v)
    return dist
```

### Dijkstra's Algorithm Implementation
```python
import heapq
INF = 10**18

def dijkstra(n, adj, src):
    # adj[u] = list of (v, w) with w >= 0
    dist = [INF] * n
    dist[src] = 0
    pq = [(0, src)]
    while pq:
        d, u = heapq.heappop(pq)
        if d != dist[u]:
            continue
        for v, w in adj[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(pq, (nd, v))
    return dist
```

### Union-Find (DSU) Implementation
```python
class DSU:
    def __init__(self, n):
        self.parent = list(range(n))
        self.size = [1]*n
        self.count = n  # components

    def find(self, x):
        while x != self.parent[x]:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]
        self.count -= 1
        return True

    def connected(self, a, b):
        return self.find(a) == self.find(b)

# Example usage
dsu = DSU(5)
dsu.union(0,1); dsu.union(1,2)
print(dsu.connected(0,2))  # True
print(dsu.count)           # number of components
```

### A* Search Implementation
```python
import heapq
INF = 10**18

def astar(adj, src, tgt, heuristic):
    g = {src: 0}
    pq = [(heuristic(src), src)]
    parent = {src: None}

    while pq:
        f, u = heapq.heappop(pq)
        if u == tgt:
            # reconstruct path
            path = []
            while u is not None:
                path.append(u)
                u = parent[u]
            return list(reversed(path)), g[tgt]

        for v, w in adj[u]:
            ng = g[u] + w
            if v not in g or ng < g[v]:
                g[v] = ng
                parent[v] = u
                heapq.heappush(pq, (ng + heuristic(v), v))
    return [], INF

# Example usage
path, cost = astar(adj, src=0, tgt=7, heuristic=lambda v: 0.0)
print(cost)  # Expected: shortest path cost (matches Dijkstra when h==0)
```
```
