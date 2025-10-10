---
title: "2D Prefix Sum"
date: "2025-10-07"
tags: []
related: []
slug: "2d-prefix-sum"
category: "algorithm"
---

# 2D Prefix Sum

## Summary
2D Prefix Sum (summed-area table) is a preprocessing technique that allows O(1) time queries for the sum of any submatrix in a 2D array by precomputing cumulative sums, commonly used for efficient range sum queries in grid problems.

## Details
### 🎯 Goal
Given a matrix of size `m x n`, create a prefix_sum matrix where each cell `(i, j)` contains the sum of the submatrix from `(0, 0)` to `(i, j)`.

### ✅ Formula
For cell `(i, j)` in the prefix sum matrix:

```python
prefix_sum[i][j] = matrix[i][j]
                 + prefix_sum[i-1][j]
                 + prefix_sum[i][j-1]
                 - prefix_sum[i-1][j-1]
```

**Note:** The last term is subtracted to remove the double-counted area.

### 📍 To Query a Submatrix Sum (r1, c1) to (r2, c2)
You can use the prefix sum to quickly get any submatrix sum in O(1) time:

```python
def query_sum(prefix, r1, c1, r2, c2):
    total = prefix[r2][c2]
    if r1 > 0:
        total -= prefix[r1 - 1][c2]
    if c1 > 0:
        total -= prefix[r2][c1 - 1]
    if r1 > 0 and c1 > 0:
        total += prefix[r1 - 1][c1 - 1]
    return total
```

## Examples / snippets

### Implementation in Python
```python
def compute_2d_prefix_sum(matrix):
    if not matrix or not matrix[0]:
        return []

    m, n = len(matrix), len(matrix[0])
    prefix_sum = [[0]*n for _ in range(m)]

    for i in range(m):
        for j in range(n):
            top = prefix_sum[i-1][j] if i > 0 else 0
            left = prefix_sum[i][j-1] if j > 0 else 0
            topleft = prefix_sum[i-1][j-1] if i > 0 and j > 0 else 0

            prefix_sum[i][j] = matrix[i][j] + top + left - topleft

    return prefix_sum
```

### Complete Implementation with Query Function
```python
class PrefixSum2D:
    def __init__(self, matrix):
        self.matrix = matrix
        self.m, self.n = len(matrix), len(matrix[0]) if matrix else (0, 0)
        self.prefix = self._compute_prefix()

    def _compute_prefix(self):
        if not self.matrix or not self.matrix[0]:
            return []

        prefix = [[0] * self.n for _ in range(self.m)]

        for i in range(self.m):
            for j in range(self.n):
                top = prefix[i-1][j] if i > 0 else 0
                left = prefix[i][j-1] if j > 0 else 0
                topleft = prefix[i-1][j-1] if i > 0 and j > 0 else 0

                prefix[i][j] = self.matrix[i][j] + top + left - topleft

        return prefix

    def query(self, r1, c1, r2, c2):
        """Query sum from (r1, c1) to (r2, c2) inclusive"""
        if r1 < 0 or r1 >= self.m or c1 < 0 or c1 >= self.n:
            return 0
        if r2 < 0 or r2 >= self.m or c2 < 0 or c2 >= self.n:
            return 0

        total = self.prefix[r2][c2]
        if r1 > 0:
            total -= self.prefix[r1 - 1][c2]
        if c1 > 0:
            total -= self.prefix[r2][c1 - 1]
        if r1 > 0 and c1 > 0:
            total += self.prefix[r1 - 1][c1 - 1]
        return total
```

### Example Usage
```python
matrix = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]

# Using functional approach
prefix = compute_2d_prefix_sum(matrix)
for row in prefix:
    print(row)
# Output:
# [1, 3, 6]
# [5, 12, 21]
# [12, 27, 45]

# Query sum from (0,0) to (1,1)
def query_sum(prefix, r1, c1, r2, c2):
    total = prefix[r2][c2]
    if r1 > 0:
        total -= prefix[r1 - 1][c2]
    if c1 > 0:
        total -= prefix[r2][c1 - 1]
    if r1 > 0 and c1 > 0:
        total += prefix[r1 - 1][c1 - 1]
    return total

result = query_sum(prefix, 0, 0, 1, 1)
print(f"Sum from (0,0) to (1,1): {result}")  # Expected: 12 (1+2+4+5)

# Using class approach
prefix_obj = PrefixSum2D(matrix)
result = prefix_obj.query(1, 1, 2, 2)
print(f"Sum from (1,1) to (2,2): {result}")  # Expected: 5+6+8+9 = 28
```

### Edge Cases and Validation
```python
# Empty matrix
empty_matrix = []
prefix_empty = compute_2d_prefix_sum(empty_matrix)
print(prefix_empty)  # Expected: []

# Single cell matrix
single_cell = [[5]]
prefix_single = compute_2d_prefix_sum(single_cell)
print(prefix_single)  # Expected: [[5]]

# Single row matrix
single_row = [[1, 2, 3, 4]]
prefix_row = compute_2d_prefix_sum(single_row)
print(prefix_row)  # Expected: [[1, 3, 6, 10]]

# Single column matrix
single_col = [[1], [2], [3]]
prefix_col = compute_2d_prefix_sum(single_col)
print(prefix_col)
# Expected:
# [[1],
#  [3],
#  [6]]

# Query entire matrix
total_sum = query_sum(prefix, 0, 0, 2, 2)
print(f"Total matrix sum: {total_sum}")  # Expected: 45 (1+2+3+4+5+6+7+8+9)
```
```
