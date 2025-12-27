---
title: "240. Search a 2D Matrix II"
date: "2025-10-08"
tags: ["Binary Search", "Matrix", "Two Pointers"]
related: []
slug: "240-search-a-2d-matrix-ii"
category: "leetcode"
leetcode_url: "https://leetcode.com/problems/search-a-2d-matrix-ii/description/?envType=problem-list-v2&envId=xxzz5vc6"
leetcode_difficulty: "Medium"
leetcode_topics: ["Binary Search", "Matrix", "Two Pointers"]
---

# 240. Search a 2D Matrix II

## Summary
Search for a target value in a 2D matrix where each row and column is sorted, using an efficient O(m+n) approach starting from the top-right corner.

## Problem Description
Write an efficient algorithm that searches for a value `target` in an `m x n` integer matrix `matrix`. This matrix has the following properties:

- Integers in each row are sorted in ascending order from left to right.
- Integers in each column are sorted in ascending order from top to bottom.

**Constraints:**
- `m == matrix.length`
- `n == matrix[i].length`
- `1 <= n, m <= 300`
- `-10^9 <= matrix[i][j] <= 10^9`
- All the integers in each row are sorted in ascending order.
- All the integers in each column are sorted in ascending order.
- `-10^9 <= target <= 10^9`

## Solution Approach
This problem can be solved using several approaches, from O(m*n) to O(m+n):

### Approach 1: Top-Right Corner Search (Optimal)
**Key Insight:** Start from top-right corner. If target is smaller than current element, move left (eliminate current column). If target is larger, move down (eliminate current row).

**Algorithm:**
1. Start from `matrix[0][n-1]` (top-right corner)
2. While within bounds:
   - If `matrix[row][col] == target`, return true
   - If `matrix[row][col] > target`, move left (`col -= 1`)
   - If `matrix[row][col] < target`, move down (`row += 1`)
3. If not found, return false

### Approach 2: Bottom-Left Corner Search
**Key Insight:** Similar logic but starting from bottom-left corner.

### Approach 3: Binary Search on Each Row
**Algorithm:**
1. For each row, perform binary search to find target
2. Return true if found in any row

### Approach 4: Treat as 1D Binary Search
**Key Insight:** Flatten the matrix conceptually and perform binary search.

## Time & Space Complexity
- **Time Complexity:**
  - Top-right corner: O(m + n)
  - Binary search per row: O(m * log n)
  - 1D binary search: O(log(m*n))
- **Space Complexity:** O(1) for all approaches

## Key Insights
- **Matrix properties**: Both rows and columns are sorted, enabling efficient search
- **Top-right strategy**: Each move eliminates either a row or column
- **Corner cases**: Empty matrix, single element, target not present
- **Comparison**: Top-right approach is most efficient for typical cases

## Examples / snippets

### Solution Code (Top-Right Corner)
```python
from typing import List

class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        if not matrix or not matrix[0]:
            return False

        m, n = len(matrix), len(matrix[0])
        row, col = 0, n - 1  # Start from top-right corner

        while row < m and col >= 0:
            if matrix[row][col] == target:
                return True
            elif matrix[row][col] > target:
                # Target is smaller, eliminate current column
                col -= 1
            else:
                # Target is larger, eliminate current row
                row += 1

        return False
```

### Solution Code (Binary Search per Row)
```python
from typing import List
import bisect

class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        for row in matrix:
            # Use bisect to search in sorted row
            if bisect.bisect_left(row, target) < len(row) and row[bisect.bisect_left(row, target)] == target:
                return True
        return False
```

### Solution Code (1D Binary Search)
```python
from typing import List

class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        if not matrix or not matrix[0]:
            return False

        m, n = len(matrix), len(matrix[0])
        left, right = 0, m * n - 1

        while left <= right:
            mid = (left + right) // 2
            # Convert 1D index to 2D coordinates
            row, col = divmod(mid, n)

            if matrix[row][col] == target:
                return True
            elif matrix[row][col] < target:
                left = mid + 1
            else:
                right = mid - 1

        return False
```

### Example Walkthrough
**Example 1:**
```
Input: matrix = [[1,4,7,11,15],[2,5,8,12,19],[3,6,9,16,22],[10,13,14,17,24],[18,21,23,26,30]], target = 5
Output: true

Search process (top-right approach):
- Start at (0,4) = 15
- 15 > 5, move left to (0,3) = 11
- 11 > 5, move left to (0,2) = 7
- 7 > 5, move left to (0,1) = 4
- 4 < 5, move down to (1,1) = 5
- 5 == 5, found! ✓
```

**Example 2:**
```
Input: matrix = [[1,4,7,11,15],[2,5,8,12,19],[3,6,9,16,22],[10,13,14,17,24],[18,21,23,26,30]], target = 20
Output: false

Search process:
- Start at (0,4) = 15
- 15 < 20, move down to (1,4) = 19
- 19 < 20, move down to (2,4) = 22
- 22 > 20, move left to (2,3) = 16
- 16 < 20, move down to (3,3) = 17
- 17 < 20, move down to (4,3) = 26
- 26 > 20, move left to (4,2) = 23
- 23 > 20, move left to (4,1) = 21
- 21 > 20, move left to (4,0) = 18
- 18 < 20, would move down but already at bottom
- Not found ✓
```

## Edge Cases & Validation
- **Empty matrix**: `[]` or `[[]]` should return false
- **Single element**: `[[5]]`, target=5 should return true, target=3 should return false
- **Target at corners**: Should handle top-left, top-right, bottom-left, bottom-right
- **Target smaller than all**: Should terminate at top-left area
- **Target larger than all**: Should terminate at bottom-right area
- **Matrix with one row**: Should work like 1D binary search
- **Matrix with one column**: Should work like 1D binary search

## Related Problems
- [74. Search a 2D Matrix](https://leetcode.com/problems/search-a-2d-matrix/) - Sorted rows, binary search approach
- [240. Search a 2D Matrix II](https://leetcode.com/problems/search-a-2d-matrix-ii/) - This problem
- [378. Kth Smallest Element in a Sorted Matrix](https://leetcode.com/problems/kth-smallest-element-in-a-sorted-matrix/) - Similar matrix traversal
- [1428. Leftmost Column with at Least a One](https://leetcode.com/problems/leftmost-column-with-at-least-a-one/) - Binary search in matrix columns
```
