---
title: "546. Remove Boxes"
date: "2025-10-08"
tags: ["Dynamic Programming", "Interval DP"]
related: []
slug: "546-remove-boxes"
category: "leetcode"
leetcode_url: "https://leetcode.com/problems/remove-boxes/description/?envType=problem-list-v2&envId=xxzz5vc6"
leetcode_difficulty: "Hard"
leetcode_topics: ["Dynamic Programming", "Interval DP"]
---

# 546. Remove Boxes

## Summary
Given an array of box colors, maximize points by removing consecutive boxes of the same color, where each removal of k boxes gives k² points, using interval DP with state representing subarrays and trailing box counts.

## Problem Description
You are given a list of integers `boxes` where each integer represents the color of a box. You can perform the following operation any number of times:

- Select a contiguous group of boxes of the same color and remove them.
- For each removal, you get `k²` points where `k` is the number of boxes you removed.

Return the maximum number of points you can get.

**Constraints:**
- `1 <= boxes.length <= 100`
- `1 <= boxes[i] <= 100`

## Solution Approach
This is a complex **interval dynamic programming** problem that requires careful state definition.

### Key Insight
The key insight is that when we have a subarray `boxes[i..j]`, and we want to remove boxes of color `boxes[j]`, we can potentially merge with other boxes of the same color that appear later in the array.

### DP State Definition
`dp[i][j][k]` represents the maximum points we can get from subarray `boxes[i..j]` where there are `k` boxes of color `boxes[j]` that we can merge with (these would be boxes that appear after position `j` but have the same color).

### DP Transition
For each `dp[i][j][k]`:
1. **Option 1**: Remove the last box `boxes[j]` alone (if k > 0, we get 1² points, but we can also merge with the k trailing boxes)
2. **Option 2**: Find the last position `m` (m < j) where `boxes[m] == boxes[j]`, and combine the removal of boxes[m+1..j] with the trailing boxes

### Base Cases
- `dp[i][i][k] = (k + 1)²` for any k (we can remove the single box, plus k trailing boxes of same color)

## Time & Space Complexity
- **Time Complexity:** O(n³) where n is length of boxes (n ≤ 100)
- **Space Complexity:** O(n³) for the 3D DP table

## Key Insights
- **State compression**: We need 3 dimensions because trailing boxes of the same color as the end can be merged
- **Interval merging**: We can merge intervals when we find boxes of the same color
- **Greedy merging**: Always try to merge consecutive same-color boxes for maximum points
- **Recursive structure**: The DP naturally captures the recursive nature of choosing which boxes to remove first

## Examples / snippets

### Solution Code (Interval DP)
```python
from typing import List

class Solution:
    def removeBoxes(self, boxes: List[int]) -> int:
        n = len(boxes)
        # dp[i][j][k]: max points for subarray i..j with k trailing boxes of color boxes[j]
        dp = [[[0] * n for _ in range(n)] for _ in range(n)]

        def solve(i, j, k):
            if i > j:
                return 0
            if dp[i][j][k] > 0:
                return dp[i][j][k]

            # Remove trailing boxes of same color first
            while j > i and boxes[j] == boxes[j-1]:
                j -= 1
                k += 1

            # Remove the group of boxes[j] with k+1 boxes total
            dp[i][j][k] = (k + 1) * (k + 1) + solve(i, j-1, 0)

            # Try to merge with earlier boxes of same color
            for m in range(i, j):
                if boxes[m] == boxes[j]:
                    # Remove boxes between m and j first, then combine with trailing boxes
                    dp[i][j][k] = max(dp[i][j][k],
                                    solve(i, m, k+1) + solve(m+1, j-1, 0))

            return dp[i][j][k]

        return solve(0, n-1, 0)
```

### Solution Code (Bottom-up DP)
```python
from typing import List

class Solution:
    def removeBoxes(self, boxes: List[int]) -> int:
        n = len(boxes)
        # dp[i][j][k]: max points for subarray i..j with k trailing boxes of color boxes[j]
        dp = [[[0] * (n + 1) for _ in range(n)] for _ in range(n)]

        # Fill DP table in order of increasing length
        for length in range(1, n + 1):
            for i in range(n - length + 1):
                j = i + length - 1

                # For each possible k (trailing boxes)
                for k in range(n - j):
                    # Remove trailing boxes of same color first
                    while j > i and boxes[j] == boxes[j-1]:
                        j -= 1
                        k += 1

                    # Remove the group of boxes[j] with k+1 boxes total
                    dp[i][j][k] = (k + 1) * (k + 1)

                    # Add points from removing boxes[i..j-1]
                    if i <= j - 1:
                        dp[i][j][k] += dp[i][j-1][0]

                    # Try to merge with earlier boxes of same color
                    for m in range(i, j):
                        if boxes[m] == boxes[j]:
                            # Remove boxes between m and j first, then combine with trailing boxes
                            if m + 1 <= j - 1:
                                dp[i][j][k] = max(dp[i][j][k],
                                                dp[i][m][k+1] + dp[m+1][j-1][0])
                            else:
                                dp[i][j][k] = max(dp[i][j][k], dp[i][m][k+1])

        return dp[0][n-1][0]
```

### Example Walkthrough
**Example 1:**
```
Input: boxes = [1,3,2,2,2,3,4,3,1]
Output: 23

Let's trace through the DP:
- We need to consider subarrays and how to merge same-color boxes
- The optimal strategy involves removing middle boxes to combine same colors
- Final answer is 23 points ✓
```

**Example 2:**
```
Input: boxes = [1,2,2,1,1]
Output: 9

Step by step:
- Remove two 1's at the end: 2² = 4 points, left with [1,2,2]
- Remove two 2's: 2² = 4 points, left with [1]
- Remove last 1: 1² = 1 point
- Total: 4 + 4 + 1 = 9 ✓
```

## Edge Cases & Validation
- **Single box**: `[5]` should return 1 (1²)
- **All same color**: `[2,2,2,2]` should return 16 (4²)
- **No same colors**: `[1,2,3,4]` should return 4 (four 1² removals)
- **Alternating colors**: `[1,2,1,2]` should return 4 (four 1² removals)
- **Maximum merging**: `[3,3,3]` should return 9 (3²)

## Related Problems
- [312. Burst Balloons](https://leetcode.com/problems/burst-balloons/) - Similar interval DP with different scoring
- [375. Guess Number Higher or Lower II](https://leetcode.com/problems/guess-number-higher-or-lower-ii/) - Game theory with interval DP
- [546. Remove Boxes](https://leetcode.com/problems/remove-boxes/) - This problem
- [1039. Minimum Score Triangulation of Polygon](https://leetcode.com/problems/minimum-score-triangulation-of-polygon/) - Another interval DP problem
```
