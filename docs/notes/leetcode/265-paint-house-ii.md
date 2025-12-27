---
title: "265. Paint House II"
date: "2025-10-08"
tags: ["Dynamic Programming", "Array"]
related: []
slug: "265-paint-house-ii"
category: "leetcode"
leetcode_url: "https://leetcode.com/problems/paint-house-ii/description/?envType=problem-list-v2&envId=xxzz5vc6"
leetcode_difficulty: "Hard"
leetcode_topics: ["Dynamic Programming", "Array"]
---

# 265. Paint House II

## Summary
Given n houses and k colors, find the minimum cost to paint all houses such that no two adjacent houses have the same color, using dynamic programming with O(n*k) time and O(1) space optimization.

## Problem Description
There are a row of `n` houses, each house can be painted with one of `k` colors. The cost of painting each house with a certain color is different. You have to paint all the houses such that no two adjacent houses have the same color.

The cost of painting each house with a certain color is represented by a `n x k` cost matrix. For example, `costs[0][0]` is the cost of painting house `0` with color `0`; `costs[1][2]` is the cost of painting house `1` with color `2`, and so on.

Return the minimum cost to paint all houses.

**Constraints:**
- `1 <= n <= 100`
- `1 <= k <= 20`
- `1 <= costs[i][j] <= 20`

## Solution Approach
This is a classic dynamic programming problem that can be solved with two approaches:

### Approach 1: Standard DP (O(n*k) time, O(n*k) space)
**Key Insight:** Use DP where `dp[i][j]` represents the minimum cost to paint first i houses, with house i-1 painted color j.

**Algorithm:**
1. `dp[i][j] = costs[i][j] + min(dp[i-1][k] for all k != j)`
2. For each house i and color j, find the minimum cost from previous row excluding the same color j
3. Final answer is `min(dp[n-1])`

### Approach 2: Optimized DP (O(n*k) time, O(1) space)
**Key Insight:** Instead of storing entire DP table, track the minimum and second minimum costs from previous row.

**Algorithm:**
1. For each house, calculate new minimum and second minimum costs
2. For each color j, the cost is `costs[i][j] + (min1 if j != prev_min_color else min2)`
3. Update min1, min2, and prev_min_color for next iteration

## Time & Space Complexity
- **Time Complexity:** O(n*k) for both approaches
- **Space Complexity:** O(n*k) for standard DP, O(1) for optimized DP

## Key Insights
- **Color constraint**: Adjacent houses cannot have same color
- **Optimization trick**: Track min1 and min2 from previous row to avoid O(k) search for each color
- **Edge case handling**: When k=1, houses must use different colors but only one available (impossible if n>1)
- **Small constraints**: n<=100, k<=20 makes O(n*k^2) acceptable but we can optimize to O(n*k)

## Examples / snippets

### Solution Code (Optimized DP)
```python
from typing import List

class Solution:
    def minCostII(self, costs: List[List[int]]) -> int:
        if not costs or not costs[0]:
            return 0

        n, k = len(costs), len(costs[0])

        # Initialize with first house costs
        prev_min1 = prev_min2 = float('inf')
        prev_min_color = -1

        for j in range(k):
            if costs[0][j] < prev_min1:
                prev_min2 = prev_min1
                prev_min1 = costs[0][j]
                prev_min_color = j
            elif costs[0][j] < prev_min2:
                prev_min2 = costs[0][j]

        # Process remaining houses
        for i in range(1, n):
            curr_min1 = curr_min2 = float('inf')
            curr_min_color = -1

            for j in range(k):
                # Choose the minimum cost from previous row
                cost = costs[i][j] + (prev_min2 if j == prev_min_color else prev_min1)

                if cost < curr_min1:
                    curr_min2 = curr_min1
                    curr_min1 = cost
                    curr_min_color = j
                elif cost < curr_min2:
                    curr_min2 = cost

            prev_min1, prev_min2, prev_min_color = curr_min1, curr_min2, curr_min_color

        return prev_min1
```

### Solution Code (Standard DP)
```python
from typing import List

class Solution:
    def minCostII(self, costs: List[List[int]]) -> int:
        if not costs or not costs[0]:
            return 0

        n, k = len(costs), len(costs[0])
        dp = [[float('inf')] * k for _ in range(n)]

        # First house - can use any color
        for j in range(k):
            dp[0][j] = costs[0][j]

        # Fill DP table
        for i in range(1, n):
            for j in range(k):
                # Find minimum cost from previous row, excluding same color
                min_prev = float('inf')
                for prev_j in range(k):
                    if prev_j != j:
                        min_prev = min(min_prev, dp[i-1][prev_j])

                dp[i][j] = costs[i][j] + min_prev

        # Return minimum cost for last house
        return min(dp[n-1])
```

### Example Walkthrough
**Example 1:**
```
Input: costs = [[1,5,3],[2,9,4]]
Output: 5

Explanation:
House 0: Paint with color 0 (cost 1), color 1 (cost 5), or color 2 (cost 3)
House 1: Paint with color 0 (cost 2), color 1 (cost 9), or color 2 (cost 4)

DP Table:
House 0: [1, 5, 3]
House 1: [2+min(5,3)=5, 9+min(1,3)=4, 4+min(1,5)=5] = [5, 4, 5]

Minimum cost: 4 (house 0 color 2, house 1 color 1)

Optimized approach:
- After house 0: min1=1 (color 0), min2=3 (color 2)
- For house 1, color 0: 2 + 3 = 5 (since color 0 != prev min color 0, use min2=3)
- For house 1, color 1: 9 + 1 = 10 (use min1=1)
- For house 1, color 2: 4 + 1 = 5 (since color 2 == prev min color 2, use min2=3? Wait no)

Wait, let me recalculate properly:
- After house 0: min1=1 (color 0), min2=3 (color 2), prev_min_color=0
- For house 1, color 0: 2 + min2 = 2 + 3 = 5 (j=0 == prev_min_color=0, so use min2)
- For house 1, color 1: 9 + min1 = 9 + 1 = 10 (j=1 != prev_min_color=0, so use min1)
- For house 1, color 2: 4 + min1 = 4 + 1 = 5 (j=2 != prev_min_color=0, so use min1)

So we get [5, 10, 5], minimum is 5 ✓
```

**Example 2:**
```
Input: costs = [[1,3],[2,4]]
Output: 5

Explanation:
- Paint house 0 with color 1 (cost 3)
- Paint house 1 with color 0 (cost 2)
- Total cost: 3 + 2 = 5

Alternative: house 0 color 0 (1) + house 1 color 1 (4) = 5, same result.
```

## Edge Cases & Validation
- **Single house**: Return minimum cost among all colors for that house
- **Two houses**: Standard DP works fine
- **k = 1**: If n > 1, impossible (return appropriate value based on constraints)
- **All costs same**: Should work correctly with DP
- **Minimum cost is 0**: Edge case where some costs are 0
- **Large n, small k**: Should handle n=100, k=20 efficiently

## Related Problems
- [256. Paint House](https://leetcode.com/problems/paint-house/) - Similar but with only 3 colors (can be O(1) space)
- [1473. Paint House III](https://leetcode.com/problems/paint-house-iii/) - More complex version with street constraints
- [265. Paint House II](https://leetcode.com/problems/paint-house-ii/) - This problem
- [Dynamic Programming Patterns](https://leetcode.com/tag/dynamic-programming/) - General DP practice
```
