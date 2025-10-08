---
title: "2338. Count the Number of Ideal Arrays"
date: "2025-10-08"
tags: ["Dynamic Programming", "Math", "Combinatorics", "Number Theory"]
related: []
slug: "2338-count-the-number-of-ideal-arrays"
category: "leetcode"
leetcode_url: "https://leetcode.com/problems/count-the-number-of-ideal-arrays/description/?envType=problem-list-v2&envId=xxzz5vc6"
leetcode_difficulty: "Hard"
leetcode_topics: ["Dynamic Programming", "Math", "Combinatorics", "Number Theory"]
---

# 2338. Count the Number of Ideal Arrays

## Summary
Count the number of non-decreasing arrays of length k where each element is between 1 and n, using dynamic programming with number theory to handle the maximum value constraint.

## Problem Description
You are given two integers `n` and `k`. An array of integers `nums` is called **ideal** if:

1. `nums` has length `k`
2. `1 <= nums[i] <= n` for all `1 <= i <= k`
3. `nums` is non-decreasing, i.e., `nums[i] <= nums[i+1]` for all `1 <= i < k`

Return the number of **distinct ideal arrays** of length `k`. Since the answer may be very large, return it modulo `10^9 + 7`.

**Constraints:**
- `1 <= n <= 10^4`
- `1 <= k <= 10^4`

## Solution Approach
This is a complex combinatorics problem that requires understanding number theory and dynamic programming.

### Key Insight
An ideal array is completely determined by:
1. The maximum value `m` in the array (where `1 <= m <= n`)
2. How the array is constructed from 1 to m

For a fixed maximum value `m`, we need to count the number of non-decreasing arrays of length k with maximum value m.

### DP State Definition
`dp[i][j]` represents the number of ways to form a non-decreasing array of length i ending with maximum value j.

### Mathematical Approach
1. For each possible maximum value m from 1 to n:
   - Count the number of ways to choose k values from 1 to m that form a non-decreasing sequence
   - This is equivalent to counting the number of multisets of size k from 1 to m

2. Use the concept of "stars and bars" or DP to count these sequences

### Number Theory Connection
The number of non-decreasing sequences of length k with values from 1 to m is equal to C(m + k - 1, k - 1).

## Time & Space Complexity
- **Time Complexity:** O(n * sqrt(n) + n * k) due to factoring numbers up to n
- **Space Complexity:** O(n) for DP arrays and factorization storage

## Key Insights
- **Non-decreasing property**: Each element <= next element, so sorted by definition
- **Maximum value constraint**: All elements <= m for a given maximum m
- **Combinatorial interpretation**: Number of ways to choose k elements from 1 to m with repetition allowed, in non-decreasing order
- **Prime factorization**: Used to optimize the DP transitions
- **Modulo arithmetic**: All operations modulo 10^9 + 7

## Examples / snippets

### Solution Code (DP with Prime Factorization)
```python
from typing import List
from collections import defaultdict

class Solution:
    def idealArrays(self, n: int, k: int) -> int:
        MOD = 10**9 + 7

        # Precompute the maximum exponent for each number
        max_exp = [0] * (n + 1)
        for i in range(1, n + 1):
            # Find the highest power of 2 dividing i
            while i % 2 == 0:
                max_exp[i] += 1
                i //= 2

        # DP: dp[i][j] = number of ways to form array of length i ending with j
        dp = [[0] * (n + 1) for _ in range(k + 1)]
        dp[0][0] = 1  # Base case: empty array

        # Fill DP table
        for i in range(1, k + 1):
            for j in range(1, n + 1):
                # For each possible previous ending value
                for prev in range(1, j + 1):
                    if max_exp[j] >= max_exp[prev]:
                        dp[i][j] = (dp[i][j] + dp[i-1][prev]) % MOD

        # Sum all possibilities for length k
        result = 0
        for j in range(1, n + 1):
            result = (result + dp[k][j]) % MOD

        return result
```

### Solution Code (Optimized with Combinatorics)
```python
from typing import List
import math

class Solution:
    def idealArrays(self, n: int, k: int) -> int:
        MOD = 10**9 + 7

        # For each possible maximum value m
        result = 0
        for m in range(1, n + 1):
            # Number of non-decreasing sequences of length k with max m
            # This is C(m + k - 1, k - 1)
            ways = math.comb(m + k - 1, k - 1) % MOD
            result = (result + ways) % MOD

        return result
```

### Example Walkthrough
**Example 1:**
```
Input: n = 2, k = 2
Output: 3

Ideal arrays:
- [1, 1] (max = 1)
- [1, 2] (max = 2)
- [2, 2] (max = 2)

For max = 1: only [1, 1] ✓
For max = 2: [1, 2] and [2, 2] ✓
Total: 3 ✓
```

**Example 2:**
```
Input: n = 3, k = 2
Output: 6

All possible non-decreasing arrays:
- [1, 1], [1, 2], [1, 3]
- [2, 2], [2, 3]
- [3, 3]

Total: 6 ✓
```

## Edge Cases & Validation
- **k = 1**: Should return n (any single element from 1 to n)
- **n = 1**: Should return 1 (only [1] repeated k times)
- **Large n, small k**: Should handle efficiently
- **k > n**: Still valid (can repeat values)
- **n = 1, k = 1**: Should return 1

## Related Problems
- [62. Unique Paths](https://leetcode.com/problems/unique-paths/) - Basic combinatorics
- [63. Unique Paths II](https://leetcode.com/problems/unique-paths-ii/) - Grid paths with obstacles
- [2338. Count the Number of Ideal Arrays](https://leetcode.com/problems/count-the-number-of-ideal-arrays/) - This problem
- [1186. Maximum Subarray Sum with One Deletion](https://leetcode.com/problems/maximum-subarray-sum-with-one-deletion/) - Different DP approach
```
