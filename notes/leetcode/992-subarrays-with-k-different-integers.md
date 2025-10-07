---
title: "992. Subarrays with K Different Integers"
date: "2025-10-08"
tags: ["Hash Table", "Two Pointers", "Sliding Window"]
related: []
slug: "992-subarrays-with-k-different-integers"
category: "leetcode"
leetcode_url: "https://leetcode.com/problems/subarrays-with-k-different-integers/description/?envType=problem-list-v2&envId=xxzz5vc6"
leetcode_difficulty: "Hard"
leetcode_topics: ["Hash Table", "Two Pointers", "Sliding Window"]
---

# 992. Subarrays with K Different Integers

## Summary
Given an integer array nums and an integer k, return the number of subarrays that contain exactly k different integers using a sliding window approach with the formula: exactly K = at most K - at most (K-1).

## Problem Description
Given an integer array `nums` and an integer `k`, return the number of subarrays that contain **exactly** `k` distinct integers.

**Constraints:**
- `1 <= nums.length <= 2 * 10^4`
- `1 <= nums[i] <= nums.length`
- `1 <= k <= nums.length`

## Solution Approach
This problem can be solved using the **sliding window** technique with a clever mathematical insight:

**Key Insight:** The number of subarrays with **exactly K** distinct integers = number of subarrays with **at most K** distinct integers - number of subarrays with **at most (K-1)** distinct integers.

**Algorithm:**
1. Create a helper function that counts subarrays with **at most** `max_distinct` integers
2. Use a sliding window with a frequency map to track distinct elements
3. For each right pointer position, expand the window until distinct count exceeds `max_distinct`
4. Count valid windows as `right - left + 1` for each valid window
5. Return `at_most(K) - at_most(K-1)` for exactly K

**Implementation Details:**
- Use a dictionary to track frequency of each number in current window
- Use a counter for distinct elements in current window
- When distinct count exceeds limit, shrink window from left until it's valid again
- For each valid window ending at `right`, add `right - left + 1` to the count

## Time & Space Complexity
- **Time Complexity:** O(n) - each element is visited at most twice (once when expanding, once when contracting)
- **Space Complexity:** O(n) - for the frequency dictionary in worst case

## Key Insights
- The "exactly K = at most K - at most (K-1)" formula is the key insight that makes this problem solvable in linear time
- We need to handle the case where K = 1 separately (no K-1 case)
- The sliding window approach works because we can efficiently maintain the count of distinct elements
- This technique is generalizable to other "exactly K" problems

## Examples / snippets

### Solution Code
```python
from typing import List
from collections import defaultdict

class Solution:
    def subarraysWithKDistinct(self, nums: List[int], k: int) -> int:
        def at_most(k_val: int) -> int:
            count = 0
            left = 0
            freq = defaultdict(int)

            for right in range(len(nums)):
                freq[nums[right]] += 1

                # Shrink window if too many distinct elements
                while len(freq) > k_val:
                    freq[nums[left]] -= 1
                    if freq[nums[left]] == 0:
                        del freq[nums[left]]
                    left += 1

                # Add all valid subarrays ending at 'right'
                count += right - left + 1

            return count

        # Exactly K = At most K - At most (K-1)
        return at_most(k) - at_most(k - 1)
```

### Example Walkthrough
**Example 1:**
```
Input: nums = [1,2,1,2,3], k = 2
Output: 7

Explanation:
Subarrays with exactly 2 distinct integers:
- [1,2] (indices 0-1)
- [2,1] (indices 1-2)
- [1,2] (indices 1-3, but wait, let's list properly)
- [1,2,3] (indices 0-4, but this has 3 distinct)
- [2,1,2] (indices 1-3, has 2 distinct)
- [1,2,3] (indices 2-4, has 3 distinct)
- [2,3] (indices 3-4, has 2 distinct)

Wait, let me be more careful:
- Length 2: [1,2], [2,1], [1,2], [2,3] = 4
- Length 3: [1,2,1], [2,1,2] = 2
- Length 4: [1,2,1,2] = 1
Total: 7 ✓

Using formula: at_most(2) - at_most(1)
- at_most(2) = 10 (all subarrays with ≤2 distinct)
- at_most(1) = 3 ([1], [2], [3] length 1)
- 10 - 3 = 7 ✓
```

**Example 2:**
```
Input: nums = [1,2,1,3,4], k = 3
Output: 3

Subarrays: [1,2,1,3], [2,1,3,4], [1,3,4]
Each has exactly 3 distinct integers.
```

## Edge Cases & Validation
- **K = 1**: Should return number of subarrays with exactly 1 distinct integer
- **K = len(nums)**: Should return 1 (the entire array, if it has exactly K distinct)
- **All elements same**: `[1,1,1,1], k=1` should return 10 (all subarrays)
- **All elements different**: `[1,2,3,4], k=4` should return 1
- **Empty array**: Not possible per constraints (n >= 1)
- **Single element**: `[5], k=1` should return 1

## Related Problems
- [340. Longest Substring with At Most K Distinct Characters](https://leetcode.com/problems/longest-substring-with-at-most-k-distinct-characters/) - Similar sliding window
- [3. Longest Substring Without Repeating Characters](https://leetcode.com/problems/longest-substring-without-repeating-characters/) - K= all distinct
- [904. Fruit Into Baskets](https://leetcode.com/problems/fruit-into-baskets/) - Exactly 2 types
- [930. Binary Subarrays With Sum](https://leetcode.com/problems/binary-subarrays-with-sum/) - Similar prefix sum technique
```
