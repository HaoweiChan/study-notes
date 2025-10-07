---
title: "1488. Avoid Flood in The City"
date: "2025-10-07"
tags: ["Hash Table", "Binary Search", "Greedy", "Heap"]
related: []
slug: "1488-avoid-flood-in-the-city"
category: "leetcode"
leetcode_url: "https://leetcode.com/problems/avoid-flood-in-the-city/description/?envType=daily-question&envId=2025-10-07"
leetcode_difficulty: "Medium"
leetcode_topics: ["Hash Table", "Binary Search", "Greedy", "Priority Queue"]
---

# 1488. Avoid Flood in The City

## Summary
Given an array representing rain amounts on different days, return an array of lake assignments for dry days (when it doesn't rain) such that no lake floods (receives rain twice before being dried).

## Problem Description
Your country has an infinite number of lakes. Initially, all lakes are empty, but when it rains over a lake, the lake becomes full. You cannot rain over a full lake. You are given an integer array `rains` where:
- `rains[i] > 0` means that the `rains[i]`-th lake gets rained on the `i`-th day
- `rains[i] == 0` means it didn't rain that day, so you can choose to dry any single full lake

Return an array `ans` of the same length as `rains` where:
- `ans[i] == rains[i]` if `rains[i] > 0`
- `ans[i]` is the lake you choose to dry if `rains[i] == 0` (or -1 if no lake is full)

If there are multiple valid answers, return any of them. If it's impossible to avoid flooding, return an empty array.

## Solution Approach
1. **Track lake states**: Use a set to track which lakes are currently full
2. **Schedule drying**: Use a priority queue (min-heap) to track when each lake became full, so we can dry the earliest full lake first
3. **Process chronologically**: Iterate through each day, and for dry days, choose to dry the lake that has been full the longest

**Algorithm**:
- Use a dictionary to track the last day each lake rained
- Use a min-heap to store (day, lake) pairs for lakes that are currently full
- For each day:
  - If it rains on lake `l`:
    - If lake `l` is already full, it's impossible (return [])
    - Mark lake `l` as full and add (current_day, l) to the heap
    - Update rains[i] = l
  - If it's a dry day:
    - If there are full lakes, dry the one that became full earliest
    - Otherwise, assign -1

## Time & Space Complexity
- **Time Complexity:** O(n log n) where n is the length of rains array - due to heap operations
- **Space Complexity:** O(n) for the heap and dictionary storage

## Key Insights
- We must dry lakes in chronological order of when they became full
- Using a min-heap ensures we always dry the lake that has been full the longest
- If a lake receives rain while already full, it's impossible to avoid flooding
- For dry days with no full lakes, we can choose -1 (do nothing)

## Examples / snippets

### Solution Code
```python
import heapq
from typing import List
from collections import defaultdict

class Solution:
    def avoidFlood(self, rains: List[int]) -> List[int]:
        n = len(rains)
        ans = [-1] * n

        # Track when each lake became full (for priority queue)
        full_lakes = []  # min-heap: (day, lake)
        # Track last rain day for each lake
        lake_to_day = {}

        for i, lake in enumerate(rains):
            if lake > 0:
                ans[i] = lake
                if lake in lake_to_day:
                    # Lake is already full, impossible
                    return []
                lake_to_day[lake] = i
                # Add to heap for potential drying
                heapq.heappush(full_lakes, (i, lake))
            else:
                # Dry day - choose a lake to dry
                if full_lakes:
                    # Dry the lake that became full earliest
                    day, lake_to_dry = heapq.heappop(full_lakes)
                    ans[i] = lake_to_dry
                    # Remove from our tracking
                    if lake_to_dry in lake_to_day:
                        del lake_to_day[lake_to_dry]
                # If no full lakes, ans[i] remains -1

        return ans
```

### Example Walkthrough
**Example 1:**
```
Input: rains = [1,2,3,4]
Output: [-1,-1,-1,-1]

Explanation: All days are rainy, no drying needed.
```

**Example 2:**
```
Input: rains = [1,2,0,0,2,1]
Output: [-1,-1,2,1,-1,-1]

Explanation:
- Day 0: rains[0]=1, ans[0]=1
- Day 1: rains[1]=2, ans[1]=2
- Day 2: rains[2]=0, no full lakes, ans[2]=-1
- Day 3: rains[3]=0, no full lakes, ans[3]=-1
- Day 4: rains[4]=2, but lake 2 is not full, ans[4]=2
- Day 5: rains[5]=1, but lake 1 is not full, ans[5]=1
```

**Example 3:**
```
Input: rains = [1,2,0,1,2]
Output: []

Explanation: Lake 1 rains on day 0 and day 3 while still full, impossible.
```

## Edge Cases & Validation
- **Empty array**: `[]` should return `[]`
- **All rainy days**: `[1,2,3]` should return `[1,2,3]` (no drying needed)
- **All dry days**: `[0,0,0]` should return `[-1,-1,-1]`
- **Single lake multiple rains**: `[1,0,1]` should return `[1,-1,-1]` (dry lake 1 on day 1)
- **Impossible case**: `[1,1,0]` should return `[]` (lake 1 rains twice)

## Related Problems
- [42. Trapping Rain Water](https://leetcode.com/problems/trapping-rain-water/) - Different rain/water problem
- [407. Trapping Rain Water II](https://leetcode.com/problems/trapping-rain-water-ii/) - 2D version
- [11. Container With Most Water](https://leetcode.com/problems/container-with-most-water/) - Two pointer technique
```
