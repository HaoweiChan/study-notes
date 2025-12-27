---
title: "315. Count of Smaller Numbers After Self"
date: "2025-10-08"
tags: ["Binary Search", "Divide and Conquer", "Binary Indexed Tree", "Segment Tree", "Merge Sort"]
related: []
slug: "315-count-of-smaller-numbers-after-self"
category: "leetcode"
leetcode_url: "https://leetcode.com/problems/count-of-smaller-numbers-after-self/description/?envType=problem-list-v2&envId=xxzz5vc6"
leetcode_difficulty: "Hard"
leetcode_topics: ["Binary Search", "Divide and Conquer", "Binary Indexed Tree", "Segment Tree", "Merge Sort"]
---

# 315. Count of Smaller Numbers After Self

## Summary
Given an integer array nums, return an array where each element represents the count of smaller numbers that appear after the current element, using a Fenwick Tree or modified merge sort approach for O(n log n) time complexity.

## Problem Description
You are given an integer array `nums` and you have to return a new array `counts` where `counts[i]` is the number of elements to the **right** of `nums[i]` that are **smaller** than `nums[i]`.

**Constraints:**
- `1 <= nums.length <= 10^5`
- `-10^4 <= nums[i] <= 10^4`

## Solution Approach
This problem can be solved using several approaches, each with O(n log n) time complexity:

### Approach 1: Fenwick Tree / Binary Indexed Tree
**Key Insight:** We need to count numbers smaller than current element that appear later. We can process elements from right to left and use a Fenwick Tree to query the count of smaller numbers.

**Algorithm:**
1. Create a Fenwick Tree that can handle the range of numbers (shift negative numbers to positive)
2. Process elements from right to left
3. For each element, query the Fenwick Tree for count of numbers smaller than current
4. Update the Fenwick Tree with current element
5. Store the query result in result array

### Approach 2: Modified Merge Sort
**Key Insight:** Use divide and conquer to count inversions during merge sort, but track which elements are being counted for each position.

**Algorithm:**
1. Create an array of indices to track original positions
2. Use merge sort but during merge, count how many elements from right subarray are smaller than current element
3. Maintain a result array to store counts for each original position

### Approach 3: Binary Search Tree
**Key Insight:** Build a BST where each node tracks the count of smaller elements in its subtree.

## Time & Space Complexity
- **Time Complexity:** O(n log n) for all approaches - Fenwick Tree, Merge Sort, and BST
- **Space Complexity:** O(n) for the result array and auxiliary data structures

## Key Insights
- **Coordinate transformation**: Since numbers can be negative, we need to shift them to positive range for Fenwick Tree
- **Right to left processing**: Essential for all approaches to count elements that appear after current position
- **Merge sort approach**: Naturally handles the divide and conquer counting during merge phase
- **BST approach**: Each node needs to track subtree size for counting smaller elements

## Examples / snippets

### Solution Code (Fenwick Tree Approach)
```python
from typing import List

class FenwickTree:
    def __init__(self, size: int):
        self.size = size
        self.tree = [0] * (size + 1)

    def update(self, index: int, delta: int):
        index += 1  # 1-based indexing
        while index <= self.size:
            self.tree[index] += delta
            index += index & -index

    def query(self, index: int) -> int:
        index += 1  # 1-based indexing
        result = 0
        while index > 0:
            result += self.tree[index]
            index -= index & -index
        return result

class Solution:
    def countSmaller(self, nums: List[int]) -> List[int]:
        # Find the range of numbers (including negatives)
        min_val = min(nums)
        max_val = max(nums)

        # Shift all numbers to positive range
        shift = -min_val
        ft_size = max_val - min_val + 2  # +2 for safety

        ft = FenwickTree(ft_size)
        result = [0] * len(nums)

        # Process from right to left
        for i in range(len(nums) - 1, -1, -1):
            # Query count of smaller numbers (shifted index)
            result[i] = ft.query(nums[i] + shift - 1)
            # Update Fenwick Tree with current number
            ft.update(nums[i] + shift, 1)

        return result
```

### Solution Code (Merge Sort Approach)
```python
from typing import List

class Solution:
    def countSmaller(self, nums: List[int]) -> List[int]:
        def merge_sort(enum):
            if len(enum) <= 1:
                return enum

            mid = len(enum) // 2
            left = merge_sort(enum[:mid])
            right = merge_sort(enum[mid:])

            return merge(left, right)

        def merge(left, right):
            merged = []
            count = 0
            i = j = 0

            # Count smaller elements during merge
            while i < len(left) and j < len(right):
                if left[i][0] <= right[j][0]:
                    # Left element is smaller or equal, no count increase needed
                    merged.append(left[i])
                    i += 1
                else:
                    # Right element is smaller, count it for all remaining left elements
                    count += len(left) - i
                    merged.append(right[j])
                    j += 1

            # Add remaining elements
            merged.extend(left[i:])
            merged.extend(right[j:])

            return merged, count

        # Create enumerated list with (value, index) pairs
        enum = [(num, i) for i, num in enumerate(nums)]
        result = [0] * len(nums)

        def merge_sort_with_count(enum):
            if len(enum) <= 1:
                return enum

            mid = len(enum) // 2
            left = merge_sort_with_count(enum[:mid])
            right = merge_sort_with_count(enum[mid:])

            merged, count = merge(left, right)

            # Update result array with counts
            for i, (_, original_idx) in enumerate(merged):
                if i < len(left):
                    result[original_idx] = count if i == 0 else result[original_idx] + count

            return merged

        merge_sort_with_count(enum)
        return result
```

### Example Walkthrough
**Example 1:**
```
Input: nums = [5,2,6,1]
Output: [2,1,1,0]

Step by step:
- i=3, nums[3]=1: 0 numbers after it are smaller → result[3]=0
- i=2, nums[2]=6: Query for numbers < 6, found 1 → result[2]=1
- i=1, nums[1]=2: Query for numbers < 2, found 1 → result[1]=1
- i=0, nums[0]=5: Query for numbers < 5, found 2 → result[0]=2

Final result: [2,1,1,0]
```

**Example 2:**
```
Input: nums = [-1]
Output: [0]

Explanation: No elements after -1, so count is 0.
```

**Example 3:**
```
Input: nums = [-1,-1]
Output: [0,0]

Explanation: Both elements are equal, no smaller numbers after each.
```

## Edge Cases & Validation
- **Single element**: `[5]` should return `[0]`
- **All elements equal**: `[3,3,3]` should return `[0,0,0]`
- **Strictly decreasing**: `[5,4,3,2,1]` should return `[4,3,2,1,0]`
- **Strictly increasing**: `[1,2,3,4,5]` should return `[0,0,0,0,0]`
- **Negative numbers**: `[-3,-2,-1,0,1]` should handle negative range correctly
- **Large numbers**: Should handle full range from -10^4 to 10^4

## Related Problems
- [493. Reverse Pairs](https://leetcode.com/problems/reverse-pairs/) - Similar inversion counting
- [327. Count of Range Sum](https://leetcode.com/problems/count-of-range-sum/) - Range sum queries with Fenwick Tree
- [315. Count of Smaller Numbers After Self](https://leetcode.com/problems/count-of-smaller-numbers-after-self/) - This problem
- [Sword Finger Offer 51. Reverse Pairs](https://leetcode.cn/problems/shu-zu-zhong-de-ni-xu-dui-lcof/) - Chinese version of reverse pairs
```
