---
title: "Monotonic Queue"
date: "2025-10-07"
tags: []
related: []
slug: "monotonic-queue"
category: "algorithm"
---

# Monotonic Queue

## Summary
A Monotonic Queue is a double-ended queue that maintains elements in increasing or decreasing order to efficiently solve sliding window min/max problems in O(n) time, commonly used for optimizing dynamic programming and sliding window algorithms.

## Details
### 🧠 What is a Monotonic Queue?

A Monotonic Queue is a double-ended queue (deque) that maintains its elements in either increasing or decreasing order, typically to solve sliding window min/max problems in O(n) time.

### ⚙️ Key Operations
- **Push:** When pushing a new element, pop elements from the back that violate monotonicity.
- **Pop:** Remove the front element if it's outside the window.
- **Front:** The maximum or minimum of the current window.

### ✅ When to Use Monotonic Queue
- You need the min/max value in a sliding window
- Optimizing dynamic programming with a sliding window
- Avoiding O(nk) brute force when k is large

### 📂 LeetCode Problems Using Monotonic Queue

| Problem | Description |
|---------|-------------|
| **239. Sliding Window Maximum** | Classic use of decreasing monotonic queue |
| **862. Shortest Subarray with Sum at Least K** | Use monotonic queue with prefix sums |
| **1438. Longest Continuous Subarray With Absolute Diff Less Than or Equal to Limit** | Use both min and max queues |
| **1696. Jump Game VI** | Monotonic queue for DP window optimization |
| **1425. Constrained Subsequence Sum** | Use monotonic queue to keep track of max DP in window |

## Examples / snippets

### Monotonic Decreasing Queue (for Maximum in Window)
```python
from collections import deque

def max_sliding_window(nums, k):
    dq = deque()
    res = []

    for i in range(len(nums)):
        # Remove elements smaller than current from the back
        while dq and nums[i] > nums[dq[-1]]:
            dq.pop()

        dq.append(i)

        # Remove front if it's out of the current window
        if dq[0] <= i - k:
            dq.popleft()

        # Append result once window is fully within range
        if i >= k - 1:
            res.append(nums[dq[0]])

    return res
```

### Monotonic Increasing Queue (for Minimum in Window)
```python
from collections import deque

def min_sliding_window(nums, k):
    dq = deque()
    res = []

    for i in range(len(nums)):
        # Remove elements larger than current from the back
        while dq and nums[i] < nums[dq[-1]]:
            dq.pop()

        dq.append(i)

        # Remove front if it's out of the current window
        if dq[0] <= i - k:
            dq.popleft()

        # Append result once window is fully within range
        if i >= k - 1:
            res.append(nums[dq[0]])

    return res
```

### Example Usage
```python
# Example for maximum sliding window
nums = [1, 3, -1, -3, 5, 3, 6, 7]
k = 3
result = max_sliding_window(nums, k)
print(result)  # Expected: [3, 3, 5, 5, 6, 7]

# Example for minimum sliding window
nums = [1, 3, -1, -3, 5, 3, 6, 7]
k = 3
result = min_sliding_window(nums, k)
print(result)  # Expected: [-1, -3, -3, -3, 3, 3]
```

### Dual Monotonic Queues for Range Queries
```python
from collections import deque

def longest_subarray_with_limit(nums, limit):
    max_dq = deque()  # Decreasing queue for maximums
    min_dq = deque()  # Increasing queue for minimums
    left = 0
    max_length = 0

    for right in range(len(nums)):
        # Maintain max_dq (decreasing)
        while max_dq and nums[right] > nums[max_dq[-1]]:
            max_dq.pop()
        max_dq.append(right)

        # Maintain min_dq (increasing)
        while min_dq and nums[right] < nums[min_dq[-1]]:
            min_dq.pop()
        min_dq.append(right)

        # Shrink window if max - min > limit
        while nums[max_dq[0]] - nums[min_dq[0]] > limit:
            left += 1
            if max_dq[0] < left:
                max_dq.popleft()
            if min_dq[0] < left:
                min_dq.popleft()

        max_length = max(max_length, right - left + 1)

    return max_length
```

## Flashcards

- What is the main purpose of a monotonic queue? ::: To efficiently find the minimum or maximum value in a sliding window in O(n) time
- What are the two main operations performed on a monotonic queue? ::: Push (add element while maintaining monotonic order) and Pop (remove element if outside window)
- What is the key property that makes monotonic queues efficient? ::: They maintain elements in increasing or decreasing order, allowing O(1) access to min/max
- When should you use a monotonic decreasing queue? ::: When you need to find the maximum value in each sliding window
- When should you use a monotonic increasing queue? ::: When you need to find the minimum value in each sliding window
- What is the time complexity for processing n elements with window size k using monotonic queue? ::: O(n) total time, O(k) space for the queue
- What happens when pushing an element to a monotonic decreasing queue? ::: Remove all elements smaller than the current element from the back before adding
- What happens when the front element goes out of the current window? ::: Remove it from the front of the queue
- What problem can be solved using dual monotonic queues (both min and max)? ::: Finding longest subarray where max - min ≤ limit
- What is the main advantage of monotonic queue over a regular sliding window approach? ::: O(1) time to get min/max per window instead of O(k) time

```
```
