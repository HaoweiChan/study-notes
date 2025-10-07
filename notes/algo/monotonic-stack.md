---
title: "Monotonic Stack"
date: "2025-10-07"
tags: []
related: []
slug: "monotonic-stack"
category: "algo"
---

# Monotonic Stack

## Summary
A Monotonic Stack maintains elements in increasing or decreasing order to efficiently solve problems involving next greater/smaller elements in O(n) time, commonly used for histogram problems, temperature queries, and rectangle calculations.

## Details
### 🧠 What is a Monotonic Stack?

A monotonic stack is a stack that maintains its elements in a specific increasing or decreasing order. It's commonly used to solve problems involving next greater/smaller elements, often in O(n) time.

### 🔧 How to Build a Monotonic Stack

1. **Increasing Stack (for finding next smaller elements)**
   ```python
   stack = []
   for i in range(len(nums)):
       while stack and nums[i] < nums[stack[-1]]:
           stack.pop()
       stack.append(i)
   ```

2. **Decreasing Stack (for finding next greater elements)**
   ```python
   stack = []
   for i in range(len(nums)):
       while stack and nums[i] > nums[stack[-1]]:
           stack.pop()
       stack.append(i)
   ```

### ✅ When to Use Monotonic Stack
- Problems asking for Next Greater/Smaller Element
- Sliding window min/max queries
- Efficient O(n) solutions to histogram, water trap, and temperature type problems

### 📂 LeetCode Problems Using Monotonic Stack

#### 🔥 Classic Problems

| Problem | Description |
|---------|-------------|
| **496. Next Greater Element I** | Find the next greater number for elements in a subset |
| **503. Next Greater Element II** | Circular array version of next greater |
| **739. Daily Temperatures** | Wait days until a warmer temperature |
| **84. Largest Rectangle in Histogram** | Find largest rectangle in a bar chart |
| **85. Maximal Rectangle** | Use monotonic stack row by row |
| **42. Trapping Rain Water** | Water between buildings |

## Examples / snippets

### Next Greater Element Implementation
```python
def next_greater_elements(nums):
    res = [-1] * len(nums)
    stack = []

    for i in range(len(nums)):
        while stack and nums[i] > nums[stack[-1]]:
            idx = stack.pop()
            res[idx] = nums[i]
        stack.append(i)

    return res
```

### Next Smaller Element Implementation
```python
def next_smaller_elements(nums):
    res = [-1] * len(nums)
    stack = []

    for i in range(len(nums)):
        while stack and nums[i] < nums[stack[-1]]:
            idx = stack.pop()
            res[idx] = nums[i]
        stack.append(i)

    return res
```

### Daily Temperatures (LeetCode 739)
```python
def daily_temperatures(temperatures):
    res = [0] * len(temperatures)
    stack = []

    for i in range(len(temperatures)):
        while stack and temperatures[i] > temperatures[stack[-1]]:
            idx = stack.pop()
            res[idx] = i - idx
        stack.append(i)

    return res
```

### Largest Rectangle in Histogram (LeetCode 84)
```python
def largest_rectangle_area(heights):
    stack = []
    max_area = 0
    heights.append(0)  # Append 0 to handle remaining elements

    for i in range(len(heights)):
        while stack and heights[i] < heights[stack[-1]]:
            h = heights[stack.pop()]
            w = i if not stack else i - stack[-1] - 1
            max_area = max(max_area, h * w)
        stack.append(i)

    return max_area
```

### Example Usage
```python
# Next Greater Element
nums = [4, 5, 2, 10, 8]
result = next_greater_elements(nums)
print(result)  # Expected: [5, 10, 10, -1, -1]

# Daily Temperatures
temperatures = [73, 74, 75, 71, 69, 72, 76, 73]
result = daily_temperatures(temperatures)
print(result)  # Expected: [1, 1, 4, 2, 1, 1, 0, 0]

# Largest Rectangle in Histogram
heights = [2, 1, 5, 6, 2, 3]
result = largest_rectangle_area(heights)
print(result)  # Expected: 10
```

### Circular Next Greater Element (LeetCode 503)
```python
def next_greater_elements_circular(nums):
    n = len(nums)
    res = [-1] * n
    stack = []

    for i in range(2 * n):
        while stack and nums[i % n] > nums[stack[-1]]:
            idx = stack.pop()
            res[idx] = nums[i % n]
        stack.append(i % n)

    return res
```
```
