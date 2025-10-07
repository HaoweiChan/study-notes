---
title: "Segment Tree"
date: "2025-10-07"
tags: []
related: []
slug: "segment-tree"
category: "algo"
---

# Segment Tree

## Summary
A Segment Tree is a binary tree data structure that enables efficient range queries (sum, min, max) and point updates in O(log n) time, with O(n) build time, making it ideal for problems requiring frequent range operations on large arrays.

## Details
### 🧠 What is a Segment Tree?

A **Segment Tree** is a binary tree used for efficient range queries and updates (e.g., sum, min, max) in an array.

- **Build:** `O(n)`
- **Query:** `O(log n)`
- **Update:** `O(log n)`

### ⚙️ Use Case

- Given an array `arr`, we want to:
  - Quickly compute the sum/min/max of a subarray `[l, r]`
  - Efficiently update a single element `arr[i] = x`

### 🧠 When to Use Segment Tree

Use Segment Trees when you need:
- Efficient **range queries** (sum, min, max, GCD, etc.)
- **Point updates** or **range updates**
- Better performance than brute-force on frequent operations

**Tip:** When brute force takes `O(n)` or `O(n²)` and the constraints are large (e.g., `n > 10⁵`), consider using a **Segment Tree** to reduce the time complexity to `O(log n)` per operation.

### 📂 Common Problem Categories

#### 1. **Range Sum / Range Minimum / Range Maximum Queries**
- Efficient queries like `sum(i, j)` or `min(i, j)`
- **Example Problems:**
  - [307. Range Sum Query - Mutable](https://leetcode.com/problems/range-sum-query-mutable/)
  - [303. Range Sum Query - Immutable](https://leetcode.com/problems/range-sum-query-immutable/) *(Fenwick Tree or prefix sum also works)*

#### 2. **Range Updates and Lazy Propagation**
- Update an entire subrange efficiently
- **Example Problems:**
  - [732. My Calendar III](https://leetcode.com/problems/my-calendar-iii/) *(can be done with Segment Tree with lazy updates)*
  - [699. Falling Squares](https://leetcode.com/problems/falling-squares/)

#### 3. **Dynamic Order Statistics**
- Track how many numbers are smaller/greater in a given range
- **Example Problems:**
  - [315. Count of Smaller Numbers After Self](https://leetcode.com/problems/count-of-smaller-numbers-after-self/)
  - [327. Count of Range Sum](https://leetcode.com/problems/count-of-range-sum/)

#### 4. **Range GCD, LCM, XOR, Frequency Count**
- Segment tree nodes can store more complex data
- **Example Problems:**
  - [2411. Smallest Subarrays With Maximum Bitwise OR](https://leetcode.com/problems/smallest-subarrays-with-maximum-bitwise-or/)
  - [2286. Booking Concert Tickets in Groups](https://leetcode.com/problems/booking-concert-tickets-in-groups/)

#### 5. **2D Segment Tree / Advanced Intervals**
- Handle grid-based queries or nested intervals
- **Example Problems:**
  - [308. Range Sum Query 2D - Mutable](https://leetcode.com/problems/range-sum-query-2d-mutable/)
  - [699. Falling Squares](https://leetcode.com/problems/falling-squares/) *(can be visualized as 2D intervals)*

## Examples / snippets

### Basic Segment Tree Implementation
```python
class SegmentTree:
    def __init__(self, arr):
        self.n = len(arr)
        self.tree = [0] * (4 * self.n)
        self.build(arr, 0, 0, self.n - 1)

    def build(self, arr, node, l, r):
        if l == r:
            self.tree[node] = arr[l]
        else:
            mid = (l + r) // 2
            self.build(arr, 2 * node + 1, l, mid)
            self.build(arr, 2 * node + 2, mid + 1, r)
            self.tree[node] = self.tree[2 * node + 1] + self.tree[2 * node + 2]
```

### Range Query: Sum in Range \[ql, qr]
```python
    def query(self, node, l, r, ql, qr):
        if ql > r or qr < l:  # no overlap
            return 0
        if ql <= l and r <= qr:  # total overlap
            return self.tree[node]
        # partial overlap
        mid = (l + r) // 2
        left_sum = self.query(2 * node + 1, l, mid, ql, qr)
        right_sum = self.query(2 * node + 2, mid + 1, r, ql, qr)
        return left_sum + right_sum
```

### Update Element at Index idx to val
```python
    def update(self, node, l, r, idx, val):
        if l == r:
            self.tree[node] = val
        else:
            mid = (l + r) // 2
            if idx <= mid:
                self.update(2 * node + 1, l, mid, idx, val)
            else:
                self.update(2 * node + 2, mid + 1, r, idx, val)
            self.tree[node] = self.tree[2 * node + 1] + self.tree[2 * node + 2]
```

### Example Usage
```python
arr = [1, 3, 5, 7, 9, 11]
st = SegmentTree(arr)

# Query sum from index 1 to 3
print(st.query(0, 0, len(arr) - 1, 1, 3))  # Output: 15

# Update index 1 to value 10
st.update(0, 0, len(arr) - 1, 1, 10)

# Query again
print(st.query(0, 0, len(arr) - 1, 1, 3))  # Output: 22
```

### Segment Tree for Range Minimum Queries
```python
class SegmentTreeMin:
    def __init__(self, arr):
        self.n = len(arr)
        self.tree = [float('inf')] * (4 * self.n)
        self.build_min(arr, 0, 0, self.n - 1)

    def build_min(self, arr, node, l, r):
        if l == r:
            self.tree[node] = arr[l]
        else:
            mid = (l + r) // 2
            self.build_min(arr, 2 * node + 1, l, mid)
            self.build_min(arr, 2 * node + 2, mid + 1, r)
            self.tree[node] = min(self.tree[2 * node + 1], self.tree[2 * node + 2])

    def query_min(self, node, l, r, ql, qr):
        if ql > r or qr < l:
            return float('inf')
        if ql <= l and r <= qr:
            return self.tree[node]
        mid = (l + r) // 2
        left_min = self.query_min(2 * node + 1, l, mid, ql, qr)
        right_min = self.query_min(2 * node + 2, mid + 1, r, ql, qr)
        return min(left_min, right_min)

    def update_min(self, node, l, r, idx, val):
        if l == r:
            self.tree[node] = val
        else:
            mid = (l + r) // 2
            if idx <= mid:
                self.update_min(2 * node + 1, l, mid, idx, val)
            else:
                self.update_min(2 * node + 2, mid + 1, r, idx, val)
            self.tree[node] = min(self.tree[2 * node + 1], self.tree[2 * node + 2])

# Example usage for minimum queries
arr = [5, 3, 8, 1, 9, 2]
st_min = SegmentTreeMin(arr)
print(st_min.query_min(0, 0, len(arr) - 1, 1, 4))  # Output: 1 (minimum in range [1,4])
```

### Segment Tree with Lazy Propagation (for Range Updates)
```python
class LazySegmentTree:
    def __init__(self, arr):
        self.n = len(arr)
        self.tree = [0] * (4 * self.n)
        self.lazy = [0] * (4 * self.n)
        self.build(arr, 0, 0, self.n - 1)

    def build(self, arr, node, l, r):
        if l == r:
            self.tree[node] = arr[l]
            return
        mid = (l + r) // 2
        self.build(arr, 2 * node + 1, l, mid)
        self.build(arr, 2 * node + 2, mid + 1, r)
        self.tree[node] = self.tree[2 * node + 1] + self.tree[2 * node + 2]

    def propagate(self, node, l, r):
        if self.lazy[node] != 0:
            self.tree[node] += (r - l + 1) * self.lazy[node]
            if l != r:
                self.lazy[2 * node + 1] += self.lazy[node]
                self.lazy[2 * node + 2] += self.lazy[node]
            self.lazy[node] = 0

    def update_range(self, node, l, r, ql, qr, val):
        self.propagate(node, l, r)
        if ql > r or qr < l:
            return
        if ql <= l and r <= qr:
            self.lazy[node] += val
            self.propagate(node, l, r)
            return
        mid = (l + r) // 2
        self.update_range(2 * node + 1, l, mid, ql, qr, val)
        self.update_range(2 * node + 2, mid + 1, r, ql, qr, val)
        self.tree[node] = self.tree[2 * node + 1] + self.tree[2 * node + 2]

    def query(self, node, l, r, ql, qr):
        self.propagate(node, l, r)
        if ql > r or qr < l:
            return 0
        if ql <= l and r <= qr:
            return self.tree[node]
        mid = (l + r) // 2
        left_sum = self.query(2 * node + 1, l, mid, ql, qr)
        right_sum = self.query(2 * node + 2, mid + 1, r, ql, qr)
        return left_sum + right_sum

# Example usage with lazy propagation
arr = [1, 2, 3, 4, 5]
lst = LazySegmentTree(arr)

# Update range [1, 3] by adding 10 to each element
lst.update_range(0, 0, len(arr) - 1, 1, 3, 10)

# Query sum from index 0 to 4
print(lst.query(0, 0, len(arr) - 1, 0, 4))  # Output: 64 (1+12+13+14+5)
```
```
