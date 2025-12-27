---
title: "Difference Array"
date: "2025-12-01"
category: "algorithm"
tags: []
related:
  - "2d-prefix-sum"
---

# Difference Array

## Summary
The Difference Array is a technique used to perform efficient range updates on an array or matrix. By recording changes at the boundaries of a range, it allows for $O(1)$ updates, with the final state reconstructed via a prefix sum pass ($O(N)$ or $O(N^2)$). It is essentially the inverse operation of the Prefix Sum.

## Details
The core difference between Difference Array and Prefix Sum lies in their **primary purpose** and the **direction of data flow**.

Think of them as inverse operations, similar to **Integration** (Prefix Sum) and **Differentiation** (Difference Array) in calculus.

### At a Glance: The "Write" vs. "Read" Trade-off

| Feature | **Difference Array** | **2D Prefix Sum** |
| :--- | :--- | :--- |
| **Primary Goal** | **Efficient Updates** (Write-Heavy) | **Efficient Queries** (Read-Heavy) |
| **Problem Type** | "Add $X$ to this submatrix 10,000 times." | "What is the sum of this submatrix 10,000 times?" |
| **Key Operation** | Modifying 4 corners to mark a change. | Using 4 corners (Inclusion-Exclusion) to retrieve a sum. |
| **Complexity** | Update: $O(1)$ <br> Final Build: $O(N^2)$ | Build: $O(N^2)$ <br> Query: $O(1)$ |
| **Relation** | You **apply** a prefix sum to a Difference Array to get the original array. | You **build** a Prefix Sum array *from* the original array. |

### 1. Difference Array (The "Lazy" Update)

**Use this when:** You need to modify many ranges/submatrices, and you only need the final result after all modifications are done.

* **Logic:** Instead of looping through the whole rectangle to add $+1$, you just "mark" the corners. You are essentially saving the "edges" of the change.
* **The "Magic":** The matrix looks like nonsense (just sparse numbers) until you run a **Prefix Sum** pass over it at the very end. The Prefix Sum "fills in" the rectangles defined by the corners.

#### 1D Logic
To add `val` to `arr[l...r]`:
1. `diff[l] += val`
2. `diff[r + 1] -= val` (if `r + 1` is within bounds)

#### 2D Logic
**Visual:**
> Update: `[r1, c1]` to `[r2, c2]`
>
> * `[r1][c1]` says "Start adding here".
> * `[r1][c2+1]` says "Stop adding for this row".
> * `[r2+1][c1]` says "Stop adding for this column".
> * `[r2+1][c2+1]` says "Cancel out the double negative".

### 2. 2D Prefix Sum (The Precomputed Cache)

**Use this when:** The matrix is static (doesn't change), but you need to answer millions of questions about the sum of different sub-rectangles.

* **Logic:** You pre-calculate the sum of the rectangle from `(0,0)` to every cell `(i,j)`.
* **The "Magic":** When asked for the sum of a submatrix `A`, you don't count cells. You take the total large area and subtract the sections you don't need (Top and Left), adding back the top-left corner that you subtracted twice.

**Visual:**
> Query Sum: `[r1, c1]` to `[r2, c2]`
>
> * `Sum = P[r2][c2] - P[r1-1][c2] - P[r2][c1-1] + P[r1-1][c1-1]`
> * (Note: This uses inclusion-exclusion, which is the exact inverse logic of the Difference Array corner update).

### Summary of the "Inverse" Relationship

1. **If you have `A` (Difference Matrix):**
    * Running **Prefix Sum** on `A` $\rightarrow$ gives you the **Original Matrix**.

2. **If you have `M` (Original Matrix):**
    * Running **Prefix Sum** on `M` $\rightarrow$ gives you the **Prefix Sum Matrix** (for quick queries).

## Examples / snippets

### 1D Difference Array Implementation

```python
def range_updates_1d(length, updates):
    """
    Applies a series of updates to an array of zeros.
    updates: List of (l, r, val)
    """
    # Initialize difference array with one extra space for boundary handling
    diff = [0] * (length + 1)

    for l, r, val in updates:
        diff[l] += val
        if r + 1 < length:
            diff[r + 1] -= val

    # Reconstruct the array using prefix sum
    result = [0] * length
    current_sum = 0
    for i in range(length):
        current_sum += diff[i]
        result[i] = current_sum

    return result

# Example
# Array size 5, updates: [0, 2] +2, [1, 4] +3
# Initial: [0, 0, 0, 0, 0]
# Update 1: diff[0]+=2, diff[3]-=2 -> [2, 0, 0, -2, 0, 0]
# Update 2: diff[1]+=3, diff[5]-=3 -> [2, 3, 0, -2, 0, -3]
# Prefix Sum:
# i=0: 2
# i=1: 2+3=5
# i=2: 5+0=5
# i=3: 5-2=3
# i=4: 3+0=3
# Result: [2, 5, 5, 3, 3]
print(range_updates_1d(5, [(0, 2, 2), (1, 4, 3)]))
```

### 2D Difference Array Implementation

```python
class DifferenceArray2D:
    def __init__(self, m, n):
        self.m = m
        self.n = n
        # Extra row and col to handle boundaries easily
        self.diff = [[0] * (n + 1) for _ in range(m + 1)]

    def update(self, r1, c1, r2, c2, val):
        """
        Add val to submatrix from (r1, c1) to (r2, c2) inclusive.
        """
        self.diff[r1][c1] += val
        self.diff[r1][c2 + 1] -= val
        self.diff[r2 + 1][c1] -= val
        self.diff[r2 + 1][c2 + 1] += val

    def compute_final_matrix(self):
        """
        Reconstruct the matrix by computing 2D prefix sum.
        """
        # We'll use the same logic as constructing a prefix sum matrix
        # but applied to our difference array.
        # result[i][j] = diff[i][j] + result[i-1][j] + result[i][j-1] - result[i-1][j-1]
        
        # Using a new grid to hold results
        res = [[0] * self.n for _ in range(self.m)]
        
        # In-place prefix sum on self.diff is also possible but let's be explicit
        # Note: self.diff is (m+1)x(n+1)
        
        # First, compute prefix sums for the diff array itself
        # We can do this in-place on self.diff or a copy.
        # Let's calculate the actual values for each cell (i, j)
        
        for i in range(self.m):
            for j in range(self.n):
                # Standard 2D prefix sum formula to aggregate changes
                top = self.diff[i-1][j] if i > 0 else 0
                left = self.diff[i][j-1] if j > 0 else 0
                top_left = self.diff[i-1][j-1] if i > 0 and j > 0 else 0
                
                # Update current cell with accumulated changes
                self.diff[i][j] += top + left - top_left
                res[i][j] = self.diff[i][j]
                
        return res

# Example Usage
da = DifferenceArray2D(3, 3)
da.update(0, 0, 1, 1, 1)  # Add 1 to top-left 2x2
da.update(1, 1, 2, 2, 2)  # Add 2 to bottom-right 2x2

# Visualizing Updates:
# (0,0) to (1,1) +1:
#  1  0 -1  0
#  0  0  0  0
# -1  0  1  0
#  0  0  0  0

# (1,1) to (2,2) +2:
#  ...
#  0  2  0 -2
#  0  0  0  0
#  0 -2  0  2

final = da.compute_final_matrix()
for row in final:
    print(row)
# Expected:
# [1, 1, 0]
# [1, 3, 2]
# [0, 2, 2]
```

## Learning Sources
- [Prefix Sums and Difference Array: 20 minutes of EVERYTHING you need to know](https://www.youtube.com/watch?v=DSQyjutKbfk) - Comprehensive video covering both 1D and 2D versions and their inverse relationship.
- [GeeksforGeeks: Difference Array Range Update](https://www.geeksforgeeks.org/difference-array-range-update-query-o1/) - Detailed text tutorial on the concept and implementation.
- [CP-Algorithms: Difference Array](https://cp-algorithms.com/data_structures/difference_array.html) - Advanced competitive programming resource discussing efficient range updates.

