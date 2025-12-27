---
title: "Reservoir Sampling"
date: "2025-12-27"
tags: ["algorithm", "probability", "data-engineering"]
related: []
slug: "reservoir-sampling"
category: "algorithm"
---

# Reservoir Sampling

## Summary
Reservoir Sampling is a randomized algorithm to select $k$ items from a stream of $n$ items, where $n$ is unknown or too large to fit in memory. It guarantees that every item has an equal probability $k/n$ of being selected.

## Details

### 1. The Problem
- **Input**: A stream of data $S = x_1, x_2, ...$
- **Constraint**: You can only pass through the data once ($O(N)$) and have limited memory ($O(k)$). $N$ is unknown.
- **Goal**: Maintain a "reservoir" of $k$ items such that at any step $i$, every item seen so far ($x_1...x_i$) has a $k/i$ chance of being in the reservoir.

### 2. Algorithm (Algorithm R)
1. **Initialization**: Store the first $k$ elements into the reservoir.
2. **Processing**: For each incoming element $x_i$ (where $i > k$):
    - Generate a random integer $j$ between 1 and $i$ (inclusive).
    - If $j \le k$: Replace the element at index $j$ in the reservoir with $x_i$.
    - Else: Discard $x_i$.

### 3. Proof of Uniformity
Why does item $x_i$ end up with probability $k/n$?
- **Base Case**: At step $n$, the probability $x_n$ is selected is $k/n$ (by definition of the algorithm).
- **Inductive Step**: What is the probability that an *existing* item stays in the reservoir?
    - It stays if it is *not* replaced.
    - $P(\text{replaced}) = P(x_n \text{ selected}) \times P(\text{index } j \text{ chosen}) = \frac{k}{n} \times \frac{1}{k} = \frac{1}{n}$.
    - $P(\text{survives}) = 1 - \frac{1}{n} = \frac{n-1}{n}$.
    - Combined with previous probability $\frac{k}{n-1}$: Final Prob = $\frac{k}{n-1} \times \frac{n-1}{n} = \frac{k}{n}$.

## Examples / snippets

### Python Implementation

```python
import random

def reservoir_sampling(stream, k):
    """
    Selects k items from an iterator 'stream' with uniform probability.
    """
    reservoir = []
    
    for i, item in enumerate(stream):
        # i starts at 0, so 'count' is i + 1
        count = i + 1
        
        if count <= k:
            # Fill the reservoir initially
            reservoir.append(item)
        else:
            # Probabilistically replace
            # Random integer between 0 and count-1 (inclusive)
            j = random.randint(0, count - 1)
            
            if j < k:
                reservoir[j] = item
                
    return reservoir

# Simulation
# Stream of 1000 numbers, select 10.
data = range(1000)
sample = reservoir_sampling(data, 10)
print(f"Sample: {sample}")
```

## Flashcards

- What is the main use case for Reservoir Sampling? ::: Selecting a random sample of $k$ items from a **stream of unknown size** or a dataset too large to fit in memory.
- What is the time complexity of Reservoir Sampling? ::: **O(N)** (single pass over the data).
- What is the space complexity of Reservoir Sampling? ::: **O(k)** (size of the sample).
- When processing the $i$-th element (where $i > k$), with what probability should we keep it? ::: Probability **$k/i$**.
- If we decide to keep the $i$-th element, which element do we replace? ::: A **randomly selected** element from the existing reservoir (index $0$ to $k-1$).

## Quizzes

### Probability Check
Q: You are sampling $k=1$ item from a stream. You see the 10th item ($x_{10}$). What is the probability that you swap it into the reservoir?
Options:
- A) 1/10
- B) 1/1
- C) 1/2
- D) 9/10
Answers: A
Explanation: For $k=1$, at step $i$, we keep the new item with probability $1/i$. So for $i=10$, prob is $1/10$.

### Distributed Sampling
Q: How would you perform reservoir sampling on a distributed system (MapReduce/Spark)?
Options:
- A) It's impossible.
- B) Assign a random number $r \in \[0, 1\]$ to every item, sort by $r$, and take the top $k$.
- C) Collect all data to one machine and run the standard algorithm.
- D) Run reservoir sampling on each partition, then just concatenate the results.
Answers: B
Explanation: This is called "Weighted Reservoir Sampling" or simply sorting by random key. Assigning a random float and taking the global top $k$ is mathematically equivalent to uniform sampling and works trivially in distributed sorts. Merging local reservoirs (D) is tricky to re-weight correctly.

## Learning Sources
- [Wikipedia: Reservoir Sampling](https://en.wikipedia.org/wiki/Reservoir_sampling) - Standard definition and Algorithm R.
- [GeeksforGeeks: Reservoir Sampling](https://www.geeksforgeeks.org/reservoir-sampling/) - Tutorial with code in C++/Java/Python.
- [Google Research: Weighted Reservoir Sampling](https://arxiv.org/abs/1010.4287) - For when items have different weights.
