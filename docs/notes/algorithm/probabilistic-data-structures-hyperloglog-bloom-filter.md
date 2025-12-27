---
title: "Probabilistic Data Structures (HyperLogLog, Bloom Filter)"
date: "2025-12-27"
tags: ["algorithm", "big-data", "system-design"]
related: []
slug: "probabilistic-data-structures-hyperloglog-bloom-filter"
category: "algorithm"
---

# Probabilistic Data Structures (HyperLogLog, Bloom Filter)

## Summary
When dealing with massive data streams (Big Data), exact answers often require impossible amounts of memory. **Probabilistic Data Structures** provide approximate answers with bounded error rates using constant space. Key structures include **Bloom Filters** (Set Membership) and **HyperLogLog** (Cardinality Estimation).

## Details

### 1. Bloom Filter (Membership)
- **Problem**: "Have I seen this URL/User ID before?"
- **Memory**: $O(1)$ (e.g., 10MB for 100M items).
- **Mechanism**:
    - Initialize a bit array of size $m$ to 0.
    - Use $k$ different hash functions.
    - **Add**: Hash item $x$ $k$ times; set bits at indices $h_1(x)...h_k(x)$ to 1.
    - **Check**: Hash item $y$. If *all* bits at indices are 1, return "Probably Present". If any is 0, return "Definitely Not Present".
- **Guarantees**:
    - **False Negatives**: Impossible (if added, bits are set).
    - **False Positives**: Possible (collisions). Probability depends on $m$ and $k$.

### 2. HyperLogLog (Cardinality)
- **Problem**: "How many *unique* visitors visited the site today?" (COUNT DISTINCT).
- **Memory**: 12KB standard (can count up to $2^{64}$ items).
- **Mechanism**:
    - Hash element $x$ to a binary string.
    - Count the number of leading zeros. (Probability of $n$ leading zeros is $1/2^n$).
    - If max leading zeros is $k$, estimated cardinality is roughly $2^k$.
    - **Harmonic Mean**: Uses multiple registers ("buckets") and averages them to reduce variance.
- **Error**: Typically $\sim 0.81\%$ (standard implementation).

### 3. Count-Min Sketch (Frequency)
- **Problem**: "How many times did IP X appear?" (Frequency Table replacement).
- **Mechanism**: Similar to Bloom Filter but with counters instead of bits. Increments multiple counters on add; takes the *minimum* of counters on query.
- **Error**: Can overestimate (due to collisions), never underestimates.

## Examples / snippets

### Bloom Filter (Python Concept)

```python
import mmh3 # MurmurHash3
from bitarray import bitarray

class BloomFilter:
    def __init__(self, size, hash_count):
        self.size = size
        self.hash_count = hash_count
        self.bit_array = bitarray(size)
        self.bit_array.setall(0)

    def add(self, string):
        for i in range(self.hash_count):
            index = mmh3.hash(string, i) % self.size
            self.bit_array[index] = 1

    def lookup(self, string):
        for i in range(self.hash_count):
            index = mmh3.hash(string, i) % self.size
            if self.bit_array[index] == 0:
                return False # Definitely not present
        return True # Probably present
```

## Learning Sources
- [Redis Documentation: HyperLogLog](https://redis.io/docs/data-types/hyperloglogs/) - Practical usage in Redis (`PFADD`, `PFCOUNT`).
- [Papers We Love: Probabilistic Data Structures](https://github.com/papers-we-love/papers-we-love) - Reference to original papers.
- [System Design Primer: Bloom Filters](https://github.com/donnemartin/system-design-primer) - Visual explanation.
- [Visualizing HyperLogLog](https://content.research.neustar.neustar/blog/hll.html) - Interactive demo.
