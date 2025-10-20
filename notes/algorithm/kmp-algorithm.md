---
title: "KMP Algorithm"
date: "2025-10-07"
tags: []
related: []
slug: "kmp-algorithm"
category: "algorithm"
---

# KMP Algorithm

## Summary
KMP (Knuth-Morris-Pratt) algorithm efficiently finds all occurrences of a pattern string within a text string using an LPS (longest proper prefix which is also suffix) array to avoid redundant comparisons, achieving O(n + m) time complexity.

## Details
### 🧠 What is the KMP Algorithm?

The **KMP (Knuth-Morris-Pratt) algorithm** is a string searching algorithm that efficiently finds all occurrences of a "pattern" string within a "text" string. It preprocesses the pattern to create a "longest proper prefix which is also a suffix" (LPS) array, which helps to avoid redundant comparisons.

- **Time Complexity:** `O(n + m)`, where `n` is the length of the text and `m` is the length of the pattern.

### ⚙️ How it Works

1. **Preprocessing the Pattern**:
   - Create an LPS array for the pattern. `lps[i]` stores the length of the longest proper prefix of `pattern[0...i]` that is also a suffix of `pattern[0...i]`.

2. **Searching**:
   - Use two pointers, one for the text and one for the pattern.
   - When a mismatch occurs, use the LPS array to decide where to resume the search in the text, avoiding unnecessary backtracking.

### 🧠 When to Use KMP
- When you need to find all occurrences of a pattern in a long text.
- When performance is critical, as KMP avoids re-comparing characters.

### 📂 Common Problem Categories
- **String Matching**: Find a substring in a larger string.
- **Pattern Recognition**: Identify repeated patterns in sequences.
- **Bioinformatics**: Searching for gene sequences in DNA.

## Examples / snippets

### LPS Array Computation
```python
def compute_lps_array(pattern):
    m = len(pattern)
    lps = [0] * m
    length = 0
    i = 1
    while i < m:
        if pattern[i] == pattern[length]:
            length += 1
            lps[i] = length
            i += 1
        else:
            if length != 0:
                length = lps[length - 1]
            else:
                lps[i] = 0
                i += 1
    return lps
```

### KMP Search Implementation
```python
def kmp_search(text, pattern):
    n = len(text)
    m = len(pattern)
    lps = compute_lps_array(pattern)
    i = 0  # pointer for text
    j = 0  # pointer for pattern
    while i < n:
        if pattern[j] == text[i]:
            i += 1
            j += 1
        if j == m:
            print(f"Found pattern at index {i - j}")
            j = lps[j - 1]
        elif i < n and pattern[j] != text[i]:
            if j != 0:
                j = lps[j - 1]
            else:
                i += 1
```

### Example Usage
```python
text = "ABABDABACDABABCABAB"
pattern = "ABABCABAB"
kmp_search(text, pattern)
# Output: Found pattern at index 10
```

## Flashcards

- What is the main advantage of the KMP algorithm over naive string matching? ::: KMP avoids redundant comparisons by using the LPS array to skip unnecessary character checks
- What does LPS stand for in the KMP algorithm? ::: Longest Proper Prefix which is also Suffix
- What is the time complexity of the KMP algorithm for pattern matching? ::: O(n + m) where n is text length and m is pattern length
- What is the space complexity of the KMP algorithm? ::: O(m) for the LPS array where m is pattern length
- In KMP, what happens when a mismatch occurs at position j in the pattern? ::: We use the LPS array to determine how far back to go in the pattern (to position lps\[j-1])
- What is the key insight that makes KMP more efficient than naive matching? ::: The pattern itself contains information about how to shift when mismatches occur
- How does KMP preprocessing work for the pattern "ABABAC"? ::: Build LPS array where each position shows longest proper prefix that is also suffix
- What is the worst-case scenario for KMP algorithm efficiency? ::: When the pattern consists of many repeated characters, but KMP still maintains O(n + m) worst case
- How many comparisons does KMP make in the worst case? ::: O(n + m) total comparisons, much better than naive O(n * m)
- What preprocessing step makes KMP efficient? ::: Computing the LPS (failure function) array that tells us where to jump when mismatches occur

### LPS Array Visualization Example
For pattern "AAACAAAA", the LPS array construction works as follows:
- The algorithm builds the longest proper prefix-suffix array to enable efficient backtracking during pattern matching.
```
