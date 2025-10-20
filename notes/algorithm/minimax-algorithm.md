---
title: "Minimax Algorithm"
date: "2025-10-07"
tags: []
related: []
slug: "minimax-algorithm"
category: "algorithm"
---

# Minimax Algorithm

## Summary
Minimax is a decision-making algorithm for two-player turn-based games that explores all possible future moves, assuming both players play optimally, with the maximizing player trying to maximize the score and the minimizing player trying to minimize it.

## Details
### 🧠 What is Minimax?

Minimax is a decision-making algorithm used in two-player, turn-based games. It assumes both players play optimally:
- The maximizing player tries to maximize the score.
- The minimizing player tries to minimize the score.

It works by exploring all possible future moves and backpropagating values from terminal states.

### 🔄 How It Works

The algorithm recursively explores:
1. All possible moves
2. At each level, switches perspective between max and min player
3. Returns the optimal value up the call stack

### 🧪 Optimizations
- **Alpha-Beta Pruning:** Cuts off branches that don't influence the final decision.
- **Memoization / DP:** Avoids recomputation of the same state.
- **Transposition Tables:** Cache evaluated board positions.

### ✅ When to Use Minimax

Use Minimax when the problem:
- Involves two players taking turns
- Requires determining if the first player can guarantee a win
- Has a small state space or constraints (e.g., n <= 20)
- Benefits from a game tree analysis (with recursion + memoization)

### 📂 LeetCode Problems Using Minimax

- **294. Flip Game II:** Determine if the starting player can guarantee a win
- **464. Can I Win:** Use memoized minimax to check if first player can win
- **877. Stone Game:** Two players taking optimal moves
- **913. Cat and Mouse:** Graph game with alternating turns
- **486. Predict the Winner:** Classic turn-based minimax
- **1406. Stone Game III:** Advanced version with more choices and scoring

## Examples / snippets

### Basic Minimax Implementation
```python
def minimax(state, depth, is_maximizing):
    if game_over(state) or depth == 0:
        return evaluate(state)

    if is_maximizing:
        max_val = float('-inf')
        for move in get_moves(state):
            val = minimax(apply_move(state, move), depth - 1, False)
            max_val = max(max_val, val)
        return max_val
    else:
        min_val = float('inf')
        for move in get_moves(state):
            val = minimax(apply_move(state, move), depth - 1, True)
            min_val = min(min_val, val)
        return min_val
```

### Minimax with Memoization
```python
from functools import lru_cache

@lru_cache(maxsize=None)
def minimax_memo(state, depth, is_maximizing):
    if game_over(state) or depth == 0:
        return evaluate(state)

    if is_maximizing:
        max_val = float('-inf')
        for move in get_moves(state):
            new_state = apply_move(state, move)
            val = minimax_memo(new_state, depth - 1, False)
            max_val = max(max_val, val)
        return max_val
    else:
        min_val = float('inf')
        for move in get_moves(state):
            new_state = apply_move(state, move)
            val = minimax_memo(new_state, depth - 1, True)
            min_val = min(min_val, val)
        return min_val
```

### Alpha-Beta Pruning Implementation
```python
def minimax_alpha_beta(state, depth, alpha, beta, is_maximizing):
    if game_over(state) or depth == 0:
        return evaluate(state)

    if is_maximizing:
        max_val = float('-inf')
        for move in get_moves(state):
            new_state = apply_move(state, move)
            val = minimax_alpha_beta(new_state, depth - 1, alpha, beta, False)
            max_val = max(max_val, val)
            alpha = max(alpha, val)
            if beta <= alpha:
                break  # Beta cutoff
        return max_val
    else:
        min_val = float('inf')
        for move in get_moves(state):
            new_state = apply_move(state, move)
            val = minimax_alpha_beta(new_state, depth - 1, alpha, beta, True)
            min_val = min(min_val, val)
            beta = min(beta, val)
            if beta <= alpha:
                break  # Alpha cutoff
        return min_val
```

### Example Usage for Game State Evaluation
```python
# Example for a simple game like Tic-Tac-Toe or Connect Four
def can_first_player_win(initial_state):
    # Returns True if first player can force a win
    result = minimax_alpha_beta(initial_state, max_depth, float('-inf'), float('inf'), True)
    return result > 0  # Assuming positive values favor first player

# Helper functions (to be implemented based on specific game)
def game_over(state):
    # Return True if game is finished
    pass

def evaluate(state):
    # Return a score for the state (positive if first player advantage)
    pass

def get_moves(state):
    # Return list of possible moves from current state
    pass

def apply_move(state, move):
    # Return new state after applying move
    pass
```

## Flashcards

- What type of algorithm is minimax and what games is it typically used for? ::: Minimax is a decision-making algorithm for two-player turn-based games where both players play optimally
- What are the two main player types in minimax algorithm? ::: Maximizing player (tries to maximize score) and minimizing player (tries to minimize score)
- What is the key assumption that minimax makes about player behavior? ::: Both players play optimally, making the best possible moves at each turn
- What technique is commonly used to optimize minimax algorithm? ::: Alpha-beta pruning to eliminate branches that won't affect the final decision
- What is the time complexity of minimax without optimization? ::: O(b^d) where b is branching factor and d is depth, exponential in the worst case
- What data structure or technique can help avoid recomputing game states in minimax? ::: Memoization or transposition tables to cache evaluated positions
- What is the main purpose of the minimax algorithm in game theory? ::: To determine the best move for a player assuming optimal play from both sides
- How does minimax determine which move to choose at each step? ::: It evaluates all possible future moves recursively and chooses the move that leads to the best outcome
- What happens at terminal nodes (game over states) in minimax? ::: The algorithm returns the evaluated score of that position
- What is a key limitation of the basic minimax algorithm? ::: It becomes computationally expensive for games with high branching factors or deep search depths

```
```
