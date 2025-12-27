---
title: "Process Reward Models (PRM)"
date: "2025-12-27"
category: "agentic"
tags: ["prm", "reinforcement-learning", "reasoning", "math-ai", "rlhf"]
related: ["reasoning-models-o1", "agent-reasoning-patterns-react-cot"]
slug: "process-reward-models-prm"
flashcards:
  - q: "What is the difference between an Outcome Reward Model (ORM) and a Process Reward Model (PRM)?"
    a: "ORM rewards the model only based on the final answer. PRM provides feedback on each intermediate step of the reasoning chain."
  - q: "Why are PRMs crucial for complex multi-step reasoning (like Math)?"
    a: "Because a model can arrive at the wrong answer due to a single early mistake, or the right answer via wrong reasoning. PRMs identify exactly *where* the error occurred."
  - q: "How are PRMs typically used during inference?"
    a: "They act as verifiers in search algorithms (like Tree of Thoughts or Best-of-N), scoring candidate steps to prune bad branches and guide the generation."
  - q: "What is the 'Let's Verify Step by Step' dataset?"
    a: "A famous dataset by OpenAI containing math problems with human-labeled correctness for each individual reasoning step, used to train PRMs."
quizzes:
  - q: "Which search strategy benefits most from a Process Reward Model?"
    options: ["Greedy Decoding", "Beam Search / Tree Search", "Random Sampling", "Temperature 0 sampling"]
    answers: [1]
    explanation: "Beam Search or Tree Search can use the PRM scores to keep only the most promising partial reasoning chains, effectively 'looking ahead'."
  - q: "What is a major challenge in training PRMs?"
    options: ["Computational cost of inference", "High cost of human annotation for every single step", "Lack of GPU memory", "Model overfitting"]
    answers: [1]
    explanation: "Labeling every step of a solution as correct/incorrect is significantly more expensive and time-consuming than just labeling the final answer."
---

# Process Reward Models (PRM)

## Summary
**Process Reward Models (PRMs)** are the secret sauce behind advanced reasoning capabilities (like in o1 or DeepSeek Math). Instead of rewarding the AI only when it gets the final answer right (**Outcome Reward**), PRMs evaluate and reward each **intermediate step** of the reasoning chain. This enables robust multi-step problem solving.

## Details

### 1. Outcome vs. Process
*   **Outcome Reward Model (ORM):** "Did you solve the math problem?" (Yes/No).
    *   *Problem:* Sparse signal. Hard to assign credit. Model might hallucinate but get lucky.
*   **Process Reward Model (PRM):** "Is Step 1 correct? Is Step 2 correct?"
    *   *Benefit:* Dense signal. Can catch errors immediately.

### 2. Training PRMs
*   **Data Collection:** Humans (or strong models) annotate individual steps of a solution:
    *   $\checkmark$ Positive (Correct step)
    *   $\times$ Negative (Logical error, calculation error)
    *   $-$ Neutral (Irrelevant but harmless)
*   **Active Learning:** The model generates solutions, and humans focus on labeling the steps where the model is uncertain.

### 3. Inference Strategy: Search
PRMs transform generation into a **Search Problem**.
Instead of just `generate()`, we use:
*   **Best-of-N:** Generate $N$ solutions, use PRM to score them, pick the highest average step-score.
*   **Tree Search (ToT):**
    1.  Generate 3 candidate "Step 1"s.
    2.  Score them with PRM.
    3.  Keep the top 2.
    4.  Generate "Step 2" from those.
    5.  Repeat.
This allows the model to "backtrack" or self-correct if a step gets a low score.

### 4. Application Areas
*   **Mathematics:** Solving Olympiad-level problems.
*   **Coding:** Generating complex modules step-by-step.
*   **Agents:** Verifying plan execution steps.

## Examples / snippets

### PRM Scoring Example

**Problem:** Solve $2x + 5 = 15$.

**Trace A (Correct):**
1.  Subtract 5 from both sides: $2x = 10$. [PRM Score: 0.99]
2.  Divide by 2: $x = 5$. [PRM Score: 0.99]
    *   **Total:** High confidence.

**Trace B (Incorrect):**
1.  Subtract 5 from both sides: $2x = 20$. [PRM Score: **0.05**] -> *Search prunes this branch.*
2.  Divide by 2: $x = 10$.

## Learning Sources
- [Let's Verify Step by Step (OpenAI)](https://arxiv.org/abs/2305.20050) - The foundational paper introducing the PRM dataset.
- [Solving Math Word Problems with Process- and Outcome-Based Feedback](https://arxiv.org/abs/2211.14275) - Comparison of ORM vs PRM.
- [DeepSeek Math](https://arxiv.org/abs/2402.03300) - Open weights model demonstrating strong PRM usage.
