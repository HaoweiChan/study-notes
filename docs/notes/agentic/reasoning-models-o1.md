---
title: "Reasoning Models & o1 (System 2)"
date: "2025-12-27"
category: "agentic"
tags: ["reasoning", "chain-of-thought", "o1", "reinforcement-learning", "system-2"]
related: ["agent-reasoning-patterns-react-cot", "process-reward-models-prm"]
slug: "reasoning-models-o1"
flashcards:
  - q: "What is the key difference between 'System 1' and 'System 2' AI models?"
    a: "System 1 models (like GPT-4) produce fast, intuitive responses. System 2 models (like o1) engage in slow, deliberative reasoning (Chain of Thought) before answering."
  - q: "What is 'Test-Time Compute' scaling?"
    a: "The observation that giving a model more time to 'think' (generate internal reasoning tokens) during inference linearly improves performance on complex tasks."
  - q: "How are reasoning models typically trained?"
    a: "Using Reinforcement Learning (RL) where the model is rewarded for correct final answers, encouraging it to develop effective internal reasoning chains."
  - q: "What is the recommended prompting strategy for OpenAI's o1?"
    a: "Keep prompts simple and direct. Do not manually instruct the model to 'think step by step' as it already does this implicitly and better."
quizzes:
  - q: "Why are reasoning models like o1 slower than standard LLMs?"
    options: ["They have more parameters", "They generate hidden 'thought tokens' before producing the final output", "They access the internet for every query", "They run on CPU instead of GPU"]
    answers: [1]
    explanation: "The latency comes from the generation of thousands of invisible 'reasoning tokens' where the model plans, critiques, and refines its answer."
  - q: "Which class of problems do Reasoning Models excel at compared to standard LLMs?"
    options: ["Creative writing", "Complex logic, Math, and Coding", "Simple classification", "Translation"]
    answers: [1]
    explanation: "The deliberative process allows them to break down complex logical puzzles, math problems, and algorithmic coding tasks that require multi-step planning."
---

# Reasoning Models & o1 (System 2)

## Summary
**Reasoning Models** (like OpenAI's **o1** or DeepSeek **R1**) represent a shift from "Next Token Prediction" to **"Test-Time Compute"**. These models are trained via Reinforcement Learning to generate a hidden **Chain of Thought (CoT)** before producing a final answer, effectively mimicking human "System 2" (slow, deliberative) thinking.

## Details

### 1. System 1 vs. System 2
*   **System 1 (GPT-4, Claude 3.5):** Fast, intuitive, pattern-matching. "Read inputs -> Output tokens". Good for RAG, creative writing, simple tasks.
*   **System 2 (o1, Strawberry):** Slow, analytical, logical. "Read inputs -> *Think/Plan/Critique* -> Output tokens". Good for Math, PhD-level Science, Competitive Coding.

### 2. The New Scaling Law: Test-Time Compute
Traditionally, models improved by adding more **Training** compute/data.
Reasoning models introduce a new dimension: **Inference** compute.
*   **Observation:** Performance improves linearly with the amount of time the model spends "thinking" (generating hidden reasoning tokens).
*   **Implication:** We can trade latency for intelligence.

### 3. Training Methodology
*   **Reinforcement Learning (RL):** Unlike SFT (Supervised Fine-Tuning) which mimics human CoT, reasoning models are trained with RL.
*   **Reward Signal:** The model is given a complex problem (e.g., Math). If the final answer is correct, the entire reasoning chain is reinforced.
*   **Self-Correction:** The model learns to backtrack ("Wait, that approach is wrong, let me try another way") because this behavior leads to higher rewards.

### 4. Prompting Changes
*   **Old Way:** "Let's think step by step" (Manual CoT).
*   **New Way (o1):** Just state the problem. The model's internal CoT is optimized and often better than human-guided instructions. In fact, interfering with its reasoning loop can degrade performance.

## Examples / snippets

### Conceptual Reasoning Trace (Hidden)

When asked "Solve for x: $log_2(x) + log_2(x-2) = 3$", a System 1 model might hallucinate. A System 2 model generates hidden thoughts:

```text
(Internal Monologue)
1. I need to combine the logs using the product rule: log(a) + log(b) = log(ab).
2. So, log_2(x(x-2)) = 3.
3. Remove the log by exponentiating: x(x-2) = 2^3.
4. x^2 - 2x = 8.
5. x^2 - 2x - 8 = 0.
6. Solve quadratic: (x-4)(x+2) = 0.
7. x = 4 or x = -2.
8. Wait, check domain constraints. Log arguments must be positive.
9. x > 0 and x-2 > 0 => x > 2.
10. So x = -2 is invalid.
11. Final answer is x = 4.
```

**Output:** "x = 4"

## Learning Sources
- [OpenAI o1 System Card](https://openai.com/index/learning-to-reason-with-llms/) - Technical details on the o1 preview.
- [Noam Brown (OpenAI) on Reasoning](https://www.youtube.com/watch?v=jPluSXJpdrA) - Talks about Poker AI and search/reasoning.
- [Chain-of-Thought Prompting Elicits Reasoning (Wei et al.)](https://arxiv.org/abs/2201.11903) - The paper that started it all.
