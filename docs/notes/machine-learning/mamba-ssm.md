---
title: "Mamba & State Space Models (SSMs)"
date: "2025-12-27"
category: "machine-learning"
tags: ["mamba", "ssm", "deep-learning", "transformers", "sequence-modeling"]
related: ["generative-diffusion-models", "reasoning-models-o1"]
slug: "mamba-ssm"
flashcards:
  - q: "What is the primary computational advantage of Mamba (SSMs) over Transformers?"
    a: "Mamba has linear $O(N)$ scaling with sequence length, whereas Transformers have quadratic $O(N^2)$ scaling due to the Attention mechanism."
  - q: "What is a State Space Model (SSM) in the context of Deep Learning?"
    a: "A sequence model inspired by control theory that maps inputs to outputs through a latent state $h_t$, combining the parallel training of CNNs with the fast inference of RNNs."
  - q: "What is the key innovation of the 'Mamba' architecture?"
    a: "The **Selection Mechanism**: making the SSM parameters (A, B, C) input-dependent, allowing the model to selectively remember or ignore information based on the current token."
  - q: "Why is Mamba's inference memory usage constant?"
    a: "Unlike Transformers which need a growing KV Cache, Mamba maintains a fixed-size state, making it extremely efficient for long-sequence generation."
quizzes:
  - q: "How does Mamba achieve parallel training (like Transformers) despite being recurrent?"
    options: ["By using a global attention mask", "By using the Parallel Scan (associative scan) algorithm", "It cannot be trained in parallel", "By processing small chunks independently"]
    answers: [1]
    explanation: "Because the underlying SSM is linear, the recurrence can be computed efficiently using a Parallel Scan algorithm on GPUs."
  - q: "Which limitation of prior SSMs (like S4) did Mamba solve?"
    options: ["Slow training speed", "Inability to perform content-based reasoning (In-Context Learning)", "High memory usage", "Vanishing gradients"]
    answers: [1]
    explanation: "Static SSMs (S4) were time-invariant. Mamba's input-dependent dynamics allow it to perform 'Selection', crucial for copying, induction heads, and in-context learning."
---

# Mamba & State Space Models (SSMs)

## Summary
**Mamba** is a new architecture that challenges the Transformer's dominance. By building on **State Space Models (SSMs)** and introducing a **Selection Mechanism**, Mamba achieves Transformer-quality performance with **Linear $O(N)$ scaling**, enabling massive context windows and faster inference.

## Details

### 1. The Bottleneck of Transformers
*   **Quadratic Attention:** Processing a sequence of length $L$ requires $L^2$ computations. 100k tokens is expensive.
*   **KV Cache:** Inference memory grows linearly with sequence length.

### 2. What are SSMs?
Originating from Control Theory:
$$ h'(t) = \mathbf{A}h(t) + \mathbf{B}x(t) $$
$$ y(t) = \mathbf{C}h(t) $$
*   **Continuous to Discrete:** We discretize this (Zero-Order Hold) to run on digital computers.
*   **Dual View:**
    *   **Recurrent View:** Efficient inference ($O(1)$ per step), like an RNN.
    *   **Convolutional View:** Parallel training (Filter over sequence), like a CNN.

### 3. The Mamba Innovation: Selection
Prior SSMs (S4) had fixed matrices $\mathbf{A}, \mathbf{B}, \mathbf{C}$. They were fast but couldn't "focus" on specific tokens (like Attention does).
*   **Selective SSMs:** Mamba makes parameters functions of the input: $\mathbf{B}(x_t), \mathbf{C}(x_t), \Delta(x_t)$.
*   **Effect:** The model can decide at every step: "Is this token important? Keep it in state. Is it noise? Forget it."
*   **Hardware Aware:** To make this fast, Mamba uses a **Hardware-Aware Parallel Scan** implemented in CUDA (Triton) to avoid materializing large matrices in HBM.

### 4. Jamba (Hybrid)
Recent models (AI21 Jamba) combine Mamba layers with Attention layers.
*   **Mamba Layers:** Handle the bulk of the sequence processing (efficient).
*   **Attention Layers:** Every few blocks, ensure high-fidelity recall (search).

## Examples / snippets

### Linear Time Complexity Comparison

| Sequence Length | Transformer Attention | Mamba / SSM |
| :--- | :--- | :--- |
| **1k tokens** | Fast | Fast |
| **32k tokens** | Slow ($32^2 = 1024$) | Fast ($32 \times 1$) |
| **1M tokens** | Impossible (OOM) | Feasible |

### Mamba Block Structure

1.  **Input Projection:** Expand dimension ($d \to 2d$).
2.  **Conv1d:** Short local convolution (like N-gram).
3.  **SSM (Selective):** The core linear recurrence.
4.  **Gating:** Multiplicative branch (Swish gate).
5.  **Output Projection:** $2d \to d$.

## Learning Sources
- [Mamba: Linear-Time Sequence Modeling with Selective State Spaces (Gu & Dao)](https://arxiv.org/abs/2312.00752) - The Mamba paper.
- [The Annotated S4](https://srush.github.io/annotated-s4/) - Sasha Rush's guide to the predecessor of Mamba.
- [Tri Dao's Blog](https://tridao.me/) - Insights on efficient hardware kernels.
