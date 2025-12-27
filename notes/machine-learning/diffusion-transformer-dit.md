---
title: "Diffusion Transformer (DiT)"
date: "2025-12-27"
category: "machine-learning"
tags: ["diffusion", "transformer", "generative-ai", "sora", "deep-learning"]
related: ["generative-diffusion-models", "llm-fine-tuning-lora-qlora"]
slug: "diffusion-transformer-dit"
flashcards:
  - q: "What is a Diffusion Transformer (DiT)?"
    a: "A class of diffusion models that replaces the traditional U-Net backbone with a Transformer architecture, operating on latent patches."
  - q: "How does DiT process images?"
    a: "It 'patchifies' the input (usually a latent representation from a VAE) into a sequence of tokens, similar to a Vision Transformer (ViT)."
  - q: "What is 'adaLN' (Adaptive Layer Normalization) in DiT?"
    a: "A mechanism to inject conditioning information (timestep $t$ and class label $c$) by predicting the scale $\gamma$ and shift $\beta$ parameters of Layer Norm layers."
  - q: "Why is DiT preferred over U-Net for large-scale generation (like Sora)?"
    a: "Transformers scale more predictably with compute (Scaling Laws), allowing for better performance at higher parameter counts and resolutions."
quizzes:
  - q: "In the DiT architecture, where does the 'Diffusion' process happen?"
    options: ["In Pixel Space", "In the Latent Space of a VAE", "In the Text Embedding Space", "In the FFT Frequency Domain"]
    answers: [1]
    explanation: "DiT typically operates as a Latent Diffusion Model (LDM), processing compressed latents to reduce computational cost."
  - q: "Which famous video generation model is built upon the DiT architecture?"
    options: ["Runway Gen-2", "Pika Labs", "OpenAI Sora", "Stable Video Diffusion"]
    answers: [2]
    explanation: "OpenAI's Sora is explicitly described as a Diffusion Transformer that operates on spacetime patches of video."
---

# Diffusion Transformer (DiT)

## Summary
**Diffusion Transformers (DiT)** represent a shift in generative modeling architecture. While standard Stable Diffusion uses a **U-Net** (CNN) backbone to predict noise, DiT replaces this with a **Transformer**. This allows diffusion models to benefit from the massive scalability and "Scaling Laws" that have driven LLM success.

## Details

### 1. The Core Shift: U-Net → Transformer
*   **Traditional Diffusion (DDPM/LDM):** Uses U-Net with ResNet blocks and Downsampling/Upsampling. Good inductive bias for images, but hard to scale.
*   **DiT:** Uses standard Transformer blocks. Treat the image (or latent) as a sequence of tokens.

### 2. Architecture (Peebles & Xie, 2023)

#### A. Patchify
*   Input: Latent representation $z$ (e.g., $32 \times 32 \times 4$ channels).
*   Process: Break into patches (e.g., $2 \times 2$).
*   Result: A sequence of $N$ tokens (like words in NLP).

#### B. Transformer Blocks
*   Standard sequence of layers:
    *   Self-Attention (Global context).
    *   Pointwise MLP.
    *   Layer Norm.

#### C. Conditioning (adaLN-Zero)
*   The model needs to know the **Timestep** ($t$) and **Class/Text** ($c$).
*   Instead of concatenating, DiT uses **Adaptive Layer Normalization (adaLN)**.
*   A simplified MLP regresses the $\gamma$ (scale) and $\beta$ (shift) parameters for the Layer Norm based on $(t, c)$.
*   This effectively "modulates" the activations based on the diffusion step.

### 3. Why DiT?
*   **Scalability:** Transformers scale efficiently with more parameters and data (Gflops vs. FID follows a power law).
*   **Flexibility:** Can handle variable aspect ratios and resolutions easily (just more tokens).
*   **Video Generation:** OpenAI's **Sora** treats video as "Spacetime Patches" (3D tokens), extending DiT to temporal dimensions naturally.

## Examples / snippets

### DiT Block Pseudocode (Python-like)

```python
class DiTBlock(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = MultiHeadAttention(hidden_size)
        self.mlp = MLP(hidden_size)
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        
        # adaLN modulation: predicts gamma/beta/alpha for conditioning
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 6 * hidden_size)
        )

    def forward(self, x, c):
        # c contains timestep and label embeddings
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).chunk(6, dim=1)

        # 1. Self-Attention Block with modulation
        x_norm = modulate(self.norm1(x), shift_msa, scale_msa)
        x = x + gate_msa * self.attn(x_norm)
        
        # 2. MLP Block with modulation
        x_norm = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp * self.mlp(x_norm)
        
        return x

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
```

## Learning Sources
- [Scalable Diffusion Models with Transformers (Peebles & Xie)](https://arxiv.org/abs/2212.09748) - The original DiT paper (ICCV 2023).
- [Sora Technical Report](https://openai.com/sora) - OpenAI's application of DiT to video.
- [Fast.ai: Diffusion Transformers](https://course.fast.ai/) - Educational breakdown.
