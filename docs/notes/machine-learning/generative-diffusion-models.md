---
title: "Generative Diffusion Models (DDPM)"
date: "2025-12-27"
category: "machine-learning"
tags: ["diffusion", "genai", "deep-learning", "ddpm", "stable-diffusion"]
related: ["llm-fine-tuning-lora-qlora", "rag-architectures-chunking"]
slug: "generative-diffusion-models"
flashcards:
  - q: "What is the core idea of Diffusion Models (DDPM)?"
    a: "They learn to generate data by reversing a gradual noise-adding process (denoising), transforming random Gaussian noise back into a structured sample."
  - q: "What is the 'Forward Process' in Diffusion?"
    a: "A fixed Markov chain that gradually adds Gaussian noise to the data until it becomes pure noise."
  - q: "What does the neural network in a DDPM predict?"
    a: "It typically predicts the noise component ($\epsilon$) present in the noisy image $x_t$ at timestep $t$."
  - q: "How does Latent Diffusion (Stable Diffusion) differ from standard DDPM?"
    a: "It performs the diffusion process in a compressed lower-dimensional 'Latent Space' (via VAE) rather than Pixel Space, significantly reducing computational cost."
  - q: "Why are Diffusion models preferred over GANs for some tasks?"
    a: "They offer better training stability (no adversarial game), better mode coverage (diversity), and higher sample quality, though sampling is slower."
quizzes:
  - q: "In the context of Stable Diffusion, how is text conditioning (e.g., 'a photo of a cat') injected into the image generation process?"
    options: ["Concatenation with input image", "Cross-Attention layers in the U-Net", "Adding text embeddings to the noise", "Post-processing filter"]
    answers: [1]
    explanation: "Text embeddings (from CLIP) are injected via Cross-Attention layers, allowing the U-Net to attend to text tokens while Denoising."
  - q: "Which statement about the sampling speed of Diffusion Models is true?"
    options: ["They are faster than GANs because they are non-adversarial", "They are slow because they require multiple iterative steps (e.g., 50-1000) to generate one sample", "They are real-time by default", "Speed depends only on image resolution, not steps"]
    answers: [1]
    explanation: "Standard diffusion requires iteratively removing noise over many timesteps, making inference significantly slower than single-pass GANs or VAEs."
---

# Generative Diffusion Models (DDPM)

## Summary
**Diffusion Models** have surpassed GANs as the state-of-the-art for image generation (e.g., DALL-E 2, Stable Diffusion). They work by destroying training data through successive addition of Gaussian noise, and then learning to recover the data by reversing this process.

## Details

### 1. The Process

#### A. Forward Process (Diffusion) - $q(x_t | x_{t-1})$
*   Gradually adds Gaussian noise to an image $x_0$ over $T$ steps (e.g., $T=1000$).
*   At $t=T$, the image is indistinguishable from isotropic Gaussian noise $\mathcal{N}(0, I)$.
*   This process is fixed (no learning involved).

#### B. Reverse Process (Denoising) - $p_\theta(x_{t-1} | x_t)$
*   The model (usually a **U-Net**) learns to predict the noise $\epsilon$ that was added to reach $x_t$.
*   By subtracting the predicted noise, we estimate $x_{t-1}$.
*   Repeat this $T$ times starting from random noise to generate a crisp image.

### 2. Key Architectures

#### Denoising Diffusion Probabilistic Models (DDPM)
*   Operates directly in **Pixel Space**.
*   Objective: Minimize MSE between added noise $\epsilon$ and predicted noise $\epsilon_\theta(x_t, t)$.
*   $$ L = || \epsilon - \epsilon_\theta(x_t, t) ||^2 $$

#### Latent Diffusion Models (LDM / Stable Diffusion)
*   **Problem:** Pixel space is high-dimensional (e.g., 512x512x3 = 786k values). Training is slow.
*   **Solution:** Use a **VAE (Variational Autoencoder)** to compress the image into a smaller **Latent Space** (e.g., 64x64x4).
*   Perform diffusion/denoising in this latent space.
*   Decode the final latent back to pixel space.

### 3. Conditioning (Text-to-Image)
To control generation (e.g., "A astronaut riding a horse"):
1.  **Text Encoder:** (e.g., CLIP) converts text to embeddings.
2.  **Cross-Attention:** The U-Net backbone uses Cross-Attention layers to "attend" to the text embeddings at each denoising step.

### 4. Comparison
| Feature | GANs | Diffusion Models |
| :--- | :--- | :--- |
| **Training** | Unstable (Adversarial) | Stable (Likelihood-based) |
| **Mode Collapse** | Common (Outputs look same) | Rare (High diversity) |
| **Quality** | High | State-of-the-Art |
| **Inference Speed** | Fast (Single pass) | Slow (Iterative) |

## Examples / snippets

### Simplified Denoising Training Loop (Pseudocode)

```python
# Training DDPM
for batch_images in dataloader:
    # 1. Sample random timesteps t
    t = torch.randint(0, T, (batch_size,))
    
    # 2. Sample random noise
    noise = torch.randn_like(batch_images)
    
    # 3. Add noise to images (Forward process)
    # x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
    noisy_images = add_noise(batch_images, noise, t)
    
    # 4. Predict noise using U-Net
    predicted_noise = model(noisy_images, t)
    
    # 5. Calculate Loss
    loss = mse_loss(predicted_noise, noise)
    
    # 6. Backprop
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

## Learning Sources
- [High-Resolution Image Synthesis with Latent Diffusion Models (Rombach et al.)](https://arxiv.org/abs/2112.10752) - The Stable Diffusion paper.
- [Denoising Diffusion Probabilistic Models (Ho et al.)](https://arxiv.org/abs/2006.11239) - The seminal DDPM paper.
- [Lilian Weng's Blog: What are Diffusion Models?](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/) - Excellent technical deep dive.
