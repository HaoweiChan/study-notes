---
title: "LLM Fine-Tuning (LoRA, QLoRA)"
date: "2025-12-27"
tags: ["genai", "fine-tuning", "peft"]
related: []
slug: "llm-fine-tuning-lora-qlora"
category: "agentic"
---

# LLM Fine-Tuning (LoRA, QLoRA)

## Summary
Fine-tuning adapts a pre-trained LLM to a specific domain or style. **PEFT (Parameter-Efficient Fine-Tuning)** techniques like **LoRA** and **QLoRA** make this possible on consumer hardware by updating only a tiny fraction of parameters (Low-Rank Adapters) while keeping the base model frozen.

## Details

### 1. Full Fine-Tuning vs. PEFT
- **Full Fine-Tuning**: Updates all weights ($W$). Requires massive VRAM (e.g., 600GB+ for Llama-3 70B). High risk of "Catastrophic Forgetting."
- **PEFT**: Updates only a small set of extra parameters.
    - Pros: Low memory, faster training, modular (swap adapters for different tasks).

### 2. LoRA (Low-Rank Adaptation)
- **Hypothesis**: Weight updates $\Delta W$ have a low "intrinsic rank."
- **Mechanism**: Instead of training $\Delta W$ (size $d \times d$), train two small matrices $A$ ($d \times r$) and $B$ ($r \times d$) where $r \ll d$ (e.g., $r=8$ or $16$).
    - $W_{new} = W_{frozen} + \Delta W = W_{frozen} + B \times A$
- **Parameter Savings**: For GPT-3, LoRA reduces trainable params by 10,000x and VRAM by 3x.
- **Inference**: Merge $B \times A$ back into $W$ for zero latency overhead.

### 3. QLoRA (Quantized LoRA)
- **Innovation**: Combines LoRA with **4-bit Quantization** of the base model.
- **Mechanism**:
    1. Load the base model in 4-bit (using NormalFloat4 data type).
    2. Add LoRA adapters (in 16-bit).
    3. Backpropagate gradients through the frozen 4-bit weights to update the 16-bit adapters.
- **Impact**: Enables fine-tuning a 65B parameter model on a single 48GB GPU (e.g., A6000), which previously required ~780GB VRAM.

## Examples / snippets

### LoRA Config with HuggingFace PEFT

```python
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM

# 1. Load Base Model
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3-8b")

# 2. Define LoRA Config
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, 
    inference_mode=False, 
    r=16,            # Rank (dimension of A and B)
    lora_alpha=32,   # Scaling factor
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"] # Apply to Attention Query/Value layers
)

# 3. Inject Adapters
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
# Output: "trainable params: 4,194,304 || all params: 7,000,000,000 || trainable%: 0.06%"
```

## Learning Sources
- [LoRA Paper (Microsoft)](https://arxiv.org/abs/2106.09685) - "LoRA: Low-Rank Adaptation of Large Language Models".
- [QLoRA Paper (UW)](https://arxiv.org/abs/2305.14314) - "QLoRA: Efficient Finetuning of Quantized LLMs".
- [HuggingFace PEFT Documentation](https://huggingface.co/docs/peft/index) - Official docs for the library.
- [The Full Guide to Fine-Tuning LLMs](https://www.philschmid.de/) - Excellent blog tutorials.
