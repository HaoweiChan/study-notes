---
title: "Distributed Training (DDP vs FSDP)"
date: "2025-12-27"
tags: ["mlops", "deep-learning", "scaling"]
related: []
slug: "distributed-training-ddp-vs-fsdp"
category: "machine-learning"
---

# Distributed Training (DDP vs FSDP)

## Summary
Training Large Language Models (LLMs) requires memory and compute beyond a single GPU. Techniques range from **Data Parallelism** (DDP) for smaller models to **Fully Sharded Data Parallel** (FSDP) and **Model Parallelism** for massive models, optimizing the trade-off between communication overhead and memory efficiency.

## Details

### 1. Data Parallelism (DDP)
- **Concept**: Replicate the *entire model* on every GPU.
- **Process**:
    1.  Split batch of data (e.g., Batch 64 -> 8 GPUs x 8 samples).
    2.  Forward pass on each GPU independently.
    3.  Backward pass computes gradients.
    4.  **All-Reduce**: Sync (average) gradients across all GPUs.
    5.  Update weights.
- **Limit**: The model parameters + optimizer state must fit in a *single* GPU's memory.

### 2. Model Parallelism
When the model itself is too big for one GPU.
- **Pipeline Parallelism**: Split layers (e.g., Layers 1-10 on GPU 0, 11-20 on GPU 1).
    - *Issue*: "Bubble" time where GPU 1 waits for GPU 0.
- **Tensor Parallelism**: Split individual matrices (e.g., $W \times X$ is split into columns).
    - *Issue*: High communication overhead.

### 3. FSDP (Fully Sharded Data Parallel)
- **Concept**: "Zero Redundancy". Instead of replicating the model, shard everything.
- **Sharding**:
    - **Parameters**: Each GPU holds only $1/N$ of the weights.
    - **Gradients**: Sharded.
    - **Optimizer State**: Sharded (biggest memory saver).
- **Process**:
    - During Forward pass: **All-Gather** required weights from other GPUs on demand, compute, then discard.
    - During Backward pass: Same.
- **Trade-off**: Trades increased communication (All-Gather) for massive memory savings. Allows training Trillion-parameter models.

### 4. ZeRO (Zero Redundancy Optimizer)
- The algorithm behind FSDP (DeepSpeed).
    - **Stage 1**: Shard Optimizer States (4x memory reduction).
    - **Stage 2**: Shard Gradients (8x).
    - **Stage 3**: Shard Parameters (Linear scaling with N GPUs).

## Examples / snippets

### PyTorch FSDP Wrapper

```python
import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

# 1. Setup Distributed Environment
torch.distributed.init_process_group(backend='nccl')

# 2. Define Model
model = MyLargeTransformer().cuda()

# 3. Wrap with FSDP
# "Auto Wrap": Automatically shard submodules larger than 100M params
my_auto_wrap_policy = functools.partial(
    size_based_auto_wrap_policy, min_num_params=100_000_000
)

fsdp_model = FSDP(
    model,
    auto_wrap_policy=my_auto_wrap_policy,
    mixed_precision=torch.distributed.fsdp.MixedPrecision(
        param_dtype=torch.float16, 
        reduce_dtype=torch.float16
    )
)

# 4. Standard Training Loop
optimizer = torch.optim.AdamW(fsdp_model.parameters())
output = fsdp_model(input)
loss = criterion(output, target)
loss.backward()
optimizer.step()
```

## Learning Sources
- [PyTorch FSDP Tutorial](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html) - Official guide.
- [ZeRO Paper (DeepSpeed)](https://arxiv.org/abs/1910.02054) - The foundational paper on Zero Redundancy Optimization.
- [HuggingFace Accelerate](https://huggingface.co/docs/accelerate/index) - Easy wrapper for DDP/FSDP.
- [A Gentle Introduction to Distributed Training](https://towardsdatascience.com/) - High-level overview.
