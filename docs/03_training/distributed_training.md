# Distributed Training

## Why This Matters for LLMs

No frontier LLM fits on a single GPU in any realistic training configuration that also uses reasonable batch sizes and sequence lengths. **Weights**, **optimizer states**, **gradients**, and **activations** each consume memory, and long contexts multiply activation footprint. If you want to train or fine-tune models beyond a few billion parameters without heroic micro-batching, you must understand **data parallelism**, **model parallelism**, **pipeline parallelism**, **tensor parallelism**, **FSDP**, and **DeepSpeed ZeRO**. Interviewers at infrastructure-heavy teams test whether you can reason about **all-reduce**, **sharding**, **pipeline bubbles**, and **mixed precision** without hand-waving.

Distributed training is also where **engineering economics** meets **algorithmic correctness**. Data parallelism duplicates the model but shards batches; tensor parallelism splits individual operators across devices; pipeline parallelism sequences micro-batches through stages; **FSDP** and **ZeRO** shard optimizer states and parameters to squeeze larger models into fixed VRAM. Choosing the wrong strategy wastes **network bandwidth**, creates **pipeline bubbles**, or serializes steps that should **overlap** with communication. Knowing the vocabulary lets you participate in decisions about **topology**, **NCCL settings**, and **checkpoint frequency**.

Finally, **numerical stability** in large jobs depends on **mixed precision** training: **FP16** or **BF16** forward and backward passes with **FP32 master weights**, **loss scaling** where needed, and careful handling of **gradient norms**. A candidate who can explain **why BF16 often needs less loss scaling than FP16** and **how ZeRO-3 interacts with parameter gathering** demonstrates readiness for on-call debugging when a multi-thousand-GPU run diverges or hangs on a collective.

---

## Core Concepts

### Why Distributed Training?

A dense transformer stores parameters and training state in device memory. **Adam** maintains **first and second moments** per parameter. For a parameter count \(N\), **FP16** weights use **\(2N\)** bytes. Optimizer states are often quoted at **roughly 12 bytes per parameter** in mixed FP16/FP32 training schemes when counting weights, gradients, and FP32 moments together in back-of-envelope blog arithmetic. **Activations** scale with **batch size**, **sequence length**, **hidden size**, and **layer count**. A **first-order** sanity check for aggregate training memory is:

\[
M_{\text{total}} \approx M_{\text{weights}} + M_{\text{optimizer}} + M_{\text{gradients}} + M_{\text{activations}}.
\]

!!! math-intuition "In Plain English"
    This sum is not exact to the last gigabyte: it ignores **fragmentation**, **communication buffers**, and **activation checkpointing**. It **does** explain why a **7B** parameter model does not fit comfortably on one **80GB** GPU for naive full training at useful batch sizes: **optimizer state** and **activations** push you past device limits unless you **shard**, **recompute**, or **offload**.

!!! example "Worked Example: Memory Budget for 70B Model"
    **Weights in FP16:** each parameter uses **2 bytes**.

    \[
    M_{\text{weights}} = 70 \times 10^9 \times 2\ \text{B} = 140 \times 10^9\ \text{B} \approx 140\ \text{GB}.
    \]

    **Adam states** using the common **12 bytes per parameter** heuristic:

    \[
    M_{\text{opt}} \approx 70 \times 10^9 \times 12\ \text{B} = 840 \times 10^9\ \text{B} \approx 840\ \text{GB}.
    \]

    **Gradients** at FP16 add roughly **140 GB** in the same ballpark as weights. **Activations** depend on **batch**, **sequence length**, and **checkpointing**; they can add **hundreds of gigabytes** without recomputation. Summing conservative terms already exceeds **1 TB** of aggregate memory across tensors for a **single** full replica. Hence **sharded** optimizers and **model-parallel** strategies are mandatory at this scale.

### Data Parallelism (DP)

Each of \(K\) devices holds a **full copy** of the model. Minibatch data is split across devices. Let \(g_k\) be the **mean gradient** over the local batch on rank \(k\). **Synchronous** data parallelism uses the **global** gradient:

\[
g = \frac{1}{K} \sum_{k=1}^{K} g_k.
\]

!!! math-intuition "In Plain English"
    Every GPU processes **different examples** but shares **identical parameters**. Averaging gradients makes the update match **larger-batch** training. Implementations use **all-reduce** to compute the average **without** a central parameter server in the common case.

**Pros:** straightforward scaling when communication is cheap relative to compute. **Cons:** **memory duplication**—each GPU stores the **entire** model and full optimizer state.

### Model Parallelism

**Pipeline parallelism** assigns **contiguous layer blocks** to **stages** on different devices. **Tensor parallelism** splits **weight matrices** within a layer across devices (**Megatron-LM** column and row parallel GEMMs). For a linear map \(Y = XW\), **column-parallel** \(W = [W_1 \mid W_2]\) yields concatenated outputs \([XW_1 \mid XW_2]\) without immediate communication depending on layout; **row-parallel** combinations require **all-reduce** of partial sums to form the next activation.

!!! math-intuition "In Plain English"
    **Tensor parallelism** splits **math** across GPUs so **no device stores the full weight matrix** for that layer. **Pipeline parallelism** splits **depth** so **no device stores every layer**, but introduces **idle time** (**bubbles**) when stages wait for work.

### FSDP (Fully Sharded Data Parallelism)

**FSDP** shards **parameters**, **gradients**, and **optimizer states** across ranks. A useful **per-rank** heuristic is that **steady-state** shard memory scales roughly **inversely** with **world size** \(K\) for the sharded tensors:

\[
M_{\text{FSDP}} \approx \frac{M_{\text{unsharded}}}{K} + M_{\text{overhead}}.
\]

!!! math-intuition "In Plain English"
    Instead of **\(K\)** redundant copies of the full model, each rank keeps **one slice** of each sharded tensor. For a given layer, ranks **all-gather** shards to build the full weight **just in time** for the forward pass, then **discard** full tensors according to implementation policy. You trade **communication** for **VRAM**.

!!! example "Worked Example: FSDP Memory Savings"
    Consider a **70B** parameter model with **FSDP** across **\(K = 64\)** GPUs.

    **FP16 weight shard per rank (order-of-magnitude):**

    \[
    \frac{70 \times 10^9 \times 2\ \text{B}}{64} \approx 2.19 \times 10^9\ \text{B} \approx 2.2\ \text{GB}.
    \]

    **Optimizer shard** using **12 bytes per parameter** scaled by \(1/64\):

    \[
    \frac{70 \times 10^9 \times 12\ \text{B}}{64} \approx 13.1\ \text{GB}.
    \]

    **Gradients** shard similarly to weights at FP16: another **~2.2 GB**. The sum is **~17.5 GB** before **activations** and **buffers**. Real runs land in **tens of gigabytes** per GPU for 70B-class models at moderate sequence lengths, which is why **80GB** devices are common for this regime.

### DeepSpeed ZeRO

**ZeRO-1** shards **optimizer states** only. **ZeRO-2** shards **optimizer states and gradients**. **ZeRO-3** shards **optimizer states, gradients, and parameters**. Let \(M_{\text{opt}}\) be optimizer memory without sharding. ZeRO-1 reduces per-rank optimizer memory roughly to:

\[
M_{\text{opt}}^{(1)} \approx \frac{M_{\text{opt}}}{K}.
\]

!!! math-intuition "In Plain English"
    ZeRO stages are **incremental sharding**. Stage 1 targets the case where **Adam moments** dominated VRAM. Stage 3 targets the case where **even parameters** cannot be replicated. **ZeRO-Offload** pushes tensors to **CPU** or **NVMe**, trading **latency** for **capacity**.

### Communication Primitives

Collective operations underpin distributed training:

- **All-reduce:** every rank contributes; **every rank** receives the same reduced result (often **sum** then **scale** for the mean gradient).
- **All-gather:** each rank holds a shard; after the collective, **each rank** has the full concatenated tensor.
- **Reduce-scatter:** each rank contributes pieces; each rank receives **one shard** of the reduced sum.
- **All-to-all:** personalized routing; appears in **Mixture-of-Experts** dispatch.

For **ring all-reduce** with payload size \(S\) bytes, a simplified **latency-bandwidth** model is:

\[
T_{\text{ring}} \approx (K-1)\alpha + \frac{(K-1)}{K}\frac{S}{\beta}
\]

where \(\alpha\) is per-hop latency and \(\beta\) is effective bandwidth.

!!! math-intuition "In Plain English"
    The **first term** dominates for **small** messages (**latency-bound** collectives). The **second term** dominates for **large** tensors (**bandwidth-bound**). Production frameworks tune **bucket sizes** to **overlap** collectives with **backward** computation.

**NCCL** provides optimized implementations on **NVLink** and **InfiniBand** topologies.

### Mixed Precision Training

Forward and backward passes often use **FP16** or **BF16**, while **master weights** stay **FP32**. **Loss scaling** multiplies the loss by \(s\) before backward to reduce **gradient underflow** in FP16:

\[
\tilde{\mathcal{L}} = s \cdot \mathcal{L},\qquad \tilde{g} = s \cdot g.
\]

The optimizer step uses **unscaled** gradients after converting to **FP32** and dividing by \(s\) in the implementation’s numerically safe path.

!!! math-intuition "In Plain English"
    FP16 has **limited dynamic range** in the mantissa; tiny gradients can **quantize to zero**. **Scaling the loss** scales **all gradients** up into representable range. **BF16** shares **exponent bits** with FP32, so many workloads need **less** or **no** loss scaling compared to FP16.

---

## Deep Dive

??? deep-dive "Deep Dive: When to Choose Tensor Parallel vs Pipeline Parallel"
    **Tensor parallel (TP)** splits individual operators across a **small** set of devices that are often **NVLink-connected** because each layer may require **low-latency** collectives. It **reduces per-device memory** for large matrices but **injects communication** into the **critical path** of each layer.

    **Pipeline parallel (PP)** splits **depth** across stages; each stage owns a **contiguous** subset of layers. It **scales to more devices** along the depth axis but introduces **pipeline bubbles** unless **many micro-batches** are in flight. **1F1B** scheduling reduces idle time compared with naive **GPipe** schedules.

    Large training stacks frequently combine **data parallel** outer loops, **tensor parallel within a node**, and **pipeline parallel across nodes**. Interview answers should tie choices to **topology**, **batch size**, and **bubble fraction**, not only parameter counts.

??? deep-dive "Deep Dive: Debugging Stragglers and NCCL Timeouts"
    **Stragglers** are ranks that run slower due to **thermal limits**, **faulty hardware**, or **local I/O**. **Synchronous** training waits for the **slowest** rank every step, **amplifying** jitter. Mitigations include **health checks**, **node exclusion**, and **elastic** job relaunch from **checkpoints**.

    **NCCL timeouts** often trace to **deadlocks** from **mismatched collective ordering**, **wrong tensor shapes**, or **network partitions**. Practical debugging: enable **NCCL debug logs**, verify **WORLD_SIZE** and **RANK**, confirm **CUDA_VISIBLE_DEVICES**, and reproduce on **two GPUs** before scaling out.

---

## Code

```python
"""
Minimal FSDP-style demonstration with PyTorch (multi-GPU requires torchrun).
Single-GPU or CPU falls back without distributed initialization.
Save as fsdp_demo.py and run:
  torchrun --nproc_per_node=2 fsdp_demo.py
"""
from __future__ import annotations

import os

import torch
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy


class TinyBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, 4 * dim)
        self.fc2 = nn.Linear(4 * dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln(x)
        h = self.fc2(torch.relu(self.fc1(h)))
        return x + h


class ToyLM(nn.Module):
    def __init__(self, dim: int, depth: int, vocab: int) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.blocks = nn.ModuleList(TinyBlock(dim) for _ in range(depth))
        self.head = nn.Linear(dim, vocab)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        x = self.embed(idx)
        for blk in self.blocks:
            x = blk(x)
        return self.head(x)


def main() -> None:
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if world > 1:
        torch.distributed.init_process_group(backend="nccl")
        device = torch.device("cuda", rank)
        torch.cuda.set_device(device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dim, depth, vocab = 256, 4, 8000
    model = ToyLM(dim, depth, vocab).to(device)

    if world > 1:
        model = FSDP(model, auto_wrap_policy=size_based_auto_wrap_policy)

    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    batch, seq = 8, 32
    x = torch.randint(0, vocab, (batch, seq), device=device)
    y = torch.randint(0, vocab, (batch, seq), device=device)

    logits = model(x)
    loss = torch.nn.functional.cross_entropy(
        logits.view(-1, vocab), y.view(-1)
    )
    loss.backward()
    opt.step()
    opt.zero_grad(set_to_none=True)

    if rank == 0:
        print("step loss", float(loss))

    if world > 1:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
```

**DeepSpeed ZeRO-3 JSON configuration sketch** (illustrative values; tune for your cluster):

```json
{
  "train_batch_size": 512,
  "gradient_accumulation_steps": 8,
  "optimizer": {
    "type": "AdamW",
    "params": {
      "lr": 0.0002,
      "betas": [0.9, 0.95],
      "eps": 1e-08,
      "weight_decay": 0.1
    }
  },
  "fp16": {
    "enabled": true,
    "loss_scale": 0,
    "loss_scale_window": 1000,
    "initial_scale_power": 12,
    "hysteresis": 2,
    "min_loss_scale": 1
  },
  "zero_optimization": {
    "stage": 3,
    "overlap_comm": true,
    "contiguous_gradients": true,
    "reduce_bucket_size": 500000000,
    "allgather_bucket_size": 500000000,
    "stage3_prefetch_bucket_size": 50000000,
    "stage3_param_persistence_threshold": 100000,
    "sub_group_size": 1000000000,
    "stage3_max_live_parameters": 1000000000,
    "stage3_max_reuse_distance": 1000000000,
    "stage3_gather_16bit_weights_on_model_save": true
  },
  "steps_per_print": 10,
  "wall_clock_breakdown": false
}
```

```python
"""
Write a DeepSpeed-compatible JSON config next to this script and validate JSON.
Run: python write_deepspeed_config.py
Requires: Python 3.9+ (stdlib only for this demo).
"""
from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    cfg = {
        "train_batch_size": 512,
        "gradient_accumulation_steps": 8,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": 0.0002,
                "betas": [0.9, 0.95],
                "eps": 1e-08,
                "weight_decay": 0.1,
            },
        },
        "fp16": {
            "enabled": True,
            "loss_scale": 0,
            "loss_scale_window": 1000,
            "initial_scale_power": 12,
            "hysteresis": 2,
            "min_loss_scale": 1,
        },
        "zero_optimization": {
            "stage": 3,
            "overlap_comm": True,
            "contiguous_gradients": True,
        },
    }
    out = Path(__file__).with_name("ds_config.generated.json")
    out.write_text(json.dumps(cfg, indent=2))
    # Round-trip parse to ensure valid JSON
    json.loads(out.read_text())
    print("Wrote", out.resolve())


if __name__ == "__main__":
    main()
```

---

## Interview Guide

!!! interview "FAANG-Level Questions"
    1. **Data parallel vs model parallel:** When is full replication acceptable, and when must you shard parameters?
    2. **All-reduce:** Describe ring all-reduce at a high level and whether it is bandwidth or latency bound for large tensors.
    3. **ZeRO stages:** Compare ZeRO-1, ZeRO-2, and ZeRO-3 in terms of memory savings and communication overhead.
    4. **FSDP:** How does FSDP differ from DDP, and what does sharding strategy control?
    5. **Pipeline bubbles:** What causes bubbles in GPipe-style schedules, and how does micro-batching mitigate them?
    6. **Tensor parallel:** Explain column versus row parallel for a linear layer and the required communication.
    7. **Mixed precision:** Why use FP32 master weights, and when does BF16 reduce the need for loss scaling compared to FP16?
    8. **Gradient accumulation:** How do you reach a large effective batch size when per-step memory is limited?
    9. **Checkpointing:** What additional complexity arises when saving ZeRO-3 checkpoints?
    10. **Fault tolerance:** How do large training jobs detect stragglers and recover from node loss?

!!! interview "Follow-up Probes"
    - “What NCCL environment variables have you tuned, and what symptoms motivated them?”
    - “How does activation checkpointing trade compute for memory?”
    - “Why might tensor parallel degree equal the number of GPUs per node?”
    - “Explain overlap of communication with backward pass in distributed optimizers.”

!!! key-phrases "Key Phrases to Use in Interviews"
    - “**Synchronous data parallelism** averages gradients with **all-reduce**.”
    - “**ZeRO shards optimizer states** to remove redundant copies across ranks.”
    - “**Tensor parallel** splits GEMMs within a layer; **pipeline parallel** splits depth across stages.”
    - “**Loss scaling** prevents **FP16 gradient underflow**.”
    - “**Ring all-reduce** has a **latency term** plus a **payload-over-bandwidth term**.”

---

## References

1. Shoeybi, M., Patwary, M., Puri, R., et al. **Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism.** arXiv:1909.08053 (2019).
2. Huang, Y., Cheng, Y., Bapna, A., et al. **GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism.** NeurIPS (2019).
3. Narayanan, D., Shoeybi, M., Casper, J., et al. **Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM.** SC (2021).
4. Rajbhandari, S., Rasley, J., Ruwase, O., He, Y. **ZeRO: Memory Optimizations Toward Training Trillion Parameter Models.** SC (2020).
5. Paszke, A., Gross, S., Massa, F., et al. **PyTorch: An Imperative Style, High-Performance Deep Learning Library.** NeurIPS (2019).
6. Micikevicius, P., Narang, S., Alben, J., et al. **Mixed Precision Training.** ICLR (2018).
7. NVIDIA **NCCL User Guide** — collective algorithms and environment variables (versioned documentation).
8. Lepikhin, D., Lee, H., Xu, Y., et al. **GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding.** JMLR (2021).
