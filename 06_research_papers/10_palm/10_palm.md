# PaLM: Scaling Language Modeling with Pathways

**Authors:** Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, and 60+ more  
**Year:** 2022 &nbsp;|&nbsp; **Venue:** arXiv / JMLR  
**Link:** [arXiv:2204.02311](https://arxiv.org/abs/2204.02311)

---

## TL;DR

PaLM trains a **540B parameter** dense Transformer on a large multilingual corpus using Google's **Pathways** infrastructure for efficient data-parallel and model-parallel training across 6,144 TPU v4 chips. The paper documents **emergent capabilities** — reasoning, code generation, translation — that appear abruptly at scale, alongside systematic evaluation on **BIG-Bench** and other benchmarks.

---

## Why This Paper Matters

PaLM is significant for several reasons:

1. **Scale demonstration:** Showed what 540B parameters can do with efficient infrastructure
2. **Emergence evidence:** Documented capabilities that appear discontinuously at scale
3. **Infrastructure blueprint:** Pathways-style parallelism is now standard for large-scale training
4. **Multilingual at scale:** Trained on diverse multilingual data, not just English
5. **BIG-Bench evaluation:** Set the standard for evaluating capabilities beyond traditional benchmarks

---

## Key Concepts Explained Simply

### Pathways Infrastructure

Training 540B parameters requires splitting the model across thousands of accelerators. Pathways orchestrates:

- **Data parallelism:** Same model on different data shards
- **Tensor parallelism:** Split individual layers across chips
- **Pipeline parallelism:** Different layers on different chips

PaLM used 2-way pod-level data parallelism with extensive tensor parallelism, achieving **57.8% hardware utilization** (impressive at this scale).

### Emergent Capabilities

Some abilities "emerge" — they're near-random at smaller scales and suddenly work at 540B:
- **Multi-step reasoning** (with chain-of-thought)
- **Joke explanation** (understanding humor)
- **Code generation** in multiple languages
- **Logical inference** chains

The emergence concept is controversial — some researchers argue it's a measurement artifact of using discrete metrics (accuracy) on smooth underlying improvements.

### BIG-Bench

**Beyond the Imitation Game Benchmark:** 150+ diverse tasks designed to probe capabilities beyond standard NLP benchmarks. Includes tasks like sarcasm detection, causal reasoning, logical deduction, and many others.

---

## The Math — Explained Step by Step

### Training Objective

Standard autoregressive cross-entropy loss:

\[
\mathcal{L}(\theta) = -\frac{1}{|\mathcal{B}|} \sum_{\mathbf{x} \in \mathcal{B}} \sum_{t=1}^{L} \log P_\theta(x_t \mid x_{<t})
\]

The same loss as GPT-2/3 — what changes is the **scale** at which it's computed.

### Throughput and Efficiency

**Model FLOPs utilization (MFU):**

\[
\text{MFU} = \frac{\text{Achieved FLOPs/sec}}{\text{Peak theoretical FLOPs/sec}}
\]

PaLM achieved **57.8% MFU** on 6,144 TPU v4 chips. This measures how well the training pipeline utilizes the hardware.

### Parallelism Strategies

For a model with \(N\) parameters across \(P\) chips:

**Data parallelism:**
- Each chip has a full model copy
- Batch is split across chips: \(B_{\text{local}} = B / P\)
- Gradients are all-reduced after each step
- Memory per chip: \(O(N)\)

**Tensor parallelism:**
- Individual weight matrices are split across chips
- Requires communication within each layer
- Reduces memory per chip: \(O(N/P)\)
- Adds latency from all-reduce within each forward pass

**Pipeline parallelism:**
- Different layers on different chips
- Forward/backward passes are pipelined as micro-batches
- "Bubble" overhead from pipeline startup/drain

---

## Python Implementation

```python
import numpy as np


def cross_entropy_loss(logits, targets):
    """Standard CE loss — same at any scale."""
    z = logits - np.max(logits, axis=-1, keepdims=True)
    log_probs = z - np.log(np.sum(np.exp(z), axis=-1, keepdims=True))
    loss = 0.0
    for t, tid in enumerate(targets):
        loss -= log_probs[t, tid]
    return loss / len(targets)


def data_parallel_shard(batch, num_shards):
    """Split a batch across data-parallel workers."""
    shard_size = len(batch) // num_shards
    shards = []
    for i in range(num_shards):
        start = i * shard_size
        end = start + shard_size
        shards.append(batch[start:end])
    return shards


def all_reduce_gradients(local_gradients):
    """
    Simulate all-reduce: average gradients across workers.
    local_gradients: list of gradient arrays from each worker
    """
    return np.mean(local_gradients, axis=0)


def tensor_parallel_linear(x, weight_shards):
    """
    Tensor-parallel linear layer: weight is split column-wise
    across devices, results are concatenated.
    """
    outputs = []
    for shard in weight_shards:
        outputs.append(x @ shard)
    return np.concatenate(outputs, axis=-1)


def pipeline_schedule(n_layers, n_micro_batches, n_stages):
    """
    Visualize pipeline parallelism schedule.
    Shows which stage processes which micro-batch at each time step.
    """
    layers_per_stage = n_layers // n_stages
    total_steps = n_micro_batches + n_stages - 1

    schedule = {}
    for step in range(total_steps):
        active = []
        for stage in range(n_stages):
            micro_batch = step - stage
            if 0 <= micro_batch < n_micro_batches:
                active.append(f"S{stage}:MB{micro_batch}")
            else:
                active.append(f"S{stage}:idle")
        schedule[step] = active
    return schedule


def compute_mfu(achieved_tflops, peak_tflops_per_chip, n_chips):
    """Model FLOPs Utilization."""
    peak_total = peak_tflops_per_chip * n_chips
    return achieved_tflops / peak_total


def estimate_training_cost(N, D, tflops_per_dollar_hour, utilization=0.5):
    """
    Estimate training cost in dollar-hours.
    N: parameters, D: tokens, assumes C ≈ 6ND FLOPs
    """
    total_flops = 6 * N * D
    effective_tflops = tflops_per_dollar_hour * utilization
    hours = total_flops / (effective_tflops * 1e12 * 3600)
    return hours


# --- Demo ---
if __name__ == "__main__":
    np.random.seed(42)

    # Data parallelism demo
    batch = list(range(32))
    shards = data_parallel_shard(batch, num_shards=4)
    print("Data parallelism:")
    for i, shard in enumerate(shards):
        print(f"  Worker {i}: {shard}")

    # All-reduce demo
    grads = [np.random.randn(4) for _ in range(4)]
    avg_grad = all_reduce_gradients(grads)
    print(f"\nAll-reduce: {len(grads)} workers → averaged gradient")

    # Tensor parallelism demo
    x = np.random.randn(3, 8)
    w1 = np.random.randn(8, 4)
    w2 = np.random.randn(8, 4)
    out = tensor_parallel_linear(x, [w1, w2])
    print(f"\nTensor parallel: input {x.shape} → output {out.shape}")

    # Pipeline schedule
    schedule = pipeline_schedule(n_layers=12, n_micro_batches=4, n_stages=3)
    print("\nPipeline schedule:")
    for step, active in schedule.items():
        print(f"  Step {step}: {' | '.join(active)}")

    # MFU calculation
    mfu = compute_mfu(
        achieved_tflops=180_000,
        peak_tflops_per_chip=275,
        n_chips=6144
    )
    print(f"\nModel FLOPs Utilization: {mfu:.1%}")

    # Training cost estimation
    hours = estimate_training_cost(
        N=540e9, D=780e9,
        tflops_per_dollar_hour=275,
        utilization=0.578
    )
    print(f"Estimated training: {hours:,.0f} GPU-hours")
```

---

## Interview Importance

PaLM is relevant for **systems interviews** about distributed training and for **emergence** debates. Less likely to be asked directly, but the concepts show up constantly.

### Difficulty Level: ⭐⭐⭐ (Medium)

---

## Interview Questions & Answers

### Q1: Name two distributed parallelism patterns and when each applies.

**Answer:**
1. **Data parallelism:** Each device has a full model copy, processes different data. Best when the model fits on a single device. Scale-out is easy but memory per device is constant. Communication: gradient all-reduce after each step.

2. **Tensor parallelism:** Individual layers/matrices are split across devices. Required when a single layer is too large for one device's memory. Lower memory per device but high communication overhead (all-reduce within each layer's forward/backward pass). Best within a node (high-bandwidth interconnect).

3. **Pipeline parallelism:** Different layers on different devices. Model is split into stages processed sequentially. Micro-batching amortizes the pipeline "bubble." Good for spanning nodes but introduces idle time.

### Q2: What does "emergence" mean in scaling papers, and why is it controversial?

**Answer:** Emergence refers to capabilities that appear **abruptly** at certain model scales — near-random performance at small sizes, then suddenly working at larger sizes. Examples: multi-step arithmetic, analogical reasoning.

**Why controversial:**
- Some researchers (Schaeffer et al., 2023) argue emergence is an artifact of **discrete metrics** (accuracy). When measured with continuous metrics (log-likelihood), improvement is smooth and predictable.
- Others argue the discrete threshold is real and meaningful — a task either works or it doesn't.
- The definition of "emergence" varies across papers, making claims hard to compare.

### Q3: How would you evaluate a multilingual LM beyond English-centric benchmarks?

**Answer:**
1. **Per-language evaluation:** Run benchmarks in each target language (not just translated English benchmarks)
2. **Cross-lingual transfer:** Test on tasks in languages not well-represented in training data
3. **Cultural appropriateness:** Evaluate for cultural biases and stereotypes per language
4. **Code-switching:** Test ability to handle mixed-language inputs
5. **Tokenizer efficiency:** Check tokens-per-word ratio across languages (Latin scripts vs. CJK vs. Arabic)
6. **Low-resource languages:** Specifically evaluate on languages with limited training data

---

## Connections to Other Papers

- **GPT-3** → PaLM scales further with better infrastructure
- **Chinchilla** → Later showed PaLM was likely undertrained (not enough tokens per parameter)
- **LLaMA** → Applied Chinchilla insights to create efficient open models
- **Gemini** → PaLM's successor with native multimodal capabilities
- **FLAN** → FLAN-PaLM adds instruction tuning on top of PaLM

---

## Key Takeaways for Quick Review

| Concept | Remember |
|---|---|
| Model size | 540B parameters (dense) |
| Infrastructure | Pathways on 6,144 TPU v4 chips |
| MFU | 57.8% hardware utilization |
| Key finding | Emergent capabilities at scale |
| Parallelism | Data + tensor + pipeline parallelism |
| Evaluation | BIG-Bench (150+ diverse tasks) |
| Limitation | Likely undertrained (Chinchilla insight) |
