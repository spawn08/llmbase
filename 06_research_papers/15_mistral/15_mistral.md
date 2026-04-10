# Mistral 7B

**Authors:** Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, and 5 more  
**Year:** 2023 &nbsp;|&nbsp; **Venue:** arXiv  
**Link:** [arXiv:2310.06825](https://arxiv.org/abs/2310.06825)

---

## TL;DR

Mistral 7B demonstrates that a **carefully trained 7B model** with **Sliding Window Attention (SWA)** and **Grouped-Query Attention (GQA)** can match or beat much larger models on many benchmarks. It emphasizes **efficient architecture** and **data/curriculum quality** over raw parameter count, strengthening the "small-but-mighty" open-weights narrative.

---

## Why This Paper Matters

Mistral 7B reshaped expectations about small models:

1. **7B matching 13B+:** Outperformed LLaMA-2 13B on most benchmarks
2. **Architectural innovations:** GQA + SWA became standard for efficient models
3. **KV cache efficiency:** GQA dramatically reduces the memory footprint during inference
4. **Foundation for Mixtral:** Led to the MoE model that further pushed efficiency
5. **Practical deployment:** A 7B model that runs on a single consumer GPU

---

## Key Concepts Explained Simply

### Grouped-Query Attention (GQA)

Standard multi-head attention (MHA) has separate key and value heads for each query head. GQA **groups** multiple query heads to share the same key and value heads:

- **MHA:** 32 query heads, 32 key heads, 32 value heads
- **MQA (multi-query):** 32 query heads, 1 key head, 1 value head
- **GQA:** 32 query heads, 8 key heads, 8 value heads (4 query heads per KV group)

Why? During inference, the **KV cache** stores key and value states for all past tokens. With GQA, you store 4× fewer KV entries, dramatically reducing memory and increasing batch size.

### Sliding Window Attention (SWA)

Instead of attending to **all** previous tokens (full causal attention), each token only attends to the last \(W\) tokens (e.g., \(W = 4096\)):

- **Full attention:** Token 10,000 attends to tokens 0-9,999 → \(O(N)\) KV cache per token
- **SWA:** Token 10,000 attends to tokens 5,904-9,999 → \(O(W)\) KV cache per token

But information can still flow long-range! Through **stacking layers**, token 10,000 can indirectly access information from token 0 through intermediate layers. With \(L\) layers and window size \(W\), the effective receptive field is \(L \times W\).

### Rolling Buffer KV Cache

With SWA, you don't need to store KV for all past tokens — only the last \(W\). This can be implemented as a **circular buffer**:

- Position \(i\) is stored at index \(i \mod W\) in the buffer
- Old entries are automatically overwritten
- KV cache memory is **fixed** regardless of sequence length

---

## The Math — Explained Step by Step

### Sliding Window Attention Mask

\[
A_{ij} = \begin{cases} \text{attend} & \text{if } i - W \leq j \leq i \\ 0 & \text{otherwise} \end{cases}
\]

The attention matrix is **banded** — each row has at most \(W + 1\) non-zero entries.

### Effective Receptive Field

With \(L\) Transformer layers and window size \(W\):

\[
\text{Receptive field at layer } l = l \times W
\]

At layer \(L\), a token can attend to information originating \(L \times W\) positions back.

For Mistral 7B: \(L = 32\), \(W = 4096\) → effective receptive field = \(32 \times 4096 = 131,072\) tokens.

### GQA Memory Savings

KV cache memory per token:

\[
\text{MHA:} \quad 2 \times n_{\text{heads}} \times d_{\text{head}} \times \text{sizeof(dtype)}
\]
\[
\text{GQA:} \quad 2 \times n_{\text{kv\_heads}} \times d_{\text{head}} \times \text{sizeof(dtype)}
\]

With 32 query heads and 8 KV heads: **4× memory reduction** in the KV cache.

### Complexity

- **Full attention:** \(O(N^2 d)\) compute, \(O(N)\) KV cache per token
- **SWA:** \(O(NW d)\) compute (linear in \(N\) for fixed \(W\)), \(O(W)\) KV cache per token

---

## Python Implementation

```python
import numpy as np


def stable_softmax(logits):
    z = logits - np.max(logits, axis=-1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=-1, keepdims=True)


def sliding_window_mask(seq_len, window_size):
    """Create a sliding window causal attention mask."""
    mask = np.zeros((seq_len, seq_len))
    for i in range(seq_len):
        start = max(0, i - window_size + 1)
        for j in range(start, i + 1):
            mask[i][j] = 1.0
    return mask


def full_causal_mask(seq_len):
    return np.tril(np.ones((seq_len, seq_len)))


def sliding_window_attention(Q, K, V, window_size):
    """Attention with sliding window mask."""
    N, d = Q.shape
    scores = (Q @ K.T) / np.sqrt(d)

    mask = sliding_window_mask(N, window_size)
    scores = np.where(mask == 0, -1e9, scores)

    attn = stable_softmax(scores)
    return attn @ V


def grouped_query_attention(Q, K, V, n_q_heads, n_kv_heads):
    """
    GQA: multiple query heads share key/value heads.
    Q: [seq_len, n_q_heads * d_head]
    K: [seq_len, n_kv_heads * d_head]
    V: [seq_len, n_kv_heads * d_head]
    """
    seq_len = Q.shape[0]
    d_head = Q.shape[1] // n_q_heads
    group_size = n_q_heads // n_kv_heads

    Q_heads = Q.reshape(seq_len, n_q_heads, d_head)
    K_heads = K.reshape(seq_len, n_kv_heads, d_head)
    V_heads = V.reshape(seq_len, n_kv_heads, d_head)

    outputs = []
    for q_idx in range(n_q_heads):
        kv_idx = q_idx // group_size  # Which KV head this Q head uses
        q = Q_heads[:, q_idx, :]  # [seq_len, d_head]
        k = K_heads[:, kv_idx, :]
        v = V_heads[:, kv_idx, :]

        scores = (q @ k.T) / np.sqrt(d_head)
        causal = np.tril(np.ones((seq_len, seq_len)))
        scores = np.where(causal == 0, -1e9, scores)
        attn = stable_softmax(scores)
        outputs.append(attn @ v)

    return np.concatenate(outputs, axis=-1)


class RollingKVCache:
    """Fixed-size circular buffer for KV cache with sliding window."""

    def __init__(self, window_size, d_model):
        self.window_size = window_size
        self.d_model = d_model
        self.k_cache = np.zeros((window_size, d_model))
        self.v_cache = np.zeros((window_size, d_model))
        self.position = 0

    def update(self, k_new, v_new):
        """Add new key/value to the rolling buffer."""
        idx = self.position % self.window_size
        self.k_cache[idx] = k_new
        self.v_cache[idx] = v_new
        self.position += 1

    def get_kv(self):
        """Get current valid keys and values."""
        valid_len = min(self.position, self.window_size)
        if self.position <= self.window_size:
            return self.k_cache[:valid_len], self.v_cache[:valid_len]
        start = self.position % self.window_size
        indices = [(start + i) % self.window_size for i in range(self.window_size)]
        return self.k_cache[indices], self.v_cache[indices]

    @property
    def memory_bytes(self):
        return 2 * self.window_size * self.d_model * 4  # float32


def compare_attention_types():
    """Compare memory for MHA, MQA, and GQA."""
    d_model = 4096
    d_head = 128
    n_q_heads = 32
    seq_len = 8192
    dtype_bytes = 2  # bfloat16

    configs = [
        ("MHA (32 KV heads)", 32),
        ("GQA (8 KV heads)", 8),
        ("GQA (4 KV heads)", 4),
        ("MQA (1 KV head)", 1),
    ]

    print(f"{'Config':<25} {'KV Cache':>12} {'Relative':>10}")
    print("-" * 50)
    for name, n_kv in configs:
        kv_bytes = 2 * n_kv * d_head * seq_len * dtype_bytes
        print(f"{name:<25} {kv_bytes/1e6:>10.1f}MB {n_kv/32*100:>9.0f}%")


def effective_receptive_field(n_layers, window_size):
    """Compute the effective receptive field through stacked layers."""
    return n_layers * window_size


# --- Demo ---
if __name__ == "__main__":
    np.random.seed(42)

    # Sliding window vs full attention
    seq_len, d = 16, 8
    Q = np.random.randn(seq_len, d)
    K = np.random.randn(seq_len, d)
    V = np.random.randn(seq_len, d)

    swa_mask = sliding_window_mask(seq_len, window_size=4)
    full_mask = full_causal_mask(seq_len)

    print("Sliding window mask (W=4):")
    print(swa_mask.astype(int))
    print(f"\nNon-zero entries: SWA={int(swa_mask.sum())}, Full={int(full_mask.sum())}")

    # GQA demo
    print("\n--- Grouped-Query Attention ---")
    n_q, n_kv, d_head = 8, 2, 16
    Q_gqa = np.random.randn(seq_len, n_q * d_head)
    K_gqa = np.random.randn(seq_len, n_kv * d_head)
    V_gqa = np.random.randn(seq_len, n_kv * d_head)
    out = grouped_query_attention(Q_gqa, K_gqa, V_gqa, n_q, n_kv)
    print(f"GQA output shape: {out.shape} (Q heads={n_q}, KV heads={n_kv})")

    # Memory comparison
    print("\n--- KV Cache Memory Comparison ---")
    compare_attention_types()

    # Rolling buffer demo
    print("\n--- Rolling KV Cache ---")
    cache = RollingKVCache(window_size=4, d_model=8)
    for i in range(8):
        cache.update(np.random.randn(8), np.random.randn(8))
        k, v = cache.get_kv()
        print(f"  Step {i}: buffer has {len(k)} entries, "
              f"memory = {cache.memory_bytes} bytes")

    # Effective receptive field
    erf = effective_receptive_field(n_layers=32, window_size=4096)
    print(f"\nEffective receptive field: 32 layers × 4096 window = {erf:,} tokens")
```

---

## Interview Importance

Mistral 7B is frequently discussed in interviews about **efficient model architectures** and **inference optimization**. GQA in particular is a must-know for serving systems.

### Difficulty Level: ⭐⭐⭐ (Medium)

---

## Interview Questions & Answers

### Q1: Compare GQA to MHA and MQA — what is saved at inference?

**Answer:**
- **MHA (Multi-Head Attention):** Each query head has its own KV head. KV cache = \(2 \times 32 \times d_{\text{head}} \times N\)
- **MQA (Multi-Query Attention):** All query heads share one KV head. KV cache = \(2 \times 1 \times d_{\text{head}} \times N\). Maximum savings but quality drops.
- **GQA:** Groups of query heads share KV heads (e.g., 4 queries per KV). KV cache = \(2 \times 8 \times d_{\text{head}} \times N\). 4× less than MHA with minimal quality loss.

At 32K sequence length with 32 heads and bfloat16: MHA = 512MB, GQA (8 groups) = 128MB, MQA = 16MB per layer.

### Q2: What is the throughput-quality trade-off of sliding window attention?

**Answer:** SWA reduces complexity from \(O(N^2)\) to \(O(NW)\), enabling:
- **Higher throughput:** Fewer attention computations per token
- **Fixed memory:** KV cache is bounded by \(W\), not \(N\)
- **Trade-off:** Tokens beyond \(W\) positions ago can only be accessed **indirectly** through stacked layers. Tasks requiring precise long-range retrieval (e.g., "repeat the first word") may suffer. For most natural language tasks, important information propagates well through layers.

### Q3: When would you still prefer a 70B dense model over a strong 7B?

**Answer:**
1. **Complex reasoning:** Multi-step math, code, and logical reasoning benefit from more parameters
2. **Knowledge-intensive tasks:** Larger models store more factual knowledge in their weights
3. **Low-resource languages:** Smaller models allocate less capacity to underrepresented languages
4. **High-stakes applications:** Medical, legal, financial tasks where accuracy matters more than latency
5. **When latency doesn't matter:** Batch processing where you can afford slower inference
6. **Instruction following:** Larger models generally follow complex, multi-constraint instructions better

### Q4: How does rolling buffer KV cache work?

**Answer:** Instead of an ever-growing KV cache, SWA uses a **fixed-size circular buffer** of size \(W\). Position \(i\) is stored at index \(i \bmod W\). When the buffer is full, new entries overwrite the oldest ones. This means:
- Memory is **constant** regardless of sequence length
- No garbage collection needed — old entries are naturally recycled
- The GPU can pre-allocate the exact buffer size needed

---

## Connections to Other Papers

- **LLaMA** → Mistral builds on LLaMA's architecture (RMSNorm, SwiGLU, RoPE)
- **FlashAttention** → Mistral uses FlashAttention-2 for efficient computation
- **Mixtral** → Extends Mistral 7B with Mixture of Experts
- **Chinchilla** → Mistral's strong 7B performance validates efficient training

---

## Key Takeaways for Quick Review

| Concept | Remember |
|---|---|
| Model size | 7B parameters |
| Key innovations | GQA + Sliding Window Attention |
| GQA | Multiple query heads share KV heads → 4× less KV cache |
| SWA window | 4,096 tokens per layer |
| Effective context | 32 layers × 4,096 = 131K tokens via stacking |
| Rolling buffer | Fixed-size KV cache using circular buffer |
| Key result | 7B matching LLaMA-2 13B and beating LLaMA-1 34B |
